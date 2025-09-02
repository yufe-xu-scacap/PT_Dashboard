import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import io
from typing import Dict, Optional, List, Tuple
import exchange_calendars as xcals
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

# --- Streamlit Page Configuration ---
st.html("""
  <style>
    [alt=Logo] {
      height: 4.5rem;
    }
  </style>
        """)
st.logo("data/logo.png", size="large")
st.set_page_config(layout="wide", page_title="Daily PnL Report Generation")
st.title("📈 Daily PnL Report Generation")


def update_log(message: str, log_list: List[str], log_placeholder, level: str = "INFO"):
    """Helper function to update the log list and the streamlit placeholder."""
    berlin_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%H:%M:%S')
    log_list.append(f"[{berlin_time}] {level}: {message}")
    # Display the last 15 log messages
    log_placeholder.markdown(f"```\n{' \n'.join(log_list[-15:])}\n```")


def get_reports(date: str, bearer_token: str, log_list: List[str], log_placeholder) -> Dict[str, pd.DataFrame]:
    """
    Retrieves daily and historical reports, splitting trades data by execution time.
    - T-0 trades are fetched from 18:06-18:10.
    - T-1 to T-4 trades are fetched from 22:09-22:12.
    - All trades data is split: before 16:00 (df_trades) and after 16:00 (df_trades_late).
    """
    date_str = str(date)
    headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
    report_dataframes = {}

    def _fetch_report(d_str: str, report_config: dict, start_time_str: str, end_time_str: str) -> Optional[
        pd.DataFrame]:
        """Internal helper to fetch a single report within a time window."""
        try:
            start_dt = datetime.strptime(f"{d_str}{start_time_str}", "%Y%m%d%H%M")
            end_dt = datetime.strptime(f"{d_str}{end_time_str}", "%Y%m%d%H%M")
            current_dt = start_dt
        except ValueError:
            update_log(f"Invalid date/time format for report fetching on {d_str}.", log_list, log_placeholder, "ERROR")
            return None

        while current_dt <= end_dt:
            time_str = current_dt.strftime("%H%M")
            server_filename = report_config["name_format"].format(date=d_str, time=time_str,
                                                                  report_name=report_config["report_name"])
            payload = {'key': f'fis/{server_filename}'}
            try:
                response = requests.post(
                    'https://franz.agent-tool.scalable.capital/agent/s3-access/buckets/prod-market-making.trade-reports.scalable-fis/download',
                    headers=headers, json=payload, timeout=20)
                if response.status_code in [200, 201]:
                    update_log(f"✅ Found and retrieved: {server_filename}", log_list, log_placeholder, "SUCCESS")
                    try:
                        return pd.read_csv(io.BytesIO(response.content))
                    except Exception as e:
                        update_log(f"Failed to parse CSV content for {server_filename}: {e}", log_list, log_placeholder,
                                   "ERROR")
                        return None
            except requests.exceptions.RequestException as e:
                pass
            current_dt += timedelta(minutes=1)
        return None

    xetr_cal = xcals.get_calendar("XETR")
    current_date_ts = pd.to_datetime(date_str).normalize()

    # --- P&L Report Fetching ---
    update_log(f"Searching for P&L reports...", log_list, log_placeholder)
    df_t0 = _fetch_report(date_str,
                          {"report_name": "ProfitandLoss", "name_format": "{date}-{time}FIS_EOD_{report_name}.csv"},
                          "1803", "1810")
    if df_t0 is not None:
        report_dataframes['ProfitandLoss'] = df_t0.copy()
    else:
        update_log(f"P&L report for {date_str} (T-0) not found.", log_list, log_placeholder, "ERROR")

    prev_day_ts = current_date_ts
    for i in range(1, 6):
        prev_day_ts = xetr_cal.previous_session(prev_day_ts)
        prev_day_str = prev_day_ts.strftime('%Y%m%d')
        df = _fetch_report(prev_day_str,
                           {"report_name": "ProfitandLoss", "name_format": "{date}-{time}FIS_EOD_{report_name}.csv"},
                           "1805", "1810")
        if df is not None:
            report_dataframes[f"ProfitandLoss_T-{i}"] = df.copy()
        else:
            update_log(f"P&L report for {prev_day_str} (T-{i}) not found.", log_list, log_placeholder, "ERROR")

    # --- Trades Report Fetching and Splitting Logic ---
    update_log(f"Searching for Trades reports and splitting by time...", log_list, log_placeholder)

    # Fetch T-0 Trades report
    trades_df_raw = _fetch_report(date_str,
                                  {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"},
                                  "1806", "1810")
    if trades_df_raw is not None and 'Execution Time' in trades_df_raw.columns:
        trades_df_raw['Execution Time'] = pd.to_datetime(trades_df_raw['Execution Time'], errors='coerce')
        trades_df_raw.dropna(subset=['Execution Time'], inplace=True)
        report_dataframes['Trades'] = trades_df_raw[trades_df_raw['Execution Time'].dt.hour < 16].copy()
        report_dataframes['Trades_late'] = trades_df_raw[trades_df_raw['Execution Time'].dt.hour >= 16].copy()
        update_log(
            f"Split T-0 trades: {len(report_dataframes['Trades'])} early, {len(report_dataframes['Trades_late'])} late.",
            log_list, log_placeholder)
    else:
        update_log(f"Trades report for {date_str} (T-0) not found or is invalid.", log_list, log_placeholder, "WARNING")

    # Fetch historical trades reports
    prev_day_ts = current_date_ts
    for i in range(1, 5):
        prev_day_ts = xetr_cal.previous_session(prev_day_ts)
        prev_day_str = prev_day_ts.strftime('%Y%m%d')
        df_trade_hist_raw = _fetch_report(prev_day_str,
                                          {"report_name": "Trades",
                                           "name_format": "{date}{time}FIS_EOD_{report_name}.csv"},
                                          "2209", "2212")

        if df_trade_hist_raw is not None and 'Execution Time' in df_trade_hist_raw.columns:
            df_trade_hist_raw['Execution Time'] = pd.to_datetime(df_trade_hist_raw['Execution Time'], errors='coerce')
            df_trade_hist_raw.dropna(subset=['Execution Time'], inplace=True)
            report_dataframes[f"Trades_T-{i}"] = df_trade_hist_raw[
                df_trade_hist_raw['Execution Time'].dt.hour < 16].copy()
            report_dataframes[f"Trades_late_T-{i}"] = df_trade_hist_raw[
                df_trade_hist_raw['Execution Time'].dt.hour >= 16].copy()
            update_log(
                f"Split T-{i} trades: {len(report_dataframes[f'Trades_T-{i}'])} early, {len(report_dataframes[f'Trades_late_T-{i}'])} late.",
                log_list, log_placeholder)
        else:
            update_log(f"Trades report for {prev_day_str} (T-{i}) not found or is invalid.", log_list, log_placeholder,
                       "WARNING")

    update_log("All search operations complete.", log_list, log_placeholder)
    return report_dataframes


def create_table_image(
        df: pd.DataFrame,
        width: int,
        height: int,
        col_widths: List[int],
        align_header: List[str],
        align_cells: List[str],
        # Add this new optional parameter for styling
        row_styles: Optional[Dict[str, Dict[str, str]]] = None,
        table_width_fraction: float = 1.0
):
    """
    Generates a PNG image for a DataFrame table, with conditional row styling.
    """
    if row_styles is None:
        row_styles = {}

    # Define default colors
    default_fill_color = 'white'
    default_font_color = '#101112' # Dark grey

    # Generate lists of colors for each row based on the styling rules
    row_fill_colors = [
        row_styles.get(row_name, {}).get('fill_color', default_fill_color) for row_name in df.index
    ]
    row_font_colors = [
        row_styles.get(row_name, {}).get('font_color', default_font_color) for row_name in df.index
    ]

    # The colors need to be applied to every cell in the row.
    # We create a list of lists, one for each column in the table.
    num_table_cols = len(df.columns) + 1  # +1 for the index column
    final_fill_colors = [row_fill_colors] * num_table_cols
    final_font_colors = [row_font_colors] * num_table_cols

    fig = go.Figure(data=[go.Table(
        domain=dict(x=[0, table_width_fraction], y=[0, 1]),
        header=dict(
            values=[f"<b>{df.index.name}</b>"] + list(df.columns),
            fill_color='#A8D0CA',
            align=align_header,
            font=dict(color='#101112', size=18),
            height=35
        ),
        cells=dict(
            values=[df.index.tolist()] + [df[col] for col in df.columns],
            fill_color=final_fill_colors,  # Use the new color list
            align=align_cells,
            font=dict(color=final_font_colors, size=16),  # Use the new font color list
            height=28
        ),
        columnwidth=col_widths
    )])

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
    )
    return pio.to_image(fig, format='png', width=width, height=height, scale=1.0)


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def create_pnl_plot(pnl_data, overnight_data, win_rate_data, pnl_turnover_data, df_date, width: int, height: int):
    """
    Generates the final PnL Breakdown plot with intelligent, compact, non-overlapping labels
    and perfectly aligned zero-axes.
    """
    # --- DATA PREPARATION ---
    plot_data_list = [
        {'Term': f'T-{i}', 'Intraday_PnL': pnl_data.get(f'T-{i}', 0), 'Overnight_PnL': overnight_data.get(f'T-{i}', 0)}
        for i in range(4, 0, -1)]
    plot_data_list.append({'Term': 'T-0', 'Intraday_PnL': pnl_data.get('T-0', 0), 'Overnight_PnL': 0})
    df_plot = pd.DataFrame(plot_data_list).merge(df_date, on='Term')
    df_win_rate = pd.DataFrame(win_rate_data)
    df_plot = df_plot.merge(df_win_rate, on='Term')
    df_plot = df_plot.merge(pnl_turnover_data, on='Term')
    df_plot['Total_PnL'] = df_plot['Intraday_PnL'] + df_plot['Overnight_PnL']
    df_plot['DateObj'] = pd.to_datetime(df_plot['Date'], format='%Y%m%d')
    df_plot = df_plot.sort_values(by='DateObj').reset_index(drop=True)
    df_plot['DateLabel'] = df_plot['DateObj'].dt.strftime('%Y-%m-%d')

    # --- PLOT CREATION ---
    fig = go.Figure()

    # --- TRACE DEFINITIONS ---
    fig.add_trace(
        go.Scatter(x=df_plot['DateLabel'], y=df_plot['Intraday_PnL'], name='Primary Session PnL (lhs)',
                   line=dict(color='#6BDBCB'),
                   fillcolor='rgba(107, 219, 203, 0.6)', mode='lines+markers', fill='tozeroy',
                   hovertemplate='Intraday: %{y:,.0f} EUR<extra></extra>'))

    x_fill_coords = list(df_plot['DateLabel'][:-1]) + list(df_plot['DateLabel'][:-1][::-1])
    y_fill_coords = list(df_plot['Total_PnL'][:-1]) + list(df_plot['Intraday_PnL'][:-1][::-1])
    fig.add_trace(
        go.Scatter(x=x_fill_coords, y=y_fill_coords, mode='none', fill='toself', fillcolor='rgba(247, 152, 128, 0.6)',
                   showlegend=False, hoverinfo='none'))

    line_data = df_plot['Total_PnL'].copy().astype(object)
    line_data.iloc[-1] = None
    overnight_pnl_for_hover = df_plot['Overnight_PnL'].copy().astype(object)
    overnight_pnl_for_hover.iloc[-1] = None
    fig.add_trace(
        go.Scatter(x=df_plot['DateLabel'], y=line_data, name='Late Session PnL (lhs)', line=dict(color='#F79880'),
                   mode='lines+markers',
                   hovertemplate='Overnight: %{customdata:,.0f} EUR<br>Total: %{y:,.0f} EUR<extra></extra>',
                   customdata=overnight_pnl_for_hover))

    fig.add_trace(go.Scatter(x=df_plot['DateLabel'], y=df_plot['Win_Rate'], name='% of Profitable Instruments (rhs)',
                             line=dict(color='#FFC880', width=2, dash='dash'),
                             marker=dict(symbol='x-thin', size=20, color='#FFC880'),
                             mode='lines+markers', yaxis='y2',
                             hovertemplate='Win Rate: %{y:.1%}<extra></extra>'))

    fig.add_trace(go.Scatter(
        x=df_plot['DateLabel'], y=df_plot['pnl_turnover_early_bps'], name='Primary Session PnL in bps (rhs)',
        line=dict(color='#371d76', width=2.5, dash='dot'),
        mode='lines+markers', yaxis='y3',
        marker=dict(symbol='cross', size=10),
        hovertemplate='Intraday PnL/Turnover: %{y:.2f} bps<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_plot['DateLabel'], y=df_plot['pnl_turnover_late_bps'], name='Late Session PnL in bps (rhs)',
        line=dict(color='#ABB6FF', width=2.5, dash='dot'),
        mode='lines+markers', yaxis='y3',
        marker=dict(symbol='cross', size=10),
        hovertemplate='Late Session PnL/Turnover: %{y:.2f} bps<extra></extra>'
    ))

    ### MODIFICATION START ###
    # --- AXIS RANGE CALCULATION FOR ALIGNED ZERO-LINES ---

    # 1. Gather all data points for EUR and BPS axes
    all_eur_values = pd.concat([df_plot['Total_PnL'].dropna(), df_plot['Intraday_PnL'].dropna()])
    all_bps_values = pd.concat([df_plot['pnl_turnover_early_bps'], df_plot['pnl_turnover_late_bps']]).dropna()

    # 2. Determine the data min/max, defaulting to 0 if empty
    eur_min = all_eur_values.min() if not all_eur_values.empty else 0
    eur_max = all_eur_values.max() if not all_eur_values.empty else 0
    bps_min = all_bps_values.min() if not all_bps_values.empty else 0
    bps_max = all_bps_values.max() if not all_bps_values.empty else 0

    # 3. If all data is non-negative, the axes simply start at 0.
    if eur_min >= 0 and bps_min >= 0:
        negative_proportion = 0
    else:
        # Calculate the proportion of the range that is negative for each axis
        eur_range = eur_max - eur_min if (eur_max - eur_min) > 0 else 1
        bps_range = bps_max - bps_min if (bps_max - bps_min) > 0 else 1

        eur_neg_prop = abs(min(0, eur_min)) / eur_range
        bps_neg_prop = abs(min(0, bps_min)) / bps_range

        # The master proportion is the LARGER of the two, ensuring both axes have enough negative space
        negative_proportion = max(eur_neg_prop, bps_neg_prop)

    # 4. Calculate final axis ranges using the master proportion
    # Add 25% padding to the top for labels and visual space
    y_axis_max = eur_max * 1.25 if eur_max > 0 else 1000
    bps_axis_max = bps_max * 1.25 if bps_max > 0 else 5

    if negative_proportion == 0:
        y_axis_min = 0
        bps_axis_min = 0
    else:
        # This formula calculates the required minimum value to achieve the desired negative_proportion
        # of the total axis range, given a fixed maximum value.
        y_axis_min = (negative_proportion * y_axis_max) / (negative_proportion - 1)
        bps_axis_min = (negative_proportion * bps_axis_max) / (negative_proportion - 1)

    bps_range_final = [bps_axis_min, bps_axis_max]

    ### MODIFICATION END ###

    # --- FINAL ANNOTATION LOGIC ---
    # This logic now uses the correctly calculated axis ranges
    y_range_for_norm = y_axis_max - y_axis_min
    bps_range_for_norm = bps_axis_max - bps_axis_min
    for i, date in enumerate(df_plot['DateLabel']):
        labels_to_plot = []

        # Gather labels for the current date
        pnl_val = df_plot['Intraday_PnL'].iloc[i]
        labels_to_plot.append(
            {'text': f'{pnl_val / 1000:,.1f}k', 'value': pnl_val, 'yref': 'y1', 'font': {'color': '#101112'}})
        if i < len(df_plot) - 1:
            total_pnl_val = df_plot['Total_PnL'].iloc[i]
            labels_to_plot.append({'text': f'{total_pnl_val / 1000:,.1f}k', 'value': total_pnl_val, 'yref': 'y1',
                                   'font': {'color': '#101112'}})
        early_bps_val = df_plot['pnl_turnover_early_bps'].iloc[i]
        if pd.notna(early_bps_val):
            labels_to_plot.append(
                {'text': f'{early_bps_val:.1f}bp', 'value': early_bps_val, 'yref': 'y3', 'font': {'color': '#371d76'}})
        late_bps_val = df_plot['pnl_turnover_late_bps'].iloc[i]
        if pd.notna(late_bps_val):
            labels_to_plot.append(
                {'text': f'{late_bps_val:.1f}bp', 'value': late_bps_val, 'yref': 'y3', 'font': {'color': '#371d76'}})

        if not labels_to_plot: continue

        # Add normalized y-values and sort labels from highest to lowest
        for label in labels_to_plot:
            if label['yref'] == 'y1':
                norm_y = (label['value'] - y_axis_min) / y_range_for_norm if y_range_for_norm else 0
            else:
                norm_y = (label['value'] - bps_axis_min) / bps_range_for_norm if bps_range_for_norm else 0
            label['normalized_y'] = norm_y
        labels_to_plot.sort(key=lambda x: x['normalized_y'], reverse=True)

        # --- Intelligent Separation Algorithm ---
        final_shifts = [0] * len(labels_to_plot)
        MIN_SEPARATION = 0.04  # Minimum separation in normalized units (4% of the axis)

        # 1. Initial placement: try to keep labels close and alternate
        for j, label in enumerate(labels_to_plot):
            direction = 'up' if j % 2 == 0 else 'down'
            if label['normalized_y'] < 0.2: direction = 'up'
            if label['normalized_y'] > 0.8: direction = 'down'
            final_shifts[j] = 20 if direction == 'up' else -20

        # 2. Nudge labels apart to resolve overlaps
        for _ in range(5):  # Run a few passes to let labels settle
            for j in range(len(labels_to_plot) - 1):
                label_a = labels_to_plot[j]
                label_b = labels_to_plot[j + 1]

                pos_a = label_a['normalized_y'] + (final_shifts[j] / (height * 0.8))
                pos_b = label_b['normalized_y'] + (final_shifts[j + 1] / (height * 0.8))
                separation = abs(pos_a - pos_b)

                if separation < MIN_SEPARATION:
                    push_amount = (MIN_SEPARATION - separation) * (height * 0.8) / 2
                    final_shifts[j] += push_amount
                    final_shifts[j + 1] -= push_amount

        # Horizontal alignment for endpoints
        if i == 0:
            x_anchor, x_shift = 'left', 10
        elif i == len(df_plot) - 1:
            x_anchor, x_shift = 'right', -10
        else:
            x_anchor, x_shift = 'center', 0

        # Add an annotation for each label with its final calculated shift
        for j, label in enumerate(labels_to_plot):
            fig.add_annotation(
                x=date, y=label['value'], yref=label['yref'], text=label['text'],
                showarrow=False, font={**label['font'], 'size': 18, 'family': 'Arial, sans-serif'},
                xanchor=x_anchor, xshift=x_shift,
                yshift=final_shifts[j]
            )

    # --- LAYOUT UPDATING ---
    fig.update_layout(
        title=dict(text='Daily PnL Breakdown (T-4 to T-0)', y=0.95, x=0.5, xanchor='center', yanchor='top',
                   font=dict(size=24, color='#101112')),
        plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=150, l=80, r=80),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, font=dict(size=18),
                    bgcolor='rgba(255,255,255,0.7)', bordercolor='#EAEAEA', borderwidth=1),
        xaxis=dict(domain=[0, 0.94], type='category', title_text='', tickfont=dict(size=20), showgrid=False,
                   showline=True, linewidth=1, linecolor='#101112', mirror=True, range=[0, len(df_plot) - 1]),
        yaxis=dict(title=dict(text='Profit and Loss (EUR)', standoff=15), tickformat='~s',
                   range=[y_axis_min, y_axis_max], title_font=dict(size=22), tickfont=dict(size=20), showgrid=False,
                   zeroline=True, zerolinewidth=1, zerolinecolor='grey',
                   showline=True, linewidth=1, linecolor='#101112', mirror=True),
        yaxis2=dict(title='Win Rate', overlaying='y', side='right', range=[0, 1], tickformat='.0%', showgrid=False,
                    showline=False, anchor='x', title_font=dict(color='#FFC880', size=22),
                    tickfont=dict(color='#FFC880', size=20)),
        yaxis3=dict(title=dict(text='PnL in bps', standoff=15), range=bps_range_final, overlaying='y', side='right',
                    showgrid=False, zeroline=True, zerolinewidth=1, zerolinecolor='grey',
                    showline=True, linewidth=2, linecolor='#371d76', anchor='free', position=1.0,
                    title_font=dict(color='#371d76', size=22), tickfont=dict(color='#371d76', size=20), ticks="outside",
                    minor=dict(ticks="outside", ticklen=5, tickcolor="black", showgrid=False, griddash='dot',
                               gridcolor='LightGrey'))
    )
    # The pio.to_image call is kept as is, assuming you have the necessary dependencies (kaleido) installed.
    # If running in an environment without them, you might want to return `fig` and call `fig.show()`.
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)

def create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date, width: int, height: int):
    """
    Generates the Turnover Breakdown plot with a design consistent with the PnL chart.
    """
    plot_data_list = [{'Term': f'T-{i}' if i > 0 else 'T-0',
                       'Total_Turnover': total_turnover_data.get(f'T-{i}' if i > 0 else 'T-0', 0),
                       'Accepted_RFQ_Turnover': accepted_turnover_data.get(f'T-{i}' if i > 0 else 'T-0', 0)} for i in
                      range(4, -1, -1)]
    df_plot = pd.DataFrame(plot_data_list)
    df_plot = pd.merge(df_plot, df_date, on='Term')
    df_plot['Hedge_Turnover'] = df_plot['Total_Turnover'] - df_plot['Accepted_RFQ_Turnover']
    df_plot['DateObj'] = pd.to_datetime(df_plot['Date'], format='%Y%m%d')
    df_plot = df_plot.sort_values(by='DateObj').reset_index(drop=True)
    df_plot['DateLabel'] = df_plot['DateObj'].dt.strftime('%Y-%m-%d')
    df_plot['Accepted_RFQ_Turnover_M'] = df_plot['Accepted_RFQ_Turnover'] / 1_000_000
    df_plot['Hedge_Turnover_M'] = df_plot['Hedge_Turnover'] / 1_000_000
    df_plot['Total_Turnover_M'] = df_plot['Total_Turnover'] / 1_000_000
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot['DateLabel'], y=df_plot['Accepted_RFQ_Turnover_M'], name='Accepted RFQ Turnover',
                         marker_color='#6BDBCB', text=df_plot['Accepted_RFQ_Turnover_M'], textposition='inside',
                         insidetextanchor='middle', texttemplate='%{text:.1f}M',
                         textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
                         hovertemplate='Accepted RFQ: %{customdata:,.0f} EUR<extra></extra>',
                         customdata=df_plot['Accepted_RFQ_Turnover'].values))
    fig.add_trace(
        go.Bar(x=df_plot['DateLabel'], y=df_plot['Hedge_Turnover_M'], name='Hedge Turnover', marker_color='#a8d0ca',
               text=df_plot['Hedge_Turnover_M'], textposition='inside', insidetextanchor='middle',
               texttemplate='%{text:.1f}M', textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
               hovertemplate='Hedge: %{customdata:,.0f} EUR<extra></extra>',
               customdata=df_plot['Hedge_Turnover'].values))
    fig.add_trace(
        go.Bar(x=df_plot['DateLabel'], y=[0] * len(df_plot), text=df_plot['Total_Turnover_M'], textposition='outside',
               texttemplate='%{text:.1f}M', textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
               cliponaxis=False, showlegend=False, hoverinfo='none'))
    fig.update_layout(barmode='stack',
                      title=dict(text='Daily Turnover Breakdown (T-4 to T-0)', y=0.95, x=0.5, xanchor='center',
                                 yanchor='top', font=dict(size=24, color='#101112', family="Arial, sans-serif")),
                      plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
                      font=dict(family="Arial, sans-serif", color='#101112'), margin=dict(t=80, b=80, l=80, r=80),
                      legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02, font=dict(size=20),
                                  bgcolor='rgba(255,255,255,0.7)', bordercolor='#EAEAEA', borderwidth=1),
                      xaxis=dict(type='category', title_text='', tickfont=dict(size=20), showgrid=False, showline=True,
                                 linewidth=1, linecolor='#101112', mirror=True, range=[-0.5, len(df_plot) - 0.5]),
                      yaxis=dict(title='Turnover (M EUR)', range=[0, df_plot['Total_Turnover_M'].max() * 1.15],
                                 title_font=dict(size=22), tickfont=dict(size=20), showgrid=False, showline=True,
                                 linewidth=1, linecolor='#101112', mirror=True))
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)


def create_top_performers_plot(df_PnL, df_trades, df_Instrument, width: int, height: int):
    """
    Generates the Top/Bottom Performers plot using Plotly, matching the standard design.
    """
    df_turnover = df_trades['Premium'].abs().groupby(df_trades['Instrument']).sum().rename('Turnover (€)')
    df_winners_pnl = df_PnL[~df_PnL['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
        'BTPLD'].sum().nlargest(10).rename('Total PnL (€)')
    df_losers_pnl = df_PnL[~df_PnL['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
        'BTPLD'].sum().nsmallest(10).rename('Total PnL (€)')
    df_combined_pnl = pd.concat([df_winners_pnl, df_losers_pnl])
    df_top_performers = pd.merge(left=df_combined_pnl.to_frame(), right=df_Instrument, left_index=True, right_on='ISIN',
                                 how='left').merge(right=df_turnover, left_on='ISIN', right_index=True,
                                                   how='left').rename(columns={'Name': 'Security Name'}).sort_values(
        by='Total PnL (€)', ascending=False).reset_index(drop=True)
    df_top_performers = df_top_performers[['ISIN', 'Security Name', 'Instrument Type', 'Turnover (€)', 'Total PnL (€)']]
    df_plot = df_top_performers.iloc[::-1].copy()

    # Ensure 'Security Name' is a string to prevent errors, filling missing names
    df_plot['Security Name'] = df_plot['Security Name'].fillna('Unknown Instrument').astype(str)

    df_plot['Abs_PnL'] = df_plot['Total PnL (€)'].abs()
    df_plot['Label'] = [f"{pnl:,.0f}€ (Turnover: {turnover:,.0f}€)" for pnl, turnover in
                        zip(df_plot['Total PnL (€)'], df_plot['Turnover (€)'])]

    ### --- REMOVED --- ###
    # We no longer need to truncate the names.
    # df_plot['Security Name Short'] = ...
    ### --- END REMOVED --- ###

    ### --- NEW DYNAMIC MARGIN CALCULATION --- ###
    try:
        # Find the length of the longest security name
        longest_name_length = df_plot['Security Name'].str.len().max()
        # Calculate margin: base padding + (pixels per character * length)
        # We estimate ~8 pixels per character for the font size used.
        left_margin = 70 + (longest_name_length * 8)
    except (ValueError, TypeError):
        # Fallback to a default margin if calculation fails (e.g., no data)
        left_margin = 250
    ### --- END NEW --- ###

    colors = ['#6BDBCB' if pnl > 0 else '#F79880' for pnl in df_plot['Total PnL (€)']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['Abs_PnL'],
        y=df_plot['Security Name'],  ### --- MODIFIED --- ### Use the original, full name for the Y-axis
        orientation='h',
        marker_color=colors,
        text=df_plot['Label'],
        textposition='auto',
        textfont=dict(size=16, family="Arial, sans-serif"),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>ISIN: %{customdata[1]}<br>PnL: %{customdata[2]:,.0f}€<br>Turnover: %{customdata[3]:,.0f}€<extra></extra>'),
        customdata=df_plot[['Security Name', 'ISIN', 'Total PnL (€)', 'Turnover (€)']].values))

    fig.update_layout(
        title=dict(text='Top 10 Winners & Losers by PnL', y=0.95, x=0.5, xanchor='center', yanchor='top',
                   font=dict(size=24, color='#101112', family="Arial, sans-serif")), plot_bgcolor='white',
        paper_bgcolor='white', font=dict(family="Arial, sans-serif", color='#101112'),
        ### --- MODIFIED --- ### Use the new dynamic left margin
        margin=dict(t=80, b=80, l=left_margin, r=40),
        showlegend=False,
        xaxis=dict(title='Daily PnL (€)', range=[0, df_plot['Abs_PnL'].max() * 1.05], title_font=dict(size=22),
                   tickfont=dict(size=20), tickformat=',.0f', showgrid=False, showline=True, linewidth=1,
                   linecolor='#101112', mirror=True),
        yaxis=dict(title='', type='category', tickfont=dict(size=16), showline=True, linewidth=1, linecolor='#101112',
                   mirror=True))
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)

### NEW ###
def process_and_display_report(report_data: Dict[str, pd.DataFrame], report_date: datetime, total_instrument: int, log_list: List[str], log_placeholder, pnl_override: Optional[Dict[str, float]] = None):
    """
    Takes the fetched report data, performs all calculations, and displays the results.
    Can optionally apply a PnL override for a specific ISIN.
    """
    try:
        # --- Define plot and table dimensions ---
        PNL_PLOT_WIDTH, PNL_PLOT_HEIGHT = 1600, 900
        TURNOVER_PLOT_WIDTH, TURNOVER_PLOT_HEIGHT = 1600, 900
        TOP_PERF_PLOT_WIDTH, TOP_PERF_PLOT_HEIGHT = 1600, 1000
        PNL_TABLE_WIDTH, PNL_TABLE_HEIGHT = 720, 330
        TRADING_TABLE_WIDTH, TRADING_TABLE_HEIGHT = 720, 360

        update_log("Processing report data...", log_list, log_placeholder)

        # --- Create local copies of dataframes to allow for modification ---
        df_PnL = report_data["ProfitandLoss"].copy()
        df_t1 = report_data["ProfitandLoss_T-1"].copy()
        df_t2 = report_data["ProfitandLoss_T-2"].copy()
        df_t3 = report_data["ProfitandLoss_T-3"].copy()
        df_t4 = report_data["ProfitandLoss_T-4"].copy()
        df_t5 = report_data["ProfitandLoss_T-5"].copy()
        df_trades = report_data.get("Trades")
        df_Instrument = pd.read_csv("data/Instrument list.csv")

        # --- APPLY PNL OVERRIDE IF PROVIDED --- ### MODIFIED SECTION ###
        if pnl_override:
            isin_to_change = pnl_override.get("isin")
            new_pnl_value = pnl_override.get("pnl")
            if isin_to_change and new_pnl_value is not None:
                # Find the original PnL to calculate the difference
                original_pnl_row = report_data["ProfitandLoss"][
                    report_data["ProfitandLoss"]['Unnamed: 0'] == isin_to_change]
                if not original_pnl_row.empty:
                    original_pnl_val = original_pnl_row['BTPLD'].iloc[0]
                    pnl_difference = new_pnl_value - original_pnl_val

                    # --- Get indices to ensure safe modification ---
                    isin_index = df_PnL.index[df_PnL['Unnamed: 0'] == isin_to_change].tolist()
                    scalmm_index = df_PnL.index[df_PnL['Unnamed: 0'] == 'SCALMM'].tolist()

                    if isin_index and scalmm_index:
                        # --- Apply change to the specific ISIN ---
                        # Update Daily PnL to the new value
                        df_PnL.loc[isin_index[0], 'BTPLD'] = new_pnl_value
                        # Update Monthly and Yearly PnL by the difference
                        df_PnL.loc[isin_index[0], 'BTPLM'] += pnl_difference # Added this line
                        df_PnL.loc[isin_index[0], 'BTPL'] += pnl_difference  # Added this line

                        # --- Apply change to the SCALMM total row ---
                        df_PnL.loc[scalmm_index[0], 'BTPLD'] += pnl_difference
                        df_PnL.loc[scalmm_index[0], 'BTPLM'] += pnl_difference # Added this line
                        df_PnL.loc[scalmm_index[0], 'BTPL'] += pnl_difference  # Added this line

                        update_log(
                            f"Applied override: ISIN {isin_to_change} PnL changed. Daily, Monthly, and Yearly totals adjusted by {pnl_difference:,.2f}.",
                            log_list, log_placeholder, "INFO")
                else:
                    update_log(f"Could not apply override: ISIN {isin_to_change} not found in PnL report.", log_list,
                               log_placeholder, "WARNING")
        ### END OF MODIFIED SECTION ###

        # --- All calculations will now use the potentially modified df_PnL ---
        Daily_PnL_MM = round(df_PnL[df_PnL['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
        Daily_PnL_t1 = round(df_t1[df_t1['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
        Daily_PnL_t2 = round(df_t2[df_t2['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
        Daily_PnL_t3 = round(df_t3[df_t3['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
        Daily_PnL_t4 = round(df_t4[df_t4['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
        Daily_PnL_MA5 = round((Daily_PnL_MM + Daily_PnL_t1 + Daily_PnL_t2 + Daily_PnL_t3 + Daily_PnL_t4) / 5)
        Total_Ytd_PnL_MM = round(
            df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
        Total_Ytd_PnL_t1 = round(
            df_t1.loc[(df_t1['Portfolio.Name'] == 'SCALMM') & (df_t1['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
        Daily_PnL_adj = Total_Ytd_PnL_MM - Total_Ytd_PnL_t1
        Total_Mtd_PnL_MM = round(
            df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BTPLM'].sum())
        Gross_Exposure_MM = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (
                    df_PnL['Unnamed: 0'] != 'SCALMM'), 'Gross Exposure'].sum())
        Net_Exposure_MM = round(
            df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BVal'].sum())
        Gross_Exposure_FR = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALFRAC') & (
                    df_PnL['Unnamed: 0'] != 'SCALFRAC'), 'Gross Exposure'].sum())
        Net_Exposure_FR = round(
            df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALFRAC') & (df_PnL['Unnamed: 0'] != 'SCALFRAC'), 'BVal'].sum())

        trade_metrics = {'Orders_Filled_MM': 'N/A', 'Orders_Filled_FR': 'N/A', 'Traded_Instruments_MM': 'N/A',
                         'Traded_Instruments_FR': 'N/A', 'Total_Turnover_MM': 'N/A', 'Total_Turnover_FR': 'N/A',
                         'Accepted_RFQ_Turnover_MM': 'N/A', 'Accepted_RFQ_Turnover_FR': 'N/A',
                         'Accepted_RFQ_Buy_Turnover_MM': 'N/A', 'Accepted_RFQ_Buy_Turnover_FR': 'N/A',
                         'PnL_per_RFQ': 'N/A', 'PnL_per_RFQ_Turnover': 'N/A', 'Net_PnL_per_RFQ_Turnover': 'N/A'}
        if df_trades is not None:
            trade_metrics['Orders_Filled_MM'] = ((df_trades["Portfolio"] == "SCALMM") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))).sum()
            trade_metrics['Traded_Instruments_MM'] = df_trades[(df_trades["Portfolio"] == "SCALMM") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Instrument'].nunique()
            trade_metrics['Total_Turnover_MM'] = round(
                df_trades[df_trades['Portfolio'] == 'SCALMM']['Premium'].abs().sum())
            trade_metrics['Accepted_RFQ_Turnover_MM'] = round(df_trades[(df_trades['Portfolio'] == 'SCALMM') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum())
            trade_metrics['Orders_Filled_FR'] = ((df_trades["Portfolio"] == "SCALFRAC") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))).sum()
            trade_metrics['Traded_Instruments_FR'] = df_trades[(df_trades["Portfolio"] == "SCALFRAC") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Instrument'].nunique()
            trade_metrics['Total_Turnover_FR'] = round(
                df_trades[df_trades['Portfolio'] == 'SCALFRAC']['Premium'].abs().sum())
            trade_metrics['Accepted_RFQ_Turnover_FR'] = round(df_trades[(df_trades['Portfolio'] == 'SCALFRAC') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum())
            buy_turnover_mm_total = df_trades[(df_trades['Portfolio'] == 'SCALMM') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                          df_trades['B/S'] == 'Buy')]['Premium'].abs().sum()
            trade_metrics[
                'Accepted_RFQ_Buy_Turnover_MM'] = f"{buy_turnover_mm_total / trade_metrics['Accepted_RFQ_Turnover_MM']:.2%}" if \
            trade_metrics['Accepted_RFQ_Turnover_MM'] > 0 else "0.00%"
            buy_turnover_fr_total = df_trades[(df_trades['Portfolio'] == 'SCALFRAC') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                          df_trades['B/S'] == 'Buy')]['Premium'].abs().sum()
            trade_metrics[
                'Accepted_RFQ_Buy_Turnover_FR'] = f"{buy_turnover_fr_total / trade_metrics['Accepted_RFQ_Turnover_FR']:.2%}" if \
            trade_metrics['Accepted_RFQ_Turnover_FR'] > 0 else "0.00%"
            trade_metrics['PnL_per_RFQ'] = round(
                Daily_PnL_MM / trade_metrics['Orders_Filled_MM'] if trade_metrics['Orders_Filled_MM'] > 0 else 0, 2)
            trade_metrics[
                'PnL_per_RFQ_Turnover'] = f"{round((Daily_PnL_MM / trade_metrics['Accepted_RFQ_Turnover_MM']) * 10000, 2)}bp" if \
            trade_metrics['Accepted_RFQ_Turnover_MM'] > 0 else "0.00bp"
            Estimated_Execution_Fees_MM = round(
                (trade_metrics['Total_Turnover_MM'] - trade_metrics['Accepted_RFQ_Turnover_MM']) * 0.00007 * 2 / 3, 2)
            trade_metrics[
                'Net_PnL_per_RFQ_Turnover'] = f"{round(((Daily_PnL_MM - Estimated_Execution_Fees_MM) / trade_metrics['Accepted_RFQ_Turnover_MM']) * 10000, 2)}bp" if \
            trade_metrics['Accepted_RFQ_Turnover_MM'] > 0 else "0.00bp"

        # --- Assemble Final DataFrames for display ---
        Statistics_data = [[trade_metrics['Orders_Filled_MM'], trade_metrics['Orders_Filled_FR']],
                           [trade_metrics['Traded_Instruments_MM'], trade_metrics['Traded_Instruments_FR']],
                           [f"{trade_metrics['Traded_Instruments_MM'] / total_instrument:.2%}" if isinstance(
                               trade_metrics['Traded_Instruments_MM'], int) else 'N/A', ''], [total_instrument, ''],
                           [f"{trade_metrics['Total_Turnover_MM']:,}" if isinstance(trade_metrics['Total_Turnover_MM'],
                                                                                    (int, float)) else 'N/A',
                            f"{trade_metrics['Total_Turnover_FR']:,}" if isinstance(trade_metrics['Total_Turnover_FR'],
                                                                                    (int, float)) else 'N/A'],
                           [f"{trade_metrics['Accepted_RFQ_Turnover_MM']:,}" if isinstance(
                               trade_metrics['Accepted_RFQ_Turnover_MM'], (int, float)) else 'N/A',
                            f"{trade_metrics['Accepted_RFQ_Turnover_FR']:,}" if isinstance(
                                trade_metrics['Accepted_RFQ_Turnover_FR'], (int, float)) else 'N/A'],
                           [trade_metrics['Accepted_RFQ_Buy_Turnover_MM'],
                            trade_metrics['Accepted_RFQ_Buy_Turnover_FR']], ['N/A' if trade_metrics[
                                                                                          'Accepted_RFQ_Buy_Turnover_MM'] == 'N/A' else f"{1 - float(trade_metrics['Accepted_RFQ_Buy_Turnover_MM'].strip('%')) / 100:.2%}",
                                                                             'N/A' if trade_metrics[
                                                                                          'Accepted_RFQ_Buy_Turnover_FR'] == 'N/A' else f"{1 - float(trade_metrics['Accepted_RFQ_Buy_Turnover_FR'].strip('%')) / 100:.2%}"],
                           [f"{round(trade_metrics['Accepted_RFQ_Turnover_MM'] / trade_metrics['Orders_Filled_MM'] if trade_metrics['Orders_Filled_MM'] not in ['N/A', 0] else 0):,.0f}" if isinstance(
                               trade_metrics['Accepted_RFQ_Turnover_MM'], (int, float)) else 'N/A', ''],
                           [f"{trade_metrics['Total_Turnover_MM'] - trade_metrics['Accepted_RFQ_Turnover_MM']:,}" if isinstance(
                               trade_metrics['Total_Turnover_MM'], (int, float)) else 'N/A',
                            f"{trade_metrics['Total_Turnover_FR'] - trade_metrics['Accepted_RFQ_Turnover_FR']:,}" if isinstance(
                                trade_metrics['Total_Turnover_FR'], (int, float)) else 'N/A']]
        row_names_Trading = ['Orders Filled', 'Traded Instruments', 'Traded Instruments %', 'Total Instruments',
                             'Total Turnover (€)', 'Accepted RFQ Turnover (€)', 'Accepted RFQ Buy Turnover %',
                             'Accepted RFQ Sell Turnover %', 'Average Accepted Order (€)', 'Hedge Volume (€)']
        df_Trading_Statistics = pd.DataFrame(data=Statistics_data, index=row_names_Trading,
                                             columns=['Market Making', 'Fractional'])
        df_Trading_Statistics.index.name = ""
        PnL_data = [f"{Daily_PnL_MM:,.0f}", f"{Daily_PnL_MA5:,.0f}",
                    # f"{Daily_PnL_adj:,.0f}",
                    trade_metrics['PnL_per_RFQ'], trade_metrics['PnL_per_RFQ_Turnover'],
                    trade_metrics['Net_PnL_per_RFQ_Turnover'], f"{Total_Mtd_PnL_MM:,.0f}", f"{Total_Ytd_PnL_MM:,.0f}",
                    f"{Gross_Exposure_MM + Gross_Exposure_FR:,.0f}", f"{Net_Exposure_MM + Net_Exposure_FR:,.0f}"]
        row_names_PnL = ['Daily PnL (€)', 'Daily PnL (MA5) (€)\uFE61',
                         # 'Total Daily PnL (€) Adj',
                         'PnL per RFQ (€)',
                         'PnL per RFQ Turnover', 'Net PnL per RFQ Turnover', 'Total MtD PnL (€)', 'Total YtD PnL (€)',
                         'Gross Exposure (€)', 'Net Exposure (€)']
        df_PnL_Statistics = pd.DataFrame(data=PnL_data, index=row_names_PnL, columns=['Market Making'])
        df_PnL_Statistics.index.name = ""

        # --- Chart Data Calculation ---
        update_log("Performing extended calculations for charts...", log_list, log_placeholder)
        xetr_cal = xcals.get_calendar("XETR")
        # Use the report_date passed into the function, not the dataframe
        date_mapping_data = [{'Term': 'T-0', 'Date': report_date.strftime('%Y%m%d')}]
        prev_day_ts = pd.to_datetime(report_date)
        for i in range(1, 5):
            prev_day_ts = xetr_cal.previous_session(prev_day_ts)
            date_mapping_data.append({'Term': f'T-{i}', 'Date': prev_day_ts.strftime('%Y%m%d')})
        df_date = pd.DataFrame(date_mapping_data)

        historical_pnl = {'T-0': df_PnL, 'T-1': df_t1, 'T-2': df_t2, 'T-3': df_t3, 'T-4': df_t4, 'T-5': df_t5}
        win_rate_data = {'Term': [f'T-{i}' for i in range(4, -1, -1)], 'Win_Rate': [
            (df[~df['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')['BTPLD'].sum() > 0).mean() for df
            in [df_t4, df_t3, df_t2, df_t1, df_PnL]]}
        ytd_pnl = {
            term: round(df.loc[(df['Portfolio.Name'] == 'SCALMM') & (df['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum()) for
            term, df in historical_pnl.items()}
        daily_intraday_pnl = {'T-0': Daily_PnL_MM, 'T-1': Daily_PnL_t1, 'T-2': Daily_PnL_t2, 'T-3': Daily_PnL_t3,
                              'T-4': Daily_PnL_t4}
        overnight_pnl_data = {f'T-{i}': (ytd_pnl[f'T-{i - 1}'] - ytd_pnl[f'T-{i}']) - daily_intraday_pnl[f'T-{i - 1}'] for i
                              in range(1, 5)}

        historical_trades_early = {f'T-{i}': report_data.get(f'Trades_T-{i}') for i in range(1, 5)}
        historical_trades_early['T-0'] = report_data.get('Trades')
        historical_trades_late = {f'T-{i}': report_data.get(f'Trades_late_T-{i}') for i in range(1, 5)}
        historical_trades_late['T-0'] = report_data.get('Trades_late')

        pnl_turnover_metrics = []
        for term in [f'T-{i}' for i in range(4, -1, -1)]:
            df_early, df_late = historical_trades_early.get(term), historical_trades_late.get(term)
            turnover_early = df_early[(df_early['Portfolio'] == 'SCALMM') & (
                df_early['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))][
                'Premium'].abs().sum() if df_early is not None else 0
            turnover_late = df_late[(df_late['Portfolio'] == 'SCALMM') & (
                df_late['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))][
                'Premium'].abs().sum() if df_late is not None else 0
            pnl_early, pnl_late = daily_intraday_pnl.get(term, 0), overnight_pnl_data.get(term, np.nan)
            bps_early = (pnl_early / turnover_early) * 10000 if turnover_early > 0 else np.nan
            bps_late = (pnl_late / turnover_late) * 10000 if turnover_late > 0 and pd.notna(pnl_late) else np.nan
            pnl_turnover_metrics.append(
                {'Term': term, 'pnl_turnover_early_bps': bps_early, 'pnl_turnover_late_bps': bps_late})
        df_pnl_turnover = pd.DataFrame(pnl_turnover_metrics)

        # --- Display Results ---
        st.subheader("📋 Copy-Friendly Report")
        st.markdown(
            """<a href="https://docs.google.com/document/d/1ij66_05uM6PPmSWehV3Pk6bfTL9FgOpd3tRIGeo4pVo/edit?usp=sharing" target="_blank">Click here for the email template</a>""",
            unsafe_allow_html=True)
        st.info(
            "You can right-click on any chart or table below and select 'Copy Image' to paste it in your report mail.")

        # --- NEW: Define styles for the PnL table ---
        pnl_row_styles = {
            'PnL per RFQ Turnover': {'fill_color': 'rgb(64, 64, 65)', 'font_color': 'white'},
            'Net PnL per RFQ Turnover': {'fill_color': 'rgb(64, 64, 65)', 'font_color': 'white'}
        }

        # --- MODIFIED: Pass the new styles dictionary to the function call ---
        pnl_table_png = create_table_image(df=df_PnL_Statistics,
                                           width=PNL_TABLE_WIDTH,
                                           height=PNL_TABLE_HEIGHT,
                                           col_widths=[2, 1],
                                           table_width_fraction=0.75,
                                           align_header=['left', 'right'],
                                           align_cells=['left', 'right'],
                                           row_styles=pnl_row_styles)  # <-- Pass the styles here
        st.image(pnl_table_png, caption="")

        # --- The rest of your code remains the same ---
        trading_table_png = create_table_image(df=df_Trading_Statistics,
                                               width=TRADING_TABLE_WIDTH,
                                               height=TRADING_TABLE_HEIGHT,
                                               col_widths=[2, 1, 1],
                                               align_header=['left', 'right', 'right'],
                                               align_cells=['left', 'right', 'right'])  # No styles passed here
        st.image(trading_table_png, caption="")
        pnl_plot_png = create_pnl_plot(daily_intraday_pnl, overnight_pnl_data, win_rate_data, df_pnl_turnover, df_date,
                                       width=PNL_PLOT_WIDTH, height=PNL_PLOT_HEIGHT)
        st.image(pnl_plot_png, caption="Daily PnL Breakdown")

        if all(f'Trades_T-{i}' in report_data for i in range(1, 5)) and 'Trades' in report_data:
            historical_trades = {'T-0': df_trades, **{f'T-{i}': report_data.get(f"Trades_T-{i}") for i in range(1, 5)}}
            total_turnover_data = {term: round(df[df['Portfolio'] == 'SCALMM']['Premium'].abs().sum()) for term, df in
                                   historical_trades.items()}
            accepted_turnover_data = {term: round(df[(df['Portfolio'] == 'SCALMM') & (
                df['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum()) for
                                      term, df in historical_trades.items()}
            turnover_plot_png = create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date,
                                                     width=TURNOVER_PLOT_WIDTH, height=TURNOVER_PLOT_HEIGHT)
            st.image(turnover_plot_png, caption="Daily Turnover Breakdown")
            top_performers_plot_png = create_top_performers_plot(df_PnL, df_trades, df_Instrument,
                                                                 width=TOP_PERF_PLOT_WIDTH, height=TOP_PERF_PLOT_HEIGHT)
            st.image(top_performers_plot_png, caption="Top 10 Winners & Losers by PnL")
        else:
            st.warning(
                "One or more historical Trades reports were not found, so Turnover and Top Performer charts cannot be generated.")

        update_log("🎉 Report generated successfully!", log_list, log_placeholder, "SUCCESS")

    except FileNotFoundError:
        update_log("Error: 'Instrument list.csv' not found. Cannot generate plots.", log_list, log_placeholder, "ERROR")
        st.error("Error: 'Instrument list.csv' not found. Please ensure the file is in the same directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred during report processing: {e}")
        st.exception(e)
        update_log(f"An unexpected error occurred: {e}", log_list, log_placeholder, "ERROR")


# --- Main Application State --- ### MODIFIED ###
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'pnl_override' not in st.session_state:
    st.session_state.pnl_override = None

# --- User Inputs on Main Page ---
st.subheader("Report Parameters")
col1, col2, col3 = st.columns([3, 1.5, 1.5])
with col1:
    bearer_token = st.text_input("Enter Bearer Token", type="password", help="https://franz.agent-tool.scalable.capital/misc/s3-access/prod-market-making_trade-reports_scalable-fis?bucketPath=fis%2F")
with col2:
    report_date = st.date_input("Select Report Date", value=datetime.now(ZoneInfo("Europe/Berlin")))
with col3:
    total_instrument = st.number_input("Total Instruments", min_value=1, value=5238,
                                       help="Set the total number of instruments.")

st.write("")
generate_button = st.button("Generate Report", type="primary", use_container_width=True)
st.divider()

# --- Main Application Logic --- ### MODIFIED ###
log_expander = st.expander("Activity Log", expanded=True)
log_placeholder = log_expander.empty()

if generate_button:
    if not bearer_token:
        st.error("Please enter your Bearer Token.")
    else:
        st.session_state.log_messages.clear()
        st.session_state.pnl_override = None  # Clear any previous overrides
        with st.spinner("Fetching report data... This may take a few minutes. ⏳"):
            date_str = report_date.strftime("%Y%m%d")
            st.session_state.report_data = get_reports(date_str, bearer_token, st.session_state.log_messages,
                                                       log_placeholder)

            # Check for essential data after fetching
            required_pnl_dfs = ["ProfitandLoss", "ProfitandLoss_T-1", "ProfitandLoss_T-2", "ProfitandLoss_T-3",
                                "ProfitandLoss_T-4", "ProfitandLoss_T-5"]
            if st.session_state.report_data:
                missing_dfs = [df for df in required_pnl_dfs if df not in st.session_state.report_data]
                if missing_dfs:
                    update_log(f"FATAL: Could not retrieve essential reports: {', '.join(missing_dfs)}.",
                               st.session_state.log_messages, log_placeholder, "ERROR")
                    st.session_state.report_data = None  # Invalidate data if essential parts are missing

# --- Report Display Area --- ### MODIFIED ###
if st.session_state.report_data:
    # Call the main function to process and display the report
    process_and_display_report(
        st.session_state.report_data,
        report_date,  # <--- Add this
        total_instrument,
        st.session_state.log_messages,
        log_placeholder,
        pnl_override=st.session_state.pnl_override
    )

    # --- NEW UI for PnL Adjustment --- ### NEW ###
    st.divider()

    st.subheader("⚙️ Adjust PnL and Re-Generate")

    df_trades_today = st.session_state.report_data.get("Trades")
    df_pnl_original = st.session_state.report_data.get("ProfitandLoss")

    if df_trades_today is not None and df_pnl_original is not None:
        traded_isins = sorted(list(df_trades_today['Instrument'].unique()))
        if not traded_isins:
            st.warning("No traded ISINs found for today to select for adjustment.")
        else:
            col_adj1, col_adj2, col_adj3 = st.columns([2.5, 2, 2.5])
            with col_adj1:
                # Add a placeholder (None) which creates a blank default option
                selected_isin = st.selectbox(
                    "Select ISIN to Adjust",
                    options=[None] + traded_isins,  # Add None to the start of the list
                    format_func=lambda x: '— Select an ISIN —' if x is None else x,  # Display text for None
                    help="Select an ISIN from today's trades to modify its PnL.",
                    key='selected_isin_for_adj'
                )

            # Only show the next widgets if a valid ISIN has been selected
            if selected_isin:
                with col_adj2:
                    default_pnl = 0.0
                    pnl_row = df_pnl_original[df_pnl_original['Unnamed: 0'] == selected_isin]
                    if not pnl_row.empty:
                        default_pnl = float(pnl_row['BTPLD'].iloc[0])

                    new_pnl = st.number_input(
                        f"New Daily PnL for {selected_isin}",
                        key=f"pnl_input_{selected_isin}",  # Use a dynamic key
                        value=default_pnl,
                        format="%.2f"
                    )
                with col_adj3:
                    # Add empty space to push the button down for better alignment
                    st.write("")
                    st.write("")
                    if st.button("Change and Re-Generate Report", use_container_width=True):
                        st.session_state.pnl_override = {
                            "isin": selected_isin,
                            "pnl": new_pnl
                        }
                        st.rerun()
    else:
        st.warning("Cannot display adjustment controls because Trades or PnL data for T-0 is missing.")

# Always update the log display at the end of every script run
update_log("Updated with latest input...", st.session_state.log_messages, log_placeholder)

# Show initial message if the app has just started
if not st.session_state.log_messages:
    update_log("Enter parameters above and click 'Generate Report' to start.", st.session_state.log_messages,
               log_placeholder)