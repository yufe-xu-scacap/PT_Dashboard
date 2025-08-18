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
st.set_page_config(layout="wide", page_title="Daily PnL Report Generation")
st.title("📈 Daily PnL Report Generation")


def update_log(message: str, log_list: List[str], log_placeholder, level: str = "INFO"):
    """Helper function to update the log list and the streamlit placeholder."""
    berlin_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%H:%M:%S')
    log_list.append(f"[{berlin_time}] {level}: {message}")
    log_placeholder.markdown(f"```\n{' \n'.join(log_list[-10:])}\n```")


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
                          "1805", "1810")
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
        table_width_fraction: float = 1.0
):
    """
    Generates a PNG image for a DataFrame table, constraining it
    to a fraction of the total image width.
    """
    fig = go.Figure(data=[go.Table(
        # --- FIX: The domain is specified here ---
        domain=dict(x=[0, table_width_fraction], y=[0, 1]),
        header=dict(
            values=[f"<b>{df.index.name}</b>"] + list(df.columns),
            fill_color='#6BDBCB',
            align='left',
            font=dict(color='#101112', size=18),
            height=40
        ),
        cells=dict(
            values=[df.index.tolist()] + [df[col] for col in df.columns],
            fill_color='white',
            align='left',
            font=dict(color='#101112', size=16),
            height=35
        ),
        columnwidth=col_widths
    )])

    # The layout update is now simpler
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
    )

    return pio.to_image(fig, format='png', width=width, height=height, scale=1.0)


# --- MODIFIED/UPDATED FUNCTION ---
def create_pnl_plot(pnl_data, overnight_data, win_rate_data, pnl_turnover_data, df_date, width: int, height: int):
    """
    Generates the PnL Breakdown plot with three Y-axes, separated right axes, and data labels.
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

    # Trace 1: Intraday PnL Area
    intraday_positions = ['middle right'] + ['bottom center'] * (len(df_plot) - 2) + ['middle left']
    fig.add_trace(
        go.Scatter(x=df_plot['DateLabel'], y=df_plot['Intraday_PnL'], name='Primary Session PnL (lhs)',
                   line=dict(color='#6BDBCB'),
                   fillcolor='rgba(107, 219, 203, 0.6)', mode='lines+markers+text', fill='tozeroy',
                   text=[f'{pnl / 1000:,.1f}k' for pnl in df_plot['Intraday_PnL']],
                   textposition=intraday_positions,
                   textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
                   hovertemplate='Intraday: %{y:,.0f} EUR<extra></extra>'))

    # Trace 2: Overnight PnL Fill Area
    x_fill_coords = list(df_plot['DateLabel'][:-1]) + list(df_plot['DateLabel'][:-1][::-1])
    y_fill_coords = list(df_plot['Total_PnL'][:-1]) + list(df_plot['Intraday_PnL'][:-1][::-1])
    fig.add_trace(
        go.Scatter(x=x_fill_coords, y=y_fill_coords, mode='none', fill='toself', fillcolor='rgba(247, 152, 128, 0.6)',
                   showlegend=False, hoverinfo='none'))

    # Trace 3: Total PnL Line
    line_data = df_plot['Total_PnL'].copy().astype(object)
    line_data.iloc[-1] = None
    text_data = line_data.copy()
    text_data.iloc[-1] = None
    overnight_pnl_for_hover = df_plot['Overnight_PnL'].copy().astype(object)
    overnight_pnl_for_hover.iloc[-1] = None
    overnight_positions = ['middle right'] + ['bottom center'] * (len(df_plot) - 3) + ['middle left']
    fig.add_trace(go.Scatter(x=df_plot['DateLabel'], y=line_data, name='Late Session PnL (lhs)', line=dict(color='#F79880'),
                             mode='lines+markers+text', text=[f'{pnl / 1000:,.1f}k' for pnl in text_data.dropna()],
                             textposition=overnight_positions,
                             textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
                             hovertemplate='Overnight: %{customdata:,.0f} EUR<br>Total: %{y:,.0f} EUR<extra></extra>',
                             customdata=overnight_pnl_for_hover))

    # Trace 4: Win Rate Line (Y-Axis 2)
    fig.add_trace(go.Scatter(x=df_plot['DateLabel'], y=df_plot['Win_Rate'], name='% of Profitable Instruments (rhs)',
                             line=dict(color='#FFC880', width=2, dash='dash'),
                             marker=dict(symbol='x-thin', size=20, color='#FFC880'),
                             mode='lines+markers', yaxis='y2',
                             hovertemplate='Win Rate: %{y:.1%}<extra></extra>'))

    # --- MODIFIED BPS TRACES ---
    num_points = len(df_plot)
    bps_positions_top = ['middle right'] + ['top center'] * (num_points - 2) + ['middle left']
    bps_positions_bottom = ['middle right'] + ['bottom center'] * (num_points - 2) + ['middle left']
    fig.add_trace(go.Scatter(
        x=df_plot['DateLabel'], y=df_plot['pnl_turnover_early_bps'], name='Primary Session PnL in bps (rhs)',
        line=dict(color='#371d76', width=2.5, dash='dot'),
        mode='lines+markers+text', yaxis='y3', marker=dict(symbol='cross', size=10),
        text=[f'{v:.1f}bp' if pd.notna(v) else '' for v in df_plot['pnl_turnover_early_bps']],
        textposition=bps_positions_top,
        textfont=dict(size=20, color='#371d76'),
        hovertemplate='Intraday PnL/Turnover: %{y:.2f} bps<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['DateLabel'], y=df_plot['pnl_turnover_late_bps'], name='Late Session PnL in bps (rhs)',
        line=dict(color='#ABB6FF', width=2.5, dash='dot'),
        mode='lines+markers+text', yaxis='y3', marker=dict(symbol='cross', size=10),
        text=[f'{v:.1f}bp' if pd.notna(v) else '' for v in df_plot['pnl_turnover_late_bps']],
        textposition=bps_positions_bottom,
        textfont=dict(size=20, color='#371d76'),
        hovertemplate='Late Session PnL/Turnover: %{y:.2f} bps<extra></extra>'
    ))

    # --- AXIS RANGE CALCULATION ---
    all_pnl_values = pd.concat([df_plot['Total_PnL'].dropna(), df_plot['Intraday_PnL'].dropna()])
    y_axis_min = min(0, all_pnl_values.min() * 1.2) if all_pnl_values.min() < 0 else 0
    y_axis_max = all_pnl_values.max() * 1.2 if all_pnl_values.max() > 0 else 1000

    # --- NEW: Calculate BPS axis range to anchor at 0 ---
    all_bps_values = pd.concat([df_plot['pnl_turnover_early_bps'], df_plot['pnl_turnover_late_bps']]).dropna()
    if not all_bps_values.empty:
        min_bps = all_bps_values.min()
        max_bps = all_bps_values.max()
        bps_axis_min = min(0, min_bps * 1.2) if min_bps < 0 else 0
        bps_axis_max = max(0, max_bps * 1.2) if max_bps > 0 else 0
        if bps_axis_min == 0 and bps_axis_max == 0:  # Add buffer if range is zero
            bps_axis_min, bps_axis_max = -1, 1
    else:
        bps_axis_min, bps_axis_max = -5, 5  # Default range if no data
    bps_range = [bps_axis_min, bps_axis_max]

    # --- LAYOUT STYLING ---
    fig.update_layout(
        title=dict(text='<b>Daily PnL Breakdown (T-4 to T-0)</b>', y=0.95, x=0.5, xanchor='center', yanchor='top',
                   font=dict(size=28, color='#101112')),
        plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=150, l=80, r=80),  # <-- Increase bottom margin for space
        legend=dict(
            orientation="h",  # <-- Set to horizontal
            yanchor="top",  # <-- Anchor legend's top to the y position
            y=-0.1,  # <-- Position legend below the plot
            xanchor="center",  # <-- Anchor legend's center to the x position
            x=0.5,  # <-- Center the legend horizontally
            font=dict(size=18),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#EAEAEA',
            borderwidth=1
        ),
        xaxis=dict(
            domain=[0, 0.94],
            type='category', title_text='', tickfont=dict(size=20),
            showgrid=False, showline=True, linewidth=1, linecolor='#101112', mirror=True,
            range=[0, len(df_plot) - 1]
        ),
        yaxis=dict(
            title=dict(text='Profit and Loss (EUR)', standoff=15),
            tickformat='~s', range=[y_axis_min, y_axis_max],
            title_font=dict(size=22), tickfont=dict(size=20),
            showgrid=False, showline=True, linewidth=1, linecolor='#101112', mirror=True
        ),
        yaxis2=dict(
            title='Win Rate', overlaying='y', side='right', range=[0, 1], tickformat='.0%',
            showgrid=False, showline=False,
            anchor='x',
            title_font=dict(color='#FFC880', size=22),
            tickfont=dict(color='#FFC880', size=20)
        ),
        yaxis3=dict(
            title=dict(text='PnL in bps', standoff=15),
            range=bps_range,  # <-- APPLY THE NEW RANGE
            overlaying='y', side='right', showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='#371d76',
            anchor='free', position=1.0,
            title_font=dict(color='#371d76', size=22),  # Corrected color
            tickfont=dict(color='#371d76', size=20),
            ticks="outside",
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickcolor="black",
                showgrid=False,
                griddash='dot',
                gridcolor='LightGrey'
            )
        )
    )
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)


def create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date, width: int, height: int):
    """
    Generates the Turnover Breakdown plot with a design consistent with the PnL chart.
    """
    # --- DATA PREPARATION ---
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

    # --- PLOT CREATION ---
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['DateLabel'], y=df_plot['Accepted_RFQ_Turnover_M'], name='Accepted RFQ Turnover',
        marker_color='#6BDBCB',
        text=df_plot['Accepted_RFQ_Turnover_M'], textposition='inside', insidetextanchor='middle',
        texttemplate='%{text:.1f}M',
        textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        hovertemplate='Accepted RFQ: %{customdata:,.0f} EUR<extra></extra>',
        customdata=df_plot['Accepted_RFQ_Turnover'].values
    ))
    fig.add_trace(go.Bar(
        x=df_plot['DateLabel'], y=df_plot['Hedge_Turnover_M'], name='Hedge Turnover', marker_color='#349386',
        text=df_plot['Hedge_Turnover_M'], textposition='inside', insidetextanchor='middle', texttemplate='%{text:.1f}M',
        textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        hovertemplate='Hedge: %{customdata:,.0f} EUR<extra></extra>', customdata=df_plot['Hedge_Turnover'].values
    ))
    fig.add_trace(go.Bar(
        x=df_plot['DateLabel'], y=[0] * len(df_plot), text=df_plot['Total_Turnover_M'], textposition='outside',
        texttemplate='<b>%{text:.1f}M</b>', textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        cliponaxis=False, showlegend=False, hoverinfo='none'
    ))

    # --- LAYOUT STYLING ---
    fig.update_layout(
        barmode='stack',
        title=dict(text='<b>Daily Turnover Breakdown (T-4 to T-0)</b>', y=0.95, x=0.5, xanchor='center', yanchor='top',
                   font=dict(size=28, color='#101112', family="Arial, sans-serif")),
        plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=80, l=80, r=80),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02, font=dict(size=20),
                    bgcolor='rgba(255,255,255,0.7)', bordercolor='#EAEAEA', borderwidth=1),
        xaxis=dict(type='category', title_text='', tickfont=dict(size=20), showgrid=False, showline=True, linewidth=1,
                   linecolor='#101112', mirror=True, range=[-0.5, len(df_plot) - 0.5]),
        yaxis=dict(title='Turnover (M EUR)', range=[0, df_plot['Total_Turnover_M'].max() * 1.15],
                   title_font=dict(size=22), tickfont=dict(size=20), showgrid=False, showline=True, linewidth=1,
                   linecolor='#101112', mirror=True)
    )
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)


def create_top_performers_plot(df_PnL, df_trades, df_Instrument, width: int, height: int):
    """
    Generates the Top/Bottom Performers plot using Plotly, matching the standard design.
    """
    # --- DATA PREPARATION ---
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
    df_plot['Abs_PnL'] = df_plot['Total PnL (€)'].abs()
    df_plot['Label'] = [f"{pnl:,.0f}€ (Turnover: {turnover:,.0f}€)" for pnl, turnover in
                        zip(df_plot['Total PnL (€)'], df_plot['Turnover (€)'])]
    colors = ['#6BDBCB' if pnl > 0 else '#F79880' for pnl in df_plot['Total PnL (€)']]

    # --- PLOT CREATION ---
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['Abs_PnL'], y=df_plot['Security Name'], orientation='h', marker_color=colors, text=df_plot['Label'],
        textposition='auto',
        textfont=dict(size=16, family="Arial, sans-serif"),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>ISIN: %{customdata[1]}<br>PnL: %{customdata[2]:,.0f}€<br>Turnover: %{customdata[3]:,.0f}€<extra></extra>'),
        customdata=df_plot[['Security Name', 'ISIN', 'Total PnL (€)', 'Turnover (€)']].values
    ))

    # --- LAYOUT STYLING ---
    fig.update_layout(
        title=dict(text='<b>Top 10 Winners & Losers by PnL</b>', y=0.95, x=0.5, xanchor='center', yanchor='top',
                   font=dict(size=28, color='#101112', family="Arial, sans-serif")),
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=80, l=250, r=40), showlegend=False,
        xaxis=dict(title='Daily PnL (€)', range=[0, df_plot['Abs_PnL'].max() * 1.05], title_font=dict(size=22),
                   tickfont=dict(size=20), tickformat=',.0f', showgrid=False, showline=True, linewidth=1,
                   linecolor='#101112', mirror=True),
        yaxis=dict(title='', type='category', tickfont=dict(size=16), showline=True, linewidth=1, linecolor='#101112',
                   mirror=True)
    )
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)


# --- User Inputs on Main Page ---
st.subheader("Report Parameters")
col1, col2, col3 = st.columns([3, 1.5, 1.5])
with col1:
    bearer_token = st.text_input("Enter Bearer Token", type="password", help="Your secret authentication token.")
with col2:
    report_date = st.date_input("Select Report Date", value=datetime.now(ZoneInfo("Europe/Berlin")))
with col3:
    total_instrument = st.number_input("Total Instruments", min_value=1, value=5238,
                                       help="Set the total number of instruments.")

st.write("")
generate_button = st.button("Generate Report", type="primary", use_container_width=True)
st.divider()

# --- Main Application Logic ---
log_expander = st.expander("Activity Log", expanded=True)
log_placeholder = log_expander.empty()
log_messages = []

if generate_button:
    if not bearer_token:
        st.error("Please enter your Bearer Token.")
    else:
        try:
            # --- Define plot and table dimensions ---
            PNL_PLOT_WIDTH = 1600
            PNL_PLOT_HEIGHT = 900
            TURNOVER_PLOT_WIDTH = 1600
            TURNOVER_PLOT_HEIGHT = 900
            TOP_PERF_PLOT_WIDTH = 1600
            TOP_PERF_PLOT_HEIGHT = 1000
            # PNL_TABLE_WIDTH = 550
            PNL_TABLE_WIDTH = 720
            PNL_TABLE_HEIGHT = 440
            TRADING_TABLE_WIDTH = 720
            TRADING_TABLE_HEIGHT = 440

            log_messages.clear()
            update_log("⏳ Initializing... Reading local files.", log_messages, log_placeholder)
            try:
                df_Instrument = pd.read_csv("Instrument list.csv")
            except FileNotFoundError:
                update_log("Error: 'Instrument list.csv' not found.", log_messages, log_placeholder, "ERROR")
                st.stop()

            date_str = report_date.strftime("%Y%m%d")
            successfully_downloaded = get_reports(date_str, bearer_token, log_messages, log_placeholder)

            required_pnl_dfs = ["ProfitandLoss", "ProfitandLoss_T-1", "ProfitandLoss_T-2", "ProfitandLoss_T-3",
                                "ProfitandLoss_T-4", "ProfitandLoss_T-5"]
            missing_dfs = [df for df in required_pnl_dfs if df not in successfully_downloaded]
            if missing_dfs:
                update_log(f"FATAL: Could not retrieve essential P&L reports: {', '.join(missing_dfs)}.", log_messages,
                           log_placeholder, "ERROR")
                st.stop()

            update_log("✔️ All required data retrieved. Performing calculations...", log_messages, log_placeholder,
                       "SUCCESS")

            # --- Extracting DataFrames from the dictionary ---
            df_PnL = successfully_downloaded["ProfitandLoss"]
            df_t1, df_t2, df_t3, df_t4, df_t5 = (successfully_downloaded[f"ProfitandLoss_T-{i}"] for i in range(1, 6))
            df_trades = successfully_downloaded.get("Trades")

            # --- Calculations ---
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
            Net_Exposure_FR = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALFRAC') & (
                        df_PnL['Unnamed: 0'] != 'SCALFRAC'), 'BVal'].sum())

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
                    (trade_metrics['Total_Turnover_MM'] - trade_metrics['Accepted_RFQ_Turnover_MM']) * 0.00007 * 2 / 3,
                    2)
                trade_metrics[
                    'Net_PnL_per_RFQ_Turnover'] = f"{round(((Daily_PnL_MM - Estimated_Execution_Fees_MM) / trade_metrics['Accepted_RFQ_Turnover_MM']) * 10000, 2)}bp" if \
                trade_metrics['Accepted_RFQ_Turnover_MM'] > 0 else "0.00bp"

            # --- Assemble Final DataFrames ---
            Statistics_data = [[trade_metrics['Orders_Filled_MM'], trade_metrics['Orders_Filled_FR']],
                               [trade_metrics['Traded_Instruments_MM'], trade_metrics['Traded_Instruments_FR']],
                               [f"{trade_metrics['Traded_Instruments_MM'] / total_instrument:.2%}" if isinstance(
                                   trade_metrics['Traded_Instruments_MM'], int) else 'N/A', ''], [total_instrument, ''],
                               [f"{trade_metrics['Total_Turnover_MM']:,}" if isinstance(
                                   trade_metrics['Total_Turnover_MM'], (int, float)) else 'N/A',
                                f"{trade_metrics['Total_Turnover_FR']:,}" if isinstance(
                                    trade_metrics['Total_Turnover_FR'], (int, float)) else 'N/A'],
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
            PnL_data = [f"{Daily_PnL_MM:,.0f}", f"{Daily_PnL_MA5:,.0f}", f"{Daily_PnL_adj:,.0f}",
                        trade_metrics['PnL_per_RFQ'], trade_metrics['PnL_per_RFQ_Turnover'],
                        trade_metrics['Net_PnL_per_RFQ_Turnover'], f"{Total_Mtd_PnL_MM:,.0f}",
                        f"{Total_Ytd_PnL_MM:,.0f}", f"{Gross_Exposure_MM + Gross_Exposure_FR:,.0f}",
                        f"{Net_Exposure_MM + Net_Exposure_FR:,.0f}"]
            row_names_PnL = ['Daily PnL (€)', 'Daily PnL (MA5) (€)', 'Total Daily PnL (€) Adj', 'PnL per RFQ (€)',
                             'PnL per RFQ Turnover', 'Net PnL per RFQ Turnover', 'Total MtD PnL (€)',
                             'Total YtD PnL (€)', 'Gross Exposure (€)', 'Net Exposure (€)']
            df_PnL_Statistics = pd.DataFrame(data=PnL_data, index=row_names_PnL, columns=['Market Making'])
            # --- NEW: Add a blank column to match the other table's structure ---
            # df_PnL_Statistics[''] = ''  # Adds a new column named 'Fractional' with blank values.
            df_PnL_Statistics.index.name = ""

            # --- Extended calculations for charts ---
            update_log("📊 Performing extended calculations for charts...", log_messages, log_placeholder)
            xetr_cal = xcals.get_calendar("XETR")
            date_mapping_data = [{'Term': 'T-0', 'Date': report_date.strftime('%Y%m%d')}]
            prev_day_ts = pd.to_datetime(report_date)
            for i in range(1, 5):
                prev_day_ts = xetr_cal.previous_session(prev_day_ts)
                date_mapping_data.append({'Term': f'T-{i}', 'Date': prev_day_ts.strftime('%Y%m%d')})
            df_date = pd.DataFrame(date_mapping_data)

            historical_pnl = {'T-0': df_PnL, 'T-1': df_t1, 'T-2': df_t2, 'T-3': df_t3, 'T-4': df_t4, 'T-5': df_t5}
            win_rate_data = {'Term': [f'T-{i}' for i in range(4, -1, -1)], 'Win_Rate': [
                (df[~df['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')['BTPLD'].sum() > 0).mean() for
                df in [df_t4, df_t3, df_t2, df_t1, df_PnL]]}
            ytd_pnl = {
                term: round(df.loc[(df['Portfolio.Name'] == 'SCALMM') & (df['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
                for term, df in historical_pnl.items()}
            daily_intraday_pnl = {'T-0': Daily_PnL_MM, 'T-1': Daily_PnL_t1, 'T-2': Daily_PnL_t2, 'T-3': Daily_PnL_t3,
                                  'T-4': Daily_PnL_t4}
            overnight_pnl_data = {f'T-{i}': (ytd_pnl[f'T-{i}'] - ytd_pnl[f'T-{i + 1}']) - daily_intraday_pnl[f'T-{i}']
                                  for i in range(1, 5)}

            # --- Calculate PnL per RFQ Turnover in BPS ---
            historical_trades_early = {f'T-{i}': successfully_downloaded.get(f'Trades_T-{i}') for i in range(1, 5)}
            historical_trades_early['T-0'] = successfully_downloaded.get('Trades')
            historical_trades_late = {f'T-{i}': successfully_downloaded.get(f'Trades_late_T-{i}') for i in range(1, 5)}
            historical_trades_late['T-0'] = successfully_downloaded.get('Trades_late')

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

            pnl_table_png = create_table_image(df=df_PnL_Statistics, width=PNL_TABLE_WIDTH, height=PNL_TABLE_HEIGHT,
                                               col_widths=[2, 1, 1], table_width_fraction=0.75)
            st.image(pnl_table_png, caption="")
            trading_table_png = create_table_image(df=df_Trading_Statistics, width=TRADING_TABLE_WIDTH,
                                                   height=TRADING_TABLE_HEIGHT, col_widths=[2, 1, 1])
            st.image(trading_table_png, caption="")

            # --- Display Plots ---
            pnl_plot_png = create_pnl_plot(daily_intraday_pnl, overnight_pnl_data, win_rate_data, df_pnl_turnover,
                                           df_date, width=PNL_PLOT_WIDTH, height=PNL_PLOT_HEIGHT)
            st.image(pnl_plot_png, caption="Daily PnL Breakdown")

            if all(f'Trades_T-{i}' in successfully_downloaded for i in
                   range(1, 5)) and 'Trades' in successfully_downloaded:
                historical_trades = {'T-0': df_trades,
                                     **{f'T-{i}': successfully_downloaded.get(f"Trades_T-{i}") for i in range(1, 5)}}
                total_turnover_data = {term: round(df[df['Portfolio'] == 'SCALMM']['Premium'].abs().sum()) for term, df
                                       in historical_trades.items()}
                accepted_turnover_data = {term: round(df[(df['Portfolio'] == 'SCALMM') & (
                    df['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum()) for
                                          term, df in historical_trades.items()}

                turnover_plot_png = create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date,
                                                         width=TURNOVER_PLOT_WIDTH, height=TURNOVER_PLOT_HEIGHT)
                st.image(turnover_plot_png, caption="Daily Turnover Breakdown")
                top_performers_plot_png = create_top_performers_plot(df_PnL, df_trades, df_Instrument,
                                                                     width=TOP_PERF_PLOT_WIDTH,
                                                                     height=TOP_PERF_PLOT_HEIGHT)
                st.image(top_performers_plot_png, caption="Top 10 Winners & Losers by PnL")
            else:
                st.warning(
                    "One or more historical Trades reports were not found, so Turnover and Top Performer charts cannot be generated.")
            update_log("🎉 Report generated successfully!", log_messages, log_placeholder, "SUCCESS")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.exception(e)
else:
    update_log("Enter parameters above and click 'Generate Report' to start.", log_messages, log_placeholder)