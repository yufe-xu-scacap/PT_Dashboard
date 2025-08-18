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

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Trading Report Dashboard")
st.title("📈 Trading Report Dashboard")


def update_log(message: str, log_list: List[str], log_placeholder, level: str = "INFO"):
    """Helper function to update the log list and the streamlit placeholder."""
    berlin_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%H:%M:%S')
    log_list.append(f"[{berlin_time}] {level}: {message}")
    log_placeholder.markdown(f"```\n{' \n'.join(log_list[-10:])}\n```")


# --- Core Data Fetching Function ---
def get_reports(date: str, bearer_token: str, log_list: List[str], log_placeholder) -> Dict[str, pd.DataFrame]:
    """
    Retrieves daily and historical reports, appending feedback to the provided log_list.
    """
    date_str = str(date)
    headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
    report_dataframes = {}

    def _fetch_report(d_str: str, report_config: dict, start_time_str: str, end_time_str: str) -> Optional[
        pd.DataFrame]:
        try:
            start_dt = datetime.strptime(f"{d_str}{start_time_str}", "%Y%m%d%H%M")
            end_dt = datetime.strptime(f"{d_str}{end_time_str}", "%Y%m%d%H%M")
            current_dt = start_dt
        except ValueError:
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
            except requests.exceptions.RequestException:
                pass
            current_dt += timedelta(minutes=1)
        return None

    xetr_cal = xcals.get_calendar("XETR")
    current_date_ts = pd.to_datetime(date_str).normalize()

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

    update_log(f"Searching for Trades reports...", log_list, log_placeholder)
    trades_df = _fetch_report(date_str,
                              {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"}, "1806",
                              "1810")
    if trades_df is not None:
        report_dataframes['Trades'] = trades_df.copy()
    else:
        update_log(f"Trades report for {date_str} (T-0) not found.", log_list, log_placeholder, "WARNING")

    prev_day_ts = current_date_ts
    for i in range(1, 5):
        prev_day_ts = xetr_cal.previous_session(prev_day_ts)
        prev_day_str = prev_day_ts.strftime('%Y%m%d')
        df_trade_hist = _fetch_report(prev_day_str,
                                      {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"},
                                      "1806", "1810")
        if df_trade_hist is not None:
            report_dataframes[f"Trades_T-{i}"] = df_trade_hist.copy()
        else:
            update_log(f"Trades report for {prev_day_str} (T-{i}) not found.", log_list, log_placeholder, "WARNING")

    update_log("All search operations complete.", log_list, log_placeholder)
    return report_dataframes


# --- NEW/MODIFIED CODE BLOCK STARTS HERE ---

def create_table_image(
        df: pd.DataFrame,
        width: int,
        height: int,
        col_widths: List[int]
):
    """
    Generates a PNG image for a single DataFrame table.
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{df.index.name}</b>"] + list(df.columns),
            fill_color='#6BDBCB',
            align='left',
            font=dict(color='#101112', size=18),
            height=40  # Slightly more height for the header
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

    # --- Layout Styling for a single table ---
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
    )

    return pio.to_image(fig, format='png', width=width, height=height, scale=1.0)


# --- NEW/MODIFIED CODE BLOCK ENDS HERE ---


def create_pnl_plot(pnl_data, overnight_data, win_rate_data, df_date, width: int, height: int):
    """
    Generates the PnL Breakdown plot with a full rectangular border and a vertical legend.
    """
    # --- DATA PREPARATION ---
    plot_data_list = [
        {'Term': f'T-{i}', 'Intraday_PnL': pnl_data.get(f'T-{i}', 0), 'Overnight_PnL': overnight_data.get(f'T-{i}', 0)}
        for i in range(4, 0, -1)]
    plot_data_list.append({'Term': 'T-0', 'Intraday_PnL': pnl_data.get('T-0', 0), 'Overnight_PnL': 0})
    df_plot = pd.DataFrame(plot_data_list).merge(df_date, on='Term')
    df_win_rate = pd.DataFrame(win_rate_data)
    df_plot = df_plot.merge(df_win_rate, on='Term')
    df_plot['Total_PnL'] = df_plot['Intraday_PnL'] + df_plot['Overnight_PnL']
    df_plot['DateObj'] = pd.to_datetime(df_plot['Date'], format='%Y%m%d')
    df_plot = df_plot.sort_values(by='DateObj').reset_index(drop=True)
    df_plot['DateLabel'] = df_plot['DateObj'].dt.strftime('%Y-%m-%d')

    # --- PLOT CREATION ---
    fig = go.Figure()

    # Trace 1: Intraday PnL Area
    intraday_positions = ['middle right'] + ['bottom center'] * (len(df_plot) - 2) + ['middle left']
    fig.add_trace(
        go.Scatter(x=df_plot['DateLabel'], y=df_plot['Intraday_PnL'], name='Primary Session PnL', line=dict(color='#6BDBCB'),
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

    fig.add_trace(go.Scatter(x=df_plot['DateLabel'], y=line_data, name='Late Session PnL', line=dict(color='#F79880'),
                             mode='lines+markers+text', text=[f'{pnl / 1000:,.1f}k' for pnl in text_data.dropna()],
                             textposition=overnight_positions,
                             textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
                             hovertemplate='Overnight: %{customdata:,.0f} EUR<br>Total: %{y:,.0f} EUR<extra></extra>',
                             customdata=overnight_pnl_for_hover))

    # Trace 4: Win Rate Line
    fig.add_trace(go.Scatter(x=df_plot['DateLabel'], y=df_plot['Win_Rate'], name='% of Profitable Instruments',
                             line=dict(color='#371d76', width=2, dash='dash'),
                             marker=dict(symbol='x-thin', size=20, color='#371d76'),
                             mode='lines+markers', yaxis='y2',
                             hovertemplate='Win Rate: %{y:.1%}<extra></extra>'))

    # --- AXIS RANGE CALCULATION ---
    all_pnl_values = pd.concat([df_plot['Total_PnL'].dropna(), df_plot['Intraday_PnL'].dropna()])
    min_pnl = all_pnl_values.min()
    max_pnl = all_pnl_values.max()
    y_axis_min = min(0, min_pnl * 1.2) if min_pnl < 0 else 0
    y_axis_max = max_pnl * 1.2 if max_pnl > 0 else 1000
    pnl_range = [y_axis_min, y_axis_max]
    num_datapoints = len(df_plot)
    xaxis_tight_range = [0, num_datapoints - 1]

    # --- LAYOUT STYLING ---
    fig.update_layout(
        title=dict(
            text='<b>Daily PnL Breakdown (T-4 to T-0)</b>',
            y=0.95, x=0.5, xanchor='center', yanchor='top',
            font=dict(size=28, color='#101112', family="Arial, sans-serif")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=80, l=80, r=80),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#EAEAEA',
            borderwidth=1
        ),
        xaxis=dict(
            type='category',
            title_text='',
            tickfont=dict(size=20),
            showgrid=False,
            showline=True,
            linewidth=1, linecolor='#101112',
            mirror=True,
            domain=[0, 1],
            range=xaxis_tight_range
        ),
        yaxis=dict(
            title=dict(text='Profit and Loss (EUR)', standoff=20),
            tickformat='~s',
            range=pnl_range,
            title_font=dict(size=22),
            tickfont=dict(size=20),
            showgrid=False,
            showline=True,
            linewidth=1, linecolor='#101112',
            mirror=True
        ),
        yaxis2=dict(
            title='Win Rate',
            overlaying='y', side='right', range=[0, 1], tickformat='.0%',
            showgrid=False,
            title_font=dict(color='#371d76', size=22),
            tickfont=dict(color='#371d76', size=20)
        )
    )
    return pio.to_image(fig, format='png', width=width, height=height, scale=1)


def create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date, width: int, height: int):
    """
    Generates the Turnover Breakdown plot with a design consistent with the PnL chart.
    """
    # --- DATA PREPARATION ---
    plot_data_list = []
    for i in range(4, -1, -1):
        term = f'T-{i}' if i > 0 else 'T-0'
        plot_data_list.append({
            'Term': term,
            'Total_Turnover': total_turnover_data.get(term, 0),
            'Accepted_RFQ_Turnover': accepted_turnover_data.get(term, 0)
        })
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
        x=df_plot['DateLabel'],
        y=df_plot['Accepted_RFQ_Turnover_M'],
        name='Accepted RFQ Turnover',
        marker_color='#6BDBCB',
        text=df_plot['Accepted_RFQ_Turnover_M'],
        textposition='inside',
        insidetextanchor='middle',
        texttemplate='%{text:.1f}M',
        textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        hovertemplate='Accepted RFQ: %{customdata:,.0f} EUR<extra></extra>',
        customdata=df_plot['Accepted_RFQ_Turnover'].values
    ))

    fig.add_trace(go.Bar(
        x=df_plot['DateLabel'],
        y=df_plot['Hedge_Turnover_M'],
        name='Hedge Turnover',
        marker_color='#349386',
        text=df_plot['Hedge_Turnover_M'],
        textposition='inside',
        insidetextanchor='middle',
        texttemplate='%{text:.1f}M',
        textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        hovertemplate='Hedge: %{customdata:,.0f} EUR<extra></extra>',
        customdata=df_plot['Hedge_Turnover'].values
    ))

    fig.add_trace(go.Bar(
        x=df_plot['DateLabel'],
        y=[0] * len(df_plot),
        text=df_plot['Total_Turnover_M'],
        textposition='outside',
        texttemplate='<b>%{text:.1f}M</b>',
        textfont=dict(size=20, color='#101112', family="Arial, sans-serif"),
        cliponaxis=False,
        showlegend=False,
        hoverinfo='none'
    ))

    # --- AXIS RANGE CALCULATION ---
    num_datapoints = len(df_plot)
    xaxis_tight_range = [-0.5, num_datapoints - 0.5]
    max_yaxis = df_plot['Total_Turnover_M'].max() * 1.15

    # --- LAYOUT STYLING ---
    fig.update_layout(
        barmode='stack',
        title=dict(
            text='<b>Daily Turnover Breakdown (T-4 to T-0)</b>',
            y=0.95, x=0.5, xanchor='center', yanchor='top',
            font=dict(size=28, color='#101112', family="Arial, sans-serif")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=80, l=80, r=80),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#EAEAEA',
            borderwidth=1
        ),
        xaxis=dict(
            type='category',
            title_text='',
            tickfont=dict(size=20),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#101112',
            mirror=True
        ),
        yaxis=dict(
            title='Turnover (M EUR)',
            range=[0, max_yaxis],
            title_font=dict(size=22),
            tickfont=dict(size=20),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#101112',
            mirror=True
        )
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
    df_plot['Label'] = [f"{pnl:,.0f}€ (Turnover: {turnover:,.0f}€)"
                        for pnl, turnover in zip(df_plot['Total PnL (€)'], df_plot['Turnover (€)'])]
    colors = ['#6BDBCB' if pnl > 0 else '#F79880' for pnl in df_plot['Total PnL (€)']]

    # --- PLOT CREATION ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_plot['Abs_PnL'],
        y=df_plot['Security Name'],
        orientation='h',
        marker_color=colors,
        text=df_plot['Label'],
        textposition='auto',
        textfont=dict(size=16, family="Arial, sans-serif"),
        hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'ISIN: %{customdata[1]}<br>' +
                'PnL: %{customdata[2]:,.0f}€<br>' +
                'Turnover: %{customdata[3]:,.0f}€<extra></extra>'
        ),
        customdata=df_plot[['Security Name', 'ISIN', 'Total PnL (€)', 'Turnover (€)']].values
    ))

    # --- LAYOUT STYLING ---
    max_x_range = df_plot['Abs_PnL'].max() * 1.05

    fig.update_layout(
        title=dict(
            text='<b>Top 10 Winners & Losers by PnL</b>',
            y=0.95, x=0.5, xanchor='center', yanchor='top',
            font=dict(size=28, color='#101112', family="Arial, sans-serif")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color='#101112'),
        margin=dict(t=80, b=80, l=250, r=40),
        showlegend=False,
        xaxis=dict(
            title='Daily PnL (€)',
            range=[0, max_x_range],
            title_font=dict(size=22),
            tickfont=dict(size=20),
            tickformat=',.0f',
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#101112',
            mirror=True
        ),
        yaxis=dict(
            title='',
            type='category',
            tickfont=dict(size=16),
            showline=True,
            linewidth=1,
            linecolor='#101112',
            mirror=True
        )
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
            # --- NEW/MODIFIED CODE BLOCK STARTS HERE ---
            # --- Define plot and table dimensions ---
            PNL_PLOT_WIDTH = 1600
            PNL_PLOT_HEIGHT = 900
            TURNOVER_PLOT_WIDTH = 1600
            TURNOVER_PLOT_HEIGHT = 900
            TOP_PERF_PLOT_WIDTH = 1600
            TOP_PERF_PLOT_HEIGHT = 1000
            # Define specific sizes for each table
            PNL_TABLE_WIDTH = 550
            PNL_TABLE_HEIGHT = 440
            TRADING_TABLE_WIDTH = 720
            TRADING_TABLE_HEIGHT = 440
            # --- NEW/MODIFIED CODE BLOCK ENDS HERE ---

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

            df_PnL = successfully_downloaded["ProfitandLoss"]
            df_t1, df_t2, df_t3, df_t4, df_t5 = (successfully_downloaded[f"ProfitandLoss_T-{i}"] for i in range(1, 6))
            df_trades = successfully_downloaded.get("Trades")
            df_trades_t1 = successfully_downloaded.get("Trades_T-1")
            df_trades_t2 = successfully_downloaded.get("Trades_T-2")
            df_trades_t3 = successfully_downloaded.get("Trades_T-3")
            df_trades_t4 = successfully_downloaded.get("Trades_T-4")

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

            trade_metrics = {
                'Orders_Filled_MM': 'N/A', 'Orders_Filled_FR': 'N/A', 'Traded_Instruments_MM': 'N/A',
                'Traded_Instruments_FR': 'N/A', 'Total_Turnover_MM': 'N/A', 'Total_Turnover_FR': 'N/A',
                'Accepted_RFQ_Turnover_MM': 'N/A', 'Accepted_RFQ_Turnover_FR': 'N/A',
                'Accepted_RFQ_Buy_Turnover_MM': 'N/A', 'Accepted_RFQ_Buy_Turnover_FR': 'N/A',
                'PnL_per_RFQ': 'N/A', 'PnL_per_RFQ_Turnover': 'N/A', 'Net_PnL_per_RFQ_Turnover': 'N/A'
            }

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
            Statistics_data = [
                [trade_metrics['Orders_Filled_MM'], trade_metrics['Orders_Filled_FR']],
                [trade_metrics['Traded_Instruments_MM'], trade_metrics['Traded_Instruments_FR']],
                [f"{trade_metrics['Traded_Instruments_MM'] / total_instrument:.2%}" if isinstance(
                    trade_metrics['Traded_Instruments_MM'], int) else 'N/A', ''],
                [total_instrument, ''],
                [f"{trade_metrics['Total_Turnover_MM']:,}" if isinstance(trade_metrics['Total_Turnover_MM'],
                                                                         (int, float)) else 'N/A',
                 f"{trade_metrics['Total_Turnover_FR']:,}" if isinstance(trade_metrics['Total_Turnover_FR'],
                                                                         (int, float)) else 'N/A'],
                [f"{trade_metrics['Accepted_RFQ_Turnover_MM']:,}" if isinstance(
                    trade_metrics['Accepted_RFQ_Turnover_MM'], (int, float)) else 'N/A',
                 f"{trade_metrics['Accepted_RFQ_Turnover_FR']:,}" if isinstance(
                     trade_metrics['Accepted_RFQ_Turnover_FR'], (int, float)) else 'N/A'],
                [trade_metrics['Accepted_RFQ_Buy_Turnover_MM'], trade_metrics['Accepted_RFQ_Buy_Turnover_FR']],
                ['N/A' if trade_metrics[
                              'Accepted_RFQ_Buy_Turnover_MM'] == 'N/A' else f"{1 - float(trade_metrics['Accepted_RFQ_Buy_Turnover_MM'].strip('%')) / 100:.2%}",
                 'N/A' if trade_metrics[
                              'Accepted_RFQ_Buy_Turnover_FR'] == 'N/A' else f"{1 - float(trade_metrics['Accepted_RFQ_Buy_Turnover_FR'].strip('%')) / 100:.2%}"],
                [f"{round(trade_metrics['Accepted_RFQ_Turnover_MM'] / trade_metrics['Orders_Filled_MM'] if trade_metrics['Orders_Filled_MM'] not in ['N/A', 0] else 0):,.0f}" if isinstance(
                    trade_metrics['Accepted_RFQ_Turnover_MM'], (int, float)) else 'N/A', ''],
                [f"{trade_metrics['Total_Turnover_MM'] - trade_metrics['Accepted_RFQ_Turnover_MM']:,}" if isinstance(
                    trade_metrics['Total_Turnover_MM'], (int, float)) else 'N/A',
                 f"{trade_metrics['Total_Turnover_FR'] - trade_metrics['Accepted_RFQ_Turnover_FR']:,}" if isinstance(
                     trade_metrics['Total_Turnover_FR'], (int, float)) else 'N/A']
            ]
            row_names_Trading = ['Orders Filled', 'Traded Instruments', 'Traded Instruments %', 'Total Instruments',
                                 'Total Turnover (€)', 'Accepted RFQ Turnover (€)', 'Accepted RFQ Buy Turnover %',
                                 'Accepted RFQ Sell Turnover %', 'Average Accepted Order (€)', 'Hedge Volume (€)']
            df_Trading_Statistics = pd.DataFrame(data=Statistics_data, index=row_names_Trading,
                                                 columns=['Market Making', 'Fractional'])
            df_Trading_Statistics.index.name = "Trading Statistics"

            PnL_data = [
                f"{Daily_PnL_MM:,.0f}", f"{Daily_PnL_MA5:,.0f}", f"{Daily_PnL_adj:,.0f}",
                trade_metrics['PnL_per_RFQ'], trade_metrics['PnL_per_RFQ_Turnover'],
                trade_metrics['Net_PnL_per_RFQ_Turnover'],
                f"{Total_Mtd_PnL_MM:,.0f}", f"{Total_Ytd_PnL_MM:,.0f}", f"{Gross_Exposure_MM + Gross_Exposure_FR:,.0f}",
                f"{Net_Exposure_MM + Net_Exposure_FR:,.0f}"
            ]
            row_names_PnL = ['Daily PnL (€)', 'Daily PnL (MA5) (€)', 'Total Daily PnL (€) Adj', 'PnL per RFQ (€)',
                             'PnL per RFQ Turnover', 'Net PnL per RFQ Turnover', 'Total MtD PnL (€)',
                             'Total YtD PnL (€)', 'Gross Exposure (€)', 'Net Exposure (€)']
            df_PnL_Statistics = pd.DataFrame(data=PnL_data, index=row_names_PnL, columns=['Market Making'])
            df_PnL_Statistics.index.name = "PnL Statistics"

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
            win_rate_t0 = (df_PnL[~df_PnL['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                               'BTPLD'].sum() > 0).mean()
            win_rate_t1 = (df_t1[~df_t1['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                               'BTPLD'].sum() > 0).mean()
            win_rate_t2 = (df_t2[~df_t2['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                               'BTPLD'].sum() > 0).mean()
            win_rate_t3 = (df_t3[~df_t3['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                               'BTPLD'].sum() > 0).mean()
            win_rate_t4 = (df_t4[~df_t4['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                               'BTPLD'].sum() > 0).mean()
            win_rate_data = {'Term': ['T-4', 'T-3', 'T-2', 'T-1', 'T-0'],
                             'Win_Rate': [win_rate_t4, win_rate_t3, win_rate_t2, win_rate_t1, win_rate_t0]}
            ytd_pnl = {
                term: round(df.loc[(df['Portfolio.Name'] == 'SCALMM') & (df['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
                for term, df in historical_pnl.items()}
            daily_intraday_pnl = {'T-0': Daily_PnL_MM, 'T-1': Daily_PnL_t1, 'T-2': Daily_PnL_t2, 'T-3': Daily_PnL_t3,
                                  'T-4': Daily_PnL_t4}
            overnight_pnl_data = {}
            for i in range(1, 5):
                term, prev_term = f'T-{i}', f'T-{i + 1}'
                daily_adj = ytd_pnl[term] - ytd_pnl[prev_term]
                overnight_pnl_data[term] = daily_adj - daily_intraday_pnl[term]

            # --- Display Results ---
            st.subheader("📋 Copy-Friendly Report")
            st.markdown("""
                <a href="https://docs.google.com/document/d/1ij66_05uM6PPmSWehV3Pk6bfTL9FgOpd3tRIGeo4pVo/edit?usp=sharing" target="_blank">Click here for the email template</a>
                """, unsafe_allow_html=True)
            st.info(
                "You can right-click on any chart or table below and select 'Copy Image' to paste it in your report mail.")

            # --- NEW/MODIFIED CODE BLOCK STARTS HERE ---
            # Generate and display the PnL statistics table
            pnl_table_png = create_table_image(
                df=df_PnL_Statistics,
                width=PNL_TABLE_WIDTH,
                height=PNL_TABLE_HEIGHT,
                col_widths=[2, 1]
            )
            st.image(pnl_table_png, caption="PnL Statistics")

            # Generate and display the Trading statistics table
            trading_table_png = create_table_image(
                df=df_Trading_Statistics,
                width=TRADING_TABLE_WIDTH,
                height=TRADING_TABLE_HEIGHT,
                col_widths=[2, 1, 1]
            )
            st.image(trading_table_png, caption="Trading Statistics")
            # --- NEW/MODIFIED CODE BLOCK ENDS HERE ---

            st.divider()

            # --- Display Plots Sequentially ---
            pnl_plot_png = create_pnl_plot(daily_intraday_pnl, overnight_pnl_data, win_rate_data, df_date,
                                           width=PNL_PLOT_WIDTH, height=PNL_PLOT_HEIGHT)
            st.image(pnl_plot_png, caption="Daily PnL Breakdown")

            if df_trades is not None and all(
                    df is not None for df in [df_trades_t1, df_trades_t2, df_trades_t3, df_trades_t4]):
                historical_trades = {'T-1': df_trades_t1, 'T-2': df_trades_t2, 'T-3': df_trades_t3, 'T-4': df_trades_t4}
                total_turnover_data = {'T-0': trade_metrics['Total_Turnover_MM']}
                accepted_turnover_data = {'T-0': trade_metrics['Accepted_RFQ_Turnover_MM']}
                for term, df_hist_trade in historical_trades.items():
                    total_turnover_data[term] = round(
                        df_hist_trade[df_hist_trade['Portfolio'] == 'SCALMM']['Premium'].abs().sum())
                    accepted_turnover_data[term] = round(df_hist_trade[(df_hist_trade['Portfolio'] == 'SCALMM') & (
                        df_hist_trade['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))][
                                                             'Premium'].abs().sum())

                turnover_plot_png = create_turnover_plot(total_turnover_data, accepted_turnover_data, df_date,
                                                         width=TURNOVER_PLOT_WIDTH, height=TURNOVER_PLOT_HEIGHT)
                st.image(turnover_plot_png, caption="Daily Turnover Breakdown")

                top_performers_plot_png = create_top_performers_plot(df_PnL, df_trades, df_Instrument,
                                                                     width=TOP_PERF_PLOT_WIDTH,
                                                                     height=TOP_PERF_PLOT_HEIGHT)
                st.image(top_performers_plot_png, caption="Top 10 Winners & Losers by PnL")
            else:
                st.warning(
                    "One or more Trades reports were not found, so Turnover and Top Performer charts cannot be generated.")

            st.divider()

            update_log("🎉 Report generated successfully!", log_messages, log_placeholder, "SUCCESS")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.exception(e)
else:
    update_log("Enter parameters above and click 'Generate Report' to start.", log_messages, log_placeholder)