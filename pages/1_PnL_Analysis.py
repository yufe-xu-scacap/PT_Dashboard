import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import io
from typing import Dict, Optional, List
import exchange_calendars as xcals
from zoneinfo import ZoneInfo
import plotly.graph_objects as go          # already used below
import plotly.io as pio                    # to export Plotly → PNG
import matplotlib.ticker as mticker        # already used further down
import matplotlib.pyplot as plt            # already used further down


# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Trading Report Dashboard")
st.title("📈 Trading Report Dashboard")


# --- Core Data Fetching Function ---
def get_reports(date: str, bearer_token: str, log_list: List[str], log_placeholder) -> Dict[str, pd.DataFrame]:
    """
    Retrieves daily and historical reports, appending feedback to the provided log_list.
    """

    def _update_log(message: str, level: str = "INFO"):
        """Helper function to update the log list and the streamlit placeholder."""
        berlin_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%H:%M:%S')
        log_list.append(f"[{berlin_time}] {level}: {message}")
        log_placeholder.markdown(f"```\n{' \n'.join(log_list[-10:])}\n```")

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
                    headers=headers, json=payload, timeout=20
                )
                if response.status_code in [200, 201]:
                    _update_log(f"✅ Found and retrieved: {server_filename}", "SUCCESS")
                    try:
                        return pd.read_csv(io.BytesIO(response.content))
                    except Exception as e:
                        _update_log(f"Failed to parse CSV content for {server_filename}: {e}", "ERROR")
                        return None
            except requests.exceptions.RequestException:
                pass
            current_dt += timedelta(minutes=1)

        return None

    date_mapping_data = []
    xetr_cal = xcals.get_calendar("XETR")
    current_date_ts = pd.to_datetime(date_str).normalize()

    _update_log(f"Searching for P&L reports...")
    df_t0 = _fetch_report(date_str,
                          {"report_name": "ProfitandLoss", "name_format": "{date}-{time}FIS_EOD_{report_name}.csv"},
                          "1805", "1810")
    if df_t0 is not None:
        report_dataframes['ProfitandLoss'] = df_t0.copy()
    else:
        _update_log(f"P&L report for {date_str} (T-0) not found.", "ERROR")

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
            _update_log(f"P&L report for {prev_day_str} (T-{i}) not found.", "ERROR")

    _update_log(f"Searching for Trades reports...")
    trades_df = _fetch_report(date_str,
                              {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"}, "1808",
                              "1810")
    if trades_df is not None:
        report_dataframes['Trades'] = trades_df.copy()
    else:
        _update_log(f"Trades report for {date_str} (T-0) not found.", "ERROR")

    prev_day_ts = current_date_ts
    for i in range(1, 5):
        prev_day_ts = xetr_cal.previous_session(prev_day_ts)
        prev_day_str = prev_day_ts.strftime('%Y%m%d')
        df_trade_hist = _fetch_report(prev_day_str,
                                      {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"},
                                      "1807", "1810")
        if df_trade_hist is not None:
            report_dataframes[f"Trades_T-{i}"] = df_trade_hist.copy()
        else:
            _update_log(f"Trades report for {prev_day_str} (T-{i}) not found.", "ERROR")

    _update_log("All search operations complete.")
    return report_dataframes


# --- User Inputs on Main Page ---
st.subheader("⚙️ Report Parameters")
col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

with col1:
    bearer_token = st.text_input("Enter Bearer Token", type="password", help="Your secret authentication token.")
with col2:
    report_date = st.date_input("Select Report Date", value=datetime(2025, 8, 11))
with col3:
    total_instrument = st.number_input("Total Instruments", min_value=1, value=5238,
                                       help="Set the total number of instruments.")
with col4:
    st.write("&#8203;")
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
            log_messages.clear()
            log_placeholder.info("⏳ Initializing... Reading local files.")
            try:
                df_Instrument = pd.read_csv("Instrument list.csv")
            except FileNotFoundError:
                log_placeholder.error("Error: 'Instrument list.csv' not found.")
                st.stop()

            date_str = report_date.strftime("%Y%m%d")

            successfully_downloaded = get_reports(date_str, bearer_token, log_messages, log_placeholder)

            required_dfs = ["ProfitandLoss", "ProfitandLoss_T-1", "ProfitandLoss_T-2", "ProfitandLoss_T-3",
                            "ProfitandLoss_T-4", "ProfitandLoss_T-5", "Trades", "Trades_T-1", "Trades_T-2",
                            "Trades_T-3", "Trades_T-4"]
            missing_dfs = [df for df in required_dfs if df not in successfully_downloaded]
            if missing_dfs:
                log_expander.error(f"FATAL: Could not retrieve essential reports: {', '.join(missing_dfs)}.")
                st.stop()

            log_expander.success("✔️ All required data retrieved. Performing calculations...")

            df_PnL = successfully_downloaded["ProfitandLoss"]
            df_t1, df_t2, df_t3, df_t4, df_t5 = (successfully_downloaded[f"ProfitandLoss_T-{i}"] for i in range(1, 6))
            df_trades = successfully_downloaded["Trades"]
            df_trades_t1, df_trades_t2, df_trades_t3, df_trades_t4 = (successfully_downloaded[f"Trades_T-{i}"] for i in
                                                                      range(1, 5))

            # --- Calculations ---
            Orders_Filled_MM = ((df_trades["Portfolio"] == "SCALMM") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))).sum()
            Orders_Filled_FR = ((df_trades["Portfolio"] == "SCALFRAC") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))).sum()
            Traded_Instruments_MM = df_trades[(df_trades["Portfolio"] == "SCALMM") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Instrument'].nunique()
            Traded_Instruments_FR = df_trades[(df_trades["Portfolio"] == "SCALFRAC") & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Instrument'].nunique()
            Total_Turnover_MM = round(df_trades[df_trades['Portfolio'] == 'SCALMM']['Premium'].abs().sum())
            Total_Turnover_FR = round(df_trades[df_trades['Portfolio'] == 'SCALFRAC']['Premium'].abs().sum())
            Accepted_RFQ_Turnover_MM = round(df_trades[(df_trades['Portfolio'] == 'SCALMM') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum())
            Accepted_RFQ_Turnover_FR = round(df_trades[(df_trades['Portfolio'] == 'SCALFRAC') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction']))]['Premium'].abs().sum())
            buy_turnover_mm_total = df_trades[(df_trades['Portfolio'] == 'SCALMM') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                          df_trades['B/S'] == 'Buy')]['Premium'].abs().sum()
            Accepted_RFQ_Buy_Turnover_MM = round(buy_turnover_mm_total / Accepted_RFQ_Turnover_MM,
                                                 4) if Accepted_RFQ_Turnover_MM > 0 else 0
            buy_turnover_fr_total = df_trades[(df_trades['Portfolio'] == 'SCALFRAC') & (
                df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                          df_trades['B/S'] == 'Buy')]['Premium'].abs().sum()
            Accepted_RFQ_Buy_Turnover_FR = round(buy_turnover_fr_total / Accepted_RFQ_Turnover_FR,
                                                 4) if Accepted_RFQ_Turnover_FR > 0 else 0
            Daily_PnL_MM = round(df_PnL[df_PnL['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
            Daily_PnL_t1 = round(df_t1[df_t1['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
            Daily_PnL_t2 = round(df_t2[df_t2['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
            Daily_PnL_t3 = round(df_t3[df_t3['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
            Daily_PnL_t4 = round(df_t4[df_t4['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
            Daily_PnL_MA5 = round((Daily_PnL_MM + Daily_PnL_t1 + Daily_PnL_t2 + Daily_PnL_t3 + Daily_PnL_t4) / 5)
            PnL_per_RFQ = round(Daily_PnL_MM / Orders_Filled_MM if Orders_Filled_MM > 0 else 0, 2)
            PnL_per_RFQ_Turnover = f"{round((Daily_PnL_MM / Accepted_RFQ_Turnover_MM) * 10000, 2)}bp" if Accepted_RFQ_Turnover_MM > 0 else "0.00bp"
            Estimated_Execution_Fees_MM = round((Total_Turnover_MM - Accepted_RFQ_Turnover_MM) * 0.00007 * 2 / 3, 2)
            Net_PnL_per_RFQ_Turnover = f"{round(((Daily_PnL_MM - Estimated_Execution_Fees_MM) / Accepted_RFQ_Turnover_MM) * 10000, 2)}bp" if Accepted_RFQ_Turnover_MM > 0 else "0.00bp"
            Total_Mtd_PnL_MM = round(
                df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BTPLM'].sum())
            Total_Ytd_PnL_MM = round(
                df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
            Total_Ytd_PnL_t1 = round(
                df_t1.loc[(df_t1['Portfolio.Name'] == 'SCALMM') & (df_t1['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
            Daily_PnL_adj = Total_Ytd_PnL_MM - Total_Ytd_PnL_t1
            Gross_Exposure_MM = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (
                        df_PnL['Unnamed: 0'] != 'SCALMM'), 'Gross Exposure'].sum())
            Gross_Exposure_FR = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALFRAC') & (
                        df_PnL['Unnamed: 0'] != 'SCALFRAC'), 'Gross Exposure'].sum())
            Net_Exposure_MM = round(
                df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALMM') & (df_PnL['Unnamed: 0'] != 'SCALMM'), 'BVal'].sum())
            Net_Exposure_FR = round(df_PnL.loc[(df_PnL['Portfolio.Name'] == 'SCALFRAC') & (
                        df_PnL['Unnamed: 0'] != 'SCALFRAC'), 'BVal'].sum())

            # --- Assemble Final DataFrames ---
            Statistics_data = [[Orders_Filled_MM, Orders_Filled_FR], [Traded_Instruments_MM, Traded_Instruments_FR],
                               [f"{Traded_Instruments_MM / total_instrument:.2%}", ''], [total_instrument, ''],
                               [f"{Total_Turnover_MM:,.0f}", f"{Total_Turnover_FR:,.0f}"],
                               [f"{Accepted_RFQ_Turnover_MM:,.0f}", f"{Accepted_RFQ_Turnover_FR:,.0f}"],
                               [f"{Accepted_RFQ_Buy_Turnover_MM:.2%}", f"{Accepted_RFQ_Buy_Turnover_FR:.2%}"],
                               [f"{1 - Accepted_RFQ_Buy_Turnover_MM:.2%}", f"{1 - Accepted_RFQ_Buy_Turnover_FR:.2%}"],
                               [f"{round(Accepted_RFQ_Turnover_MM / Orders_Filled_MM if Orders_Filled_MM > 0 else 0):,.0f}",
                                ''], [f"{Total_Turnover_MM - Accepted_RFQ_Turnover_MM:,.0f}",
                                      f"{Total_Turnover_FR - Accepted_RFQ_Turnover_FR:,.0f}"]]
            row_names_Trading = ['Orders Filled', 'Traded Instruments', 'Traded Instruments %', 'Total Instruments',
                                 'Total Turnover (€)', 'Accepted RFQ Turnover (€)', 'Accepted RFQ Buy Turnover %',
                                 'Accepted RFQ Sell Turnover %', 'Average Accepted Order (€)', 'Hedge Volume (€)']
            df_Trading_Statistics = pd.DataFrame(data=Statistics_data, index=row_names_Trading,
                                                 columns=['Market Making (SCALMM)', 'Fractional (SCALFRAC)'])

            PnL_data = [f"{Daily_PnL_MM:,.0f}", f"{Daily_PnL_MA5:,.0f}", f"{Daily_PnL_adj:,.0f}", f"{PnL_per_RFQ:,.2f}",
                        PnL_per_RFQ_Turnover, Net_PnL_per_RFQ_Turnover, f"{Total_Mtd_PnL_MM:,.0f}",
                        f"{Total_Ytd_PnL_MM:,.0f}", f"{Gross_Exposure_MM + Gross_Exposure_FR:,.0f}",
                        f"{Net_Exposure_MM + Net_Exposure_FR:,.0f}"]
            row_names_PnL = ['Daily PnL (€)', 'Daily PnL (MA5) (€)', 'Total Daily PnL (€) Adj', 'PnL per RFQ (€)',
                             'PnL per RFQ Turnover', 'Net PnL per RFQ Turnover', 'Total MtD PnL (€)',
                             'Total YtD PnL (€)', 'Gross Exposure (€)', 'Net Exposure (€)']
            df_PnL_Statistics = pd.DataFrame(data=PnL_data, index=row_names_PnL, columns=['Market Making (SCALMM)'])

            # --- Display Results ---
            st.subheader("📊 Report Results")

            # Display tables side-by-side in columns
            col_pnl, col_trading = st.columns(2)
            with col_pnl:
                st.markdown("##### PnL Statistics")
                st.dataframe(df_PnL_Statistics)
            with col_trading:
                st.markdown("##### Trading Statistics")
                st.dataframe(df_Trading_Statistics)

            # --- Display copy-friendly text areas ---
            st.subheader("📋 Copy-Friendly Report")
            st.info(
                "Click inside the box below, then press Ctrl+A (or Cmd+A) and Ctrl+C (or Cmd+C) to copy only the data values.")
            st.info(
                "Data is provided in two separate boxes below. Copy the contents of each box and paste it into the corresponding section of your spreadsheet.")

            pnl_tsv = df_PnL_Statistics.to_csv(sep='\t', index=False, header=False)
            trading_tsv = df_Trading_Statistics.to_csv(sep='\t', index=False, header=False)

            # Create columns for the copy-friendly text areas
            col_pnl_copy, col_trading_copy = st.columns(2)
            with col_pnl_copy:
                st.markdown("###### PnL Statistics Data")
                st.text_area("PnL Data (for pasting)", pnl_tsv, height=300)

            with col_trading_copy:
                st.markdown("###### Trading Statistics Data")
                st.text_area("Trading Data (for pasting)", trading_tsv, height=300)

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.exception(e)

else:
    log_placeholder.info("Enter parameters above and click 'Generate Report' to start.")