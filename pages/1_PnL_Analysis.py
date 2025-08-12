import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timezone
from io import StringIO
import re

# --- Page Configuration ---
st.set_page_config(page_title="PnL Analysis & Report", layout="wide")

# --- UPDATED CSS ---
# We've simplified the styling to use a single container for the report.
# I've also added a light gray background to the main app to make the white report "pop".
# --- UPDATED CSS ---
st.markdown("""
<style>
    .main-header {
        color: #AFC7CA !important;
    }
    /* This sets the entire app background to white */
    .stApp {
        background-color: white;
    }
    /* This container will still give your report section a nice border and shadow */
    .report-page-container {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e6e6e6; /* A subtle border for definition */
    }
</style>
""", unsafe_allow_html=True)

# Apply the custom class to the title
st.markdown('<h1 class="main-header">📈 Daily PnL Analysis & Report</h1>', unsafe_allow_html=True)


# --- Helper Functions (No changes needed here) ---

def style_dataframe_for_email(df: pd.DataFrame) -> str:
    """
    Applies CSS styling to a DataFrame and returns it as an HTML string.
    Hides the index. Sets white background, black text and borders, Arial font.
    """
    styles = [
        dict(selector="", props=[("border-collapse", "collapse"),
                                 ("font-family", "Arial, sans-serif")]),
        dict(selector="th",
             props=[("color", "black"), ("font-weight", "bold"), ("padding", "10px"),
                    ("border", "1px solid black"), ("text-align", "center"),
                    ("background-color", "white"), ("font-family", "Arial, sans-serif"),
                    ("font-size", "10px")]),
        dict(selector="td", props=[("padding", "8px"), ("border", "1px solid black"),
                                   ("text-align", "right"), ("background-color", "white"),
                                   ("font-family", "Arial, sans-serif"), ("font-size", "10px"),
                                   ("color", "black")])
    ]
    styled_df = df.style.set_table_styles(styles).format(precision=2).hide(axis="index")
    return styled_df.to_html()


def style_summary_df(df: pd.DataFrame) -> str:
    """
    Applies CSS styling to a DataFrame and returns it as an HTML string.
    Keeps the index visible. Sets white background, black text and borders, Arial font.
    """
    styles = [
        dict(selector="", props=[("border-collapse", "collapse"),
                                 ("font-family", "Arial, sans-serif")]),
        dict(selector="th.col_heading",
             props=[("color", "black"), ("font-weight", "bold"), ("padding", "10px"),
                    ("border", "1px solid black"), ("text-align", "center"),
                    ("background-color", "white"), ("font-family", "Arial, sans-serif"),
                    ("font-size", "10px")]),
        dict(selector="th.col_heading.level0",
             props=[("color", "black"), ("font-weight", "bold"), ("padding", "10px"),
                    ("border", "1px solid black"), ("text-align", "center"),
                    ("background-color", "white"), ("font-family", "Arial, sans-serif"),
                    ("font-size", "10px")]),
        dict(selector="th.row_heading",
             props=[("padding", "8px"), ("border", "1px solid black"), ("text-align", "left"),
                    ("font-weight", "bold"), ("background-color", "white"),
                    ("font-family", "Arial, sans-serif"), ("font-size", "10px"),
                    ("color", "black")]),
        dict(selector="td", props=[("padding", "8px"), ("border", "1px solid black"),
                                   ("text-align", "right"), ("background-color", "white"),
                                   ("font-family", "Arial, sans-serif"), ("font-size", "10px"),
                                   ("color", "black")])
    ]
    styled_df = df.style.set_table_styles(styles).format(precision=2)
    return styled_df.to_html()


def get_popup_window_button(html_content: str) -> str:
    """
    Generates HTML/JS for a button that opens the given HTML content in a new browser window.
    """
    # Safely embed the HTML content into a JavaScript string variable
    escaped_html = json.dumps(html_content)

    button_html = f'''
        <button
            id="newWindowBtn"
            style="padding: 10px 15px; border-radius: 5px; background-color: #007bff; color: white; border: none; cursor: pointer; font-size: 16px;"
            onclick="openReportWindow()"
        >
            🚀 Open Report in New Window for Copying
        </button>

        <script>
        function openReportWindow() {{
            const reportHtml = {escaped_html};
            const newWindow = window.open("", "_blank", "width=800,height=600");
            if (newWindow) {{
                newWindow.document.write(reportHtml);
                newWindow.document.close();
            }} else {{
                alert("Popup was blocked. Please allow popups for this site to open the report.");
            }}
        }}
        </script>
    '''
    return button_html


def get_copy_area_for_report(html_content: str) -> str:
    """
    Creates a selectable text area with report content for manual copying.
    """
    # Process HTML to extract just the data
    text_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL)
    text_content = re.sub(r'<script.*?</script>', '', text_content, flags=re.DOTALL)
    text_content = re.sub(r'<[^>]*>', ' ', text_content)
    text_content = re.sub(r'\s+', ' ', text_content)
    text_content = text_content.strip()

    return f"""
    <div style="margin-top: 20px; margin-bottom: 20px;">
        <p style="font-weight: bold;">📋 Select all text below and copy manually:</p>
        <textarea id="reportTextArea" style="width: 100%; height: 200px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-family: monospace; margin-top: 10px;">
{text_content}
        </textarea>
    </div>
    """


def get_reports_in_memory(date: str, bearer_token: str, status_placeholder) -> dict:
    """
    Finds and fetches a list of daily report files, returning their content in memory.
    """
    reports_to_find = [
        {"report_name": "ProfitandLoss", "name_format": "{date}-{time}FIS_EOD_{report_name}.csv"},
        {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"}
    ]
    headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
    downloaded_data = {}
    base_url = 'https://franz.agent-tool.scalable.capital/agent/s3-access/buckets/prod-market-making.trade-reports.scalable-fis/download'

    for report in reports_to_find:
        st.info(f"Searching for report: {report['report_name']}...")
        found_file = False
        for hour in range(18, 24):
            for minute in range(60):
                time_str = f"{hour:02d}{minute:02d}"
                file_key = f'fis/{report["name_format"].format(date=date, time=time_str, report_name=report["report_name"])}'
                payload = {'key': file_key}
                status_placeholder.text(f"Attempting to fetch: {file_key}")
                try:
                    response = requests.post(base_url, headers=headers, json=payload)
                    if response.status_code in [200, 201]:
                        st.success(f"Found and downloaded: {file_key}")
                        downloaded_data[report["report_name"]] = response.content
                        found_file = True
                        break
                except requests.RequestException as e:
                    st.error(f"An error occurred: {e}")
                    return {}
            if found_file: break
        if not found_file: st.warning(f"Could not find report '{report['report_name']}' for date {date}.")
    status_placeholder.empty()
    return downloaded_data


# --- Streamlit User Interface ---

bearer_token = st.text_input(
    "Enter your Bearer Token",
    type="password",
    help="Need help finding the token? [Franz Agent Tool](https://franz.agent-tool.scalable.capital/misc/s3-access/prod-market-making_trade-reports_scalable-fis?bucketPath=fis%2F)"
)
report_date = st.date_input("Select Report Date", value=datetime.now(timezone.utc))

if st.button("Fetch and Analyze Reports"):
    if not bearer_token:
        st.warning("Please enter your Bearer Token.")
    else:
        date_str = report_date.strftime("%Y%m%d")
        status_placeholder = st.empty()

        with st.spinner("Searching for report files... This may take a few minutes."):
            downloaded_reports = get_reports_in_memory(date_str, bearer_token, status_placeholder)

        if downloaded_reports and "ProfitandLoss" in downloaded_reports and "Trades" in downloaded_reports:
            st.success("All reports fetched and processed successfully!")
            try:
                # --- 1. Load all data into DataFrames ---
                df_pnl = pd.read_csv(StringIO(downloaded_reports["ProfitandLoss"].decode('utf-8')))
                df_trades = pd.read_csv(StringIO(downloaded_reports["Trades"].decode('utf-8')))
                try:
                    df_Instrument = pd.read_csv("Instrument list.csv")
                except FileNotFoundError:
                    st.error(
                        "Error: 'Instrument list.csv' not found. Please place it in the same directory as the script.")
                    st.stop()

                # --- 2. Calculations (No changes needed here) ---
                total_instrument = 5238
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
                if Accepted_RFQ_Turnover_MM > 0:
                    Accepted_RFQ_Buy_Turnover_MM = round(df_trades[(df_trades['Portfolio'] == 'SCALMM') & (
                        df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                                           df_trades['B/S'] == 'Buy')][
                                                             'Premium'].abs().sum() / Accepted_RFQ_Turnover_MM, 4)
                else:
                    Accepted_RFQ_Buy_Turnover_MM = 0
                if Accepted_RFQ_Turnover_FR > 0:
                    Accepted_RFQ_Buy_Turnover_FR = round(df_trades[(df_trades['Portfolio'] == 'SCALFRAC') & (
                        df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])) & (
                                                                           df_trades['B/S'] == 'Buy')][
                                                             'Premium'].abs().sum() / Accepted_RFQ_Turnover_FR, 4)
                else:
                    Accepted_RFQ_Buy_Turnover_FR = 0

                Daily_PnL_MM = round(df_pnl[df_pnl['Unnamed: 0'] == "SCALMM"]["BTPLD"].iloc[0])
                Daily_PnL_FR = round(df_pnl[df_pnl['Unnamed: 0'] == "SCALFRAC"]["BTPLD"].iloc[0])
                PnL_per_RFQ = round(Daily_PnL_MM / Orders_Filled_MM, 2) if Orders_Filled_MM > 0 else 0
                PnL_per_RFQ_Turnover = f"{round((Daily_PnL_MM / Accepted_RFQ_Turnover_MM) * 10000, 2)}bp" if Accepted_RFQ_Turnover_MM > 0 else "0.00bp"
                Estimated_Execution_Fees_MM = round((Total_Turnover_MM - Accepted_RFQ_Turnover_MM) * 0.00007 * 2 / 3, 2)
                Estimated_Execution_Fees_FR = round((Total_Turnover_FR - Accepted_RFQ_Turnover_FR) * 0.00007, 2)
                Net_PnL_per_RFQ_Turnover = f"{round(((Daily_PnL_MM - Estimated_Execution_Fees_MM) / Accepted_RFQ_Turnover_MM) * 10000, 2)}bp" if Accepted_RFQ_Turnover_MM > 0 else "0.00bp"
                Total_Mtd_PnL_MM = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALMM') & (
                        df_pnl['Unnamed: 0'] != 'SCALMM'), 'BTPLM'].sum())
                Total_Mtd_PnL_FR = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALFRAC') & (
                        df_pnl['Unnamed: 0'] != 'SCALFRAC'), 'BTPLM'].sum())
                Total_Ytd_PnL_MM = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALMM') & (
                        df_pnl['Unnamed: 0'] != 'SCALMM'), 'BTPL'].sum())
                Total_Ytd_PnL_FR = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALFRAC') & (
                        df_pnl['Unnamed: 0'] != 'SCALFRAC'), 'BTPL'].sum())

                Gross_Exposure_MM = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALMM') & (
                        df_pnl['Unnamed: 0'] != 'SCALMM'), 'Gross Exposure'].sum())
                Gross_Exposure_FR = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALFRAC') & (
                        df_pnl['Unnamed: 0'] != 'SCALFRAC'), 'Gross Exposure'].sum())
                Net_Exposure_MM = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALMM') & (
                        df_pnl['Unnamed: 0'] != 'SCALMM'), 'BVal'].sum())
                Net_Exposure_FR = round(df_pnl.loc[(df_pnl['Portfolio.Name'] == 'SCALFRAC') & (
                        df_pnl['Unnamed: 0'] != 'SCALFRAC'), 'BVal'].sum())

                # --- Create DataFrames (No changes needed here) ---
                df_Trading_Statistics = pd.DataFrame(data={
                    'Market Making (SCALMM)': [Orders_Filled_MM, Traded_Instruments_MM,
                                               f"{Traded_Instruments_MM / total_instrument:.2%}", total_instrument,
                                               Total_Turnover_MM, Accepted_RFQ_Turnover_MM,
                                               f"{Accepted_RFQ_Buy_Turnover_MM:.2%}",
                                               f"{1 - Accepted_RFQ_Buy_Turnover_MM:.2%}", round(
                            Accepted_RFQ_Turnover_MM / Orders_Filled_MM if Orders_Filled_MM > 0 else 0),
                                               Total_Turnover_MM - Accepted_RFQ_Turnover_MM],
                    'Fractional (SCALFRAC)': [Orders_Filled_FR, Traded_Instruments_FR, '', '', Total_Turnover_FR,
                                              Accepted_RFQ_Turnover_FR, f"{Accepted_RFQ_Buy_Turnover_FR:.2%}",
                                              f"{1 - Accepted_RFQ_Buy_Turnover_FR:.2%}", '',
                                              Total_Turnover_FR - Accepted_RFQ_Turnover_FR]},
                    index=['Orders Filled', 'Traded Instruments',
                           'Coverage % by Instrument Type', 'Total Instruments',
                           'Total Turnover (€)', 'Accepted RFQ Turnover (€)',
                           'Accepted RFQ Buy %', 'Accepted RFQ Sell %',
                           'Average Accepted RFQ Trade Size (€)',
                           'Avg Auto-Hedging Turnover (€)'])
                df_PnL_Statistics = pd.DataFrame(data={
                    'Market Making (SCALMM)': [Daily_PnL_MM, '', PnL_per_RFQ, PnL_per_RFQ_Turnover,
                                               Net_PnL_per_RFQ_Turnover, Estimated_Execution_Fees_MM, Total_Mtd_PnL_MM,
                                               Total_Ytd_PnL_MM],
                    'Fractional (SCALFRAC)': [Daily_PnL_FR, '', '', '', '', Estimated_Execution_Fees_FR,
                                              Total_Mtd_PnL_FR, Total_Ytd_PnL_FR]},
                    index=['Daily PnL (€)', 'Total Daily PnL (€) Adj***',
                           'PnL per RFQ (€/RFQ)', 'PnL per Accepted RFQ Turnover',
                           'Net PnL per Accepted RFQ Turnover',
                           'Estimated Execution Fees (€)', 'MTD PnL (€)', 'YTD PnL (€)'])
                df_Delta = pd.DataFrame(data={'Market Making (SCALMM)': [Gross_Exposure_MM, Net_Exposure_MM],
                                              'Fractional (SCALFRAC)': [Gross_Exposure_FR, Net_Exposure_FR]},
                                        index=['Gross Exposure (€)', 'Net Exposure (€)'])

                df_top_5_trades = pd.merge(
                    df_trades[df_trades['Cpty'].isin(['CLIENTALLOCATION', 'LOMSCALOFPS', 'CorpAction'])].assign(
                        AbsPremium=lambda df: df['Premium'].abs()).nlargest(5, 'AbsPremium')[
                        ['Instrument', 'AbsPremium']], df_Instrument, left_on='Instrument', right_on='ISIN',
                    how='left').rename(columns={'Name': 'Security Name', 'AbsPremium': 'Value (€)'})[
                    ['ISIN', 'Security Name', 'Instrument Type', 'Value (€)']]
                df_top_5_mosted_trades = pd.merge(
                    df_trades.groupby('Instrument')['Premium'].apply(lambda x: x.abs().sum()).nlargest(5).rename(
                        'Total_Turnover'), df_Instrument, left_on='Instrument', right_on='ISIN', how='left').rename(
                    columns={'Name': 'Security Name', 'Total_Turnover': 'Turnover (€)'})[
                    ['ISIN', 'Security Name', 'Instrument Type', 'Turnover (€)']]
                df_top_10_winners = pd.merge(
                    df_pnl[~df_pnl['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')['BTPLD'].apply(
                        lambda x: x.sum()).nlargest(10).rename('Total_PnL'), df_Instrument, left_on='Unnamed: 0',
                    right_on='ISIN', how='left').rename(
                    columns={'Name': 'Security Name', 'Total_PnL': 'Total PnL (€)'})[
                    ['ISIN', 'Security Name', 'Instrument Type', 'Total PnL (€)']]
                df_top_10_losers = pd.merge(
                    df_pnl[~df_pnl['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')['BTPLD'].apply(
                        lambda x: x.sum()).nsmallest(10).rename('Total_PnL'), df_Instrument, left_on='Unnamed: 0',
                    right_on='ISIN', how='left').rename(
                    columns={'Name': 'Security Name', 'Total_PnL': 'Total PnL (€)'})[
                    ['ISIN', 'Security Name', 'Instrument Type', 'Total PnL (€)']]
                df_top_5_largest_positions = pd.merge(
                    df_pnl[~df_pnl['Unnamed: 0'].isin(['SCALMM', 'SCALFRAC'])].groupby('Unnamed: 0')[
                        'Gross Exposure'].apply(lambda x: x.sum()).nlargest(5).rename('GP'), df_Instrument,
                    left_on='Unnamed: 0', right_on='ISIN', how='left').rename(
                    columns={'Name': 'Security Name', 'GP': 'Open Delta Exposure (€)'})[
                    ['ISIN', 'Security Name', 'Instrument Type', 'Open Delta Exposure (€)']]


                # --- 3. Display all data on a single page (UPDATED SECTION) ---
                # We've simplified this section to use the single report container.
                st.header(f"Daily Report: {report_date.strftime('%B %d, %Y')}")
                st.divider()

                # This markdown injects the opening tag of our styled container.
                # All Streamlit elements that follow will be placed inside this div.
                st.markdown('<div class="report-page-container">', unsafe_allow_html=True)

                st.subheader("Report Preview")

                # Display all the dataframes within the container
                st.markdown("<h4>PnL Statistics</h4>", unsafe_allow_html=True)
                st.markdown(style_summary_df(df_PnL_Statistics), unsafe_allow_html=True)
                st.markdown("<h4>Delta Exposure</h4>", unsafe_allow_html=True)
                st.markdown(style_summary_df(df_Delta), unsafe_allow_html=True)
                st.markdown("<h4>Trading Statistics</h4>", unsafe_allow_html=True)
                st.markdown(style_summary_df(df_Trading_Statistics), unsafe_allow_html=True)
                st.markdown("<h4>Top 5 Largest Positions by Gross Exposure</h4>", unsafe_allow_html=True)
                st.markdown(style_dataframe_for_email(df_top_5_largest_positions), unsafe_allow_html=True)
                st.markdown("<h4>Top 5 Most Traded Instruments by Turnover</h4>", unsafe_allow_html=True)
                st.markdown(style_dataframe_for_email(df_top_5_mosted_trades), unsafe_allow_html=True)
                st.markdown("<h4>Top 5 Trades by Value</h4>", unsafe_allow_html=True)
                st.markdown(style_dataframe_for_email(df_top_5_trades), unsafe_allow_html=True)
                st.markdown("<h4>Top 10 Winners</h4>", unsafe_allow_html=True)
                st.markdown(style_dataframe_for_email(df_top_10_winners), unsafe_allow_html=True)
                st.markdown("<h4>Top 10 Losers</h4>", unsafe_allow_html=True)
                st.markdown(style_dataframe_for_email(df_top_10_losers), unsafe_allow_html=True)


                # This markdown injects the closing tag for our container.
                st.markdown('</div>', unsafe_allow_html=True)


            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                import traceback

                st.error(traceback.format_exc())
        else:
            st.error("Could not retrieve all necessary reports. Please check the logs above.")