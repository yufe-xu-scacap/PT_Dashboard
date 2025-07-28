import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
from io import StringIO

# --- Page Configuration ---
st.set_page_config(page_title="Trade Report Downloader", layout="wide")
st.title("📊 Trade Report Downloader")


# --- Reusable Function (Modified for In-Memory Handling) ---
def get_reports_in_memory(date: str, bearer_token: str, status_placeholder) -> dict:
    """
    Finds and fetches a list of daily report files, returning their content in memory.
    """
    reports_to_find = [
        {"report_name": "ProfitandLoss", "name_format": "{date}-{time}FIS_EOD_{report_name}.csv"},
        {"report_name": "Trades", "name_format": "{date}{time}FIS_EOD_{report_name}.csv"}
    ]
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json'
    }
    # This dictionary will store the report content
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

                    if response.status_code == 201 or response.status_code == 200:
                        st.success(f"Successfully fetched content for: {report['report_name']}")
                        # Store the file content in the dictionary instead of saving to disk
                        downloaded_data[report['report_name']] = response.content
                        found_file = True
                        break  # Exit minute loop

                except requests.RequestException as e:
                    st.error(f"An error occurred: {e}")
                    return {}  # Stop process on request error

            if found_file:
                break  # Exit hour loop

        if not found_file:
            st.warning(f"Could not find report '{report['report_name']}' for date {date}.")

    status_placeholder.empty()
    return downloaded_data


# --- Streamlit User Interface ---

bearer_token = st.text_input(
    "Enter your Bearer Token",
    type="password",
    help="Need help finding the token? [Click here for instructions](https://your-help-url.com)"
)
report_date = st.date_input("Select Report Date", value=datetime.now(timezone.utc))

if st.button("Fetch Reports"):
    if not bearer_token:
        st.warning("Please enter your Bearer Token.")
    else:
        date_str = report_date.strftime("%Y%m%d")
        status_placeholder = st.empty()

        with st.spinner("Searching for report files... This may take a few minutes."):
            downloaded_reports = get_reports_in_memory(date_str, bearer_token, status_placeholder)

        if downloaded_reports and "ProfitandLoss" in downloaded_reports and "Trades" in downloaded_reports:
            st.success("All reports fetched successfully!")

            try:
                # Read the Profit and Loss data from memory
                pnl_content = downloaded_reports["ProfitandLoss"].decode('utf-8')
                df_pnl = pd.read_csv(StringIO(pnl_content))
                st.subheader("Profit and Loss Report")
                st.dataframe(df_pnl)

                # Read the Trades data from memory
                trades_content = downloaded_reports["Trades"].decode('utf-8')
                df_trades = pd.read_csv(StringIO(trades_content))
                st.subheader("Trades Report")
                st.dataframe(df_trades)
            except Exception as e:
                st.error(f"Failed to process the in-memory data. Error: {e}")
        else:
            st.error("Could not retrieve all necessary reports. Please check the logs above.")