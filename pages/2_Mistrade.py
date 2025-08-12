import streamlit as st
import pandas as pd
import requests
import io
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qs
import pandas_market_calendars as mcal
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Mistrade Check & CSV Splitter",
    page_icon="🔎",
    layout="wide"
)

# ... (the rest of your code remains the same)

# --- Helper Functions ---

def extract_identifier_from_url(url: str) -> str | None:
    """
    Parses a URL to extract the unique file identifier from the 'bucketPath' parameter.
    """
    try:
        parsed_url = urlparse(url.strip())
        query_params = parse_qs(parsed_url.query)
        if 'bucketPath' not in query_params:
            st.error("URL is malformed or does not contain a 'bucketPath' parameter.")
            return None
        bucket_path = query_params['bucketPath'][0]
        identifier = os.path.basename(bucket_path)
        if not identifier:
            st.error("Could not extract a valid identifier from the bucketPath.")
            return None
        return identifier
    except (IndexError, KeyError) as e:
        st.error(f"Failed to parse URL. Please check the format is correct. Error: {e}")
        return None


@st.cache_data
def get_previous_trading_day() -> date:
    """
    Calculates the trading day before the most recent one (T-1).
    """
    xetra = mcal.get_calendar('XETR')
    end_date = date.today()
    start_date = end_date - timedelta(days=14)
    valid_days = xetra.valid_days(start_date=start_date, end_date=end_date)
    if len(valid_days) < 2:
        st.error(f"Could not determine the T-1 trading day in the range {start_date} to {end_date}.")
        return date.today()
    return valid_days[-2].date()


@st.cache_data
def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Converts a DataFrame into an in-memory Excel file (bytes).
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# --- Streamlit App UI ---
st.title("🔎 Mistrade Check & CSV Splitter")
st.markdown(
    "This tool fetches data from a report URL, splits it based on suspected mistrades, and provides the results for download.")

# --- Initialize Session State ---
# This ensures the 'processed_data' key exists.
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

# --- 1. Data Source Inputs ---
st.header("1. Provide Data Source")

report_url = st.text_input(
    "Enter the Report URL:",
    help="Paste the full user-facing URL for the report. The app will automatically extract the required ID from the 'bucketPath' parameter."
)

bearer_token = st.text_input(
    "Enter Bearer Token:",
    type="password",
    help="Enter the Bearer Authentication token required to access the data."
)

# --- 2. Process Data ---
st.header("2. Fetch and Process")
if st.button("Fetch and Process Data"):
    if not report_url or not bearer_token:
        st.warning("Please provide both the Report URL and the Bearer Token.")
        st.session_state['processed_data'] = None  # Clear previous results if any
    else:
        try:
            file_identifier = extract_identifier_from_url(report_url)
            if file_identifier:
                st.info(f"Successfully extracted file identifier: **{file_identifier}**")
                with st.spinner("Fetching data from URL..."):
                    download_endpoint = 'https://franz.agent-tool.scalable.capital/agent/s3-access/buckets/prod-franz.sharp-output.scalable/download'
                    file_key = f'quote_analysis/results/v1/mistrade_check/{file_identifier}/{file_identifier}_quote_analysis_mistrade_check_summary_results.csv'
                    headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
                    payload = {'key': file_key}
                    response = requests.post(download_endpoint, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    csv_data_bytes = response.content

                with st.spinner("Processing and splitting data..."):
                    df = pd.read_csv(io.StringIO(csv_data_bytes.decode('utf-8')))
                    if "is_suspected_mistrade" not in df.columns:
                        st.error(
                            'Error: The fetched data does not contain the required column "is_suspected_mistrade".')
                        st.session_state['processed_data'] = None
                    else:
                        st.success("Data fetched and processed successfully!")

                        # Rule 1: The original non-suspected flag is False
                        condition1 = df['is_suspected_mistrade'] == False

                        # Rule 2: Execution price is valid AND the deviation is small
                        condition2 = (df['is_execution_price_valid_gettex'] == True) & (
                                    df['price_pct_deviation_gettex'] <= 0.01)

                        # Rule 3: The normalized PnL is small
                        # Use .abs() for absolute value
                        condition3 = (df['theoretical_pnl_against_mid_gettex'].abs() / df['trade_volume']) <= 0.01

                        # Combine the rules: a trade is "non-suspected" if ANY of the conditions are met (OR logic)
                        is_non_suspected = condition1 | condition2 | condition3

                        # Create the two new DataFrames
                        df_false = df[is_non_suspected]
                        df_true = df[~is_non_suspected]  # The ~ character inverts the condition

                        #old logic
                        # df_true = df[df["is_suspected_mistrade"] == True].copy()
                        # df_false = df[df["is_suspected_mistrade"] == False].copy()

                        # --- Store results in session state ---
                        st.session_state['processed_data'] = {
                            "suspected": df_true,
                            "non_suspected": df_false
                        }

        except requests.exceptions.HTTPError as e:
            st.error(
                f"HTTP Error: Failed to fetch data. Status code: {e.response.status_code}. Please check the URL and Token.")
            st.session_state['processed_data'] = None
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: Could not connect to the URL. Details: {e}")
            st.session_state['processed_data'] = None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state['processed_data'] = None

# --- 3. View Results and Download (Now outside the button click block) ---
if st.session_state['processed_data']:
    st.header("3. View Results and Download")

    last_trading_day = get_previous_trading_day().strftime('%Y-%m-%d')

    df_true = st.session_state['processed_data']['suspected']
    df_false = st.session_state['processed_data']['non_suspected']

    st.subheader(f"Suspected Mistrades ({len(df_true)} rows)")
    st.dataframe(df_true)
    st.download_button(
        label="📥 Download Suspected Mistrades as Excel",
        data=convert_df_to_excel(df_true),
        file_name=f"suspected_mistrades_{last_trading_day}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    st.subheader(f"Non-Suspected Mistrades ({len(df_false)} rows)")
    st.dataframe(df_false)
    st.download_button(
        label="📥 Download Non-Suspected as Excel",
        data=convert_df_to_excel(df_false),
        file_name=f"non_suspected_mistrades_{last_trading_day}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )