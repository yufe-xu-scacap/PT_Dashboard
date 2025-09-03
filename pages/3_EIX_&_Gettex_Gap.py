import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Dict
import io
import re
from bs4 import BeautifulSoup

st.html("""
  <style>
    [alt=Logo] {
      height: 4.5rem;
    }
  </style>
        """)
st.logo("data/logo.png", size="large")

st.set_page_config(
    page_title="PT Dashboard",
    page_icon="data/icon.png",
    layout="wide"
)
# --- Helper Function for Downloading ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    """Converts a DataFrame to a CSV bytes object for downloading."""
    return df.to_csv(index=False, sep=';').encode('utf-8')


# --- Core Logic Functions (Slightly Modified for Streamlit) ---

def download_file_content(file_key: str, bearer_token: str, download_url: str) -> Optional[bytes]:
    """Downloads a single file's content into memory."""
    headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
    payload = {'key': file_key}
    try:
        response = requests.post(download_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        st.info(f"Successfully downloaded content of '{file_key}'")
        return response.content
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error(f"Failed to download '{file_key}'. Error: 401 Unauthorized. Check Bearer Token.")
        else:
            st.error(f"Failed to download '{file_key}'. HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download '{file_key}'. Request Error: {e}")
    return None


def process_and_combine_content(file_contents: Dict[str, bytes]) -> Optional[pd.DataFrame]:
    """Reads CSV content from memory for EIX files, combines them, and returns a single DataFrame."""
    all_dataframes = []
    columns_to_keep = {3: 'ISIN', 7: 'Name'}
    for name, content in file_contents.items():
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None, sep=';', usecols=columns_to_keep.keys(),
                             names=columns_to_keep.values(), on_bad_lines='skip', encoding='utf-8')
            all_dataframes.append(df)
        except Exception as e:
            st.error(f"An unexpected error occurred while processing '{name}': {e}")
    if not all_dataframes:
        return None
    return pd.concat(all_dataframes, ignore_index=True)


def process_final_file(file_content: bytes) -> Optional[pd.DataFrame]:
    """Processes the Gettex file from memory."""
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep=',', header=None, on_bad_lines='skip',
                         encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"An unexpected error occurred while processing content: {e}")
        return None


# --- ALIGNED XETRA SCRAPER ---
def run_xetra_scraper() -> Optional[pd.DataFrame]:
    """
    Scrapes the Xetra website using the logic from track_EIX_Gettex_gap(v2).py.
    """
    st.info("--- Starting Xetra New Listings Scraper ---")
    url = "https://www.xetra.com/xetra-en/instruments/etfs-etps/statistics/new-etfs-etps"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        all_text = soup.get_text(separator=' ', strip=True)
        cleaned_text = re.sub(r'\s+', ' ', all_text)

        start_marker = "Asset Class Date Name Trading Currency ISIN Xetra Symbol"
        end_marker = "Additional Information"
        start_pos = cleaned_text.find(start_marker)
        end_pos = cleaned_text.rfind(end_marker)

        if start_pos == -1 or end_pos == -1 or end_pos <= start_pos:
            st.error("Could not find the start/end markers on the Xetra page. The text content may have changed.")
            return None

        data_block = cleaned_text[start_pos:end_pos].strip()
        content = data_block.replace(start_marker, "").strip()

        pattern = re.compile(
            r"(?P<date>\d{2}\.\d{2}\.\d{4})\s+"
            r"(?P<name>.*?)\s+"
            r"(?P<currency>EUR|USD)\s+"
            r"(?P<isin>[A-Z0-9]{12})\s+"
            r"(?P<symbol>[A-Z0-9\.]+)"
        )

        matches = list(pattern.finditer(content))
        if not matches:
            st.warning("Could not find any data records matching the pattern on the Xetra page.")
            return pd.DataFrame()  # Return empty dataframe

        data_rows = []
        for i, match in enumerate(matches):
            record = match.groupdict()
            # Determine asset class based on text between matches
            asset_class_text = content[:match.start()].strip() if i == 0 else content[matches[
                                                                                          i - 1].end():match.start()].strip()
            record['Asset Class'] = asset_class_text
            data_rows.append({
                'Asset Class': record['Asset Class'], 'Date': record['date'],
                'Name': record['name'], 'Trading Currency': record['currency'],
                'ISIN': record['isin'], 'Xetra Symbol': record['symbol']
            })

        df = pd.DataFrame(data_rows)
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

        today_date = date.today()
        start_of_last_week = today_date - timedelta(days=today_date.weekday(), weeks=1)
        end_of_current_week = today_date + timedelta(days=(6 - today_date.weekday()))

        st.info(
            f"Filtering Xetra data between {start_of_last_week.strftime('%Y-%m-%d')} and {end_of_current_week.strftime('%Y-%m-%d')}...")

        mask = (df['Date'].dt.date >= start_of_last_week) & (df['Date'].dt.date <= end_of_current_week)
        filtered_df = df[mask].copy()

        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%d.%m.%Y')

        return filtered_df

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the Xetra request: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing the Xetra page: {e}")
    return None


# --- Streamlit Page UI ---

st.title("⚙️ EIX & Gettex Data Downloader and Processor")
st.markdown("This page downloads, processes, and merges master data from EIX, MWB, and Gettex.")

st.header("1. Gettex and EIX/MWB Data Processing")

help_text = """
**EIX & MWB URL:**
https://franz.agent-tool.scalable.capital/misc/s3-access/regds_data-external-reporting_scalable-prod?bucketPath=vwdts%2FSCALABLE%2FEIX%2Fmasterdata%2Fin%2F

**Gettex URL:**
https://franz.agent-tool.scalable.capital/misc/s3-access/financial-instrument-data_prod-b2c?bucketPath=exchangeDataArchive
"""

bearer_token = st.text_input(
    "Enter Bearer Token",
    type="password",
    help=help_text
)

if st.button("Start Download and Process"):
    if not bearer_token:
        st.warning("Please enter the Bearer Token.")
    else:
        with st.spinner("Processing... Please wait."):
            today_utc = datetime.now(timezone.utc)
            today_str = today_utc.strftime("%Y%m%d")
            today_gettex_str = today_utc.strftime("%Y-%m-%d")

            files_to_process_eix = {
                'MDIF_SCALABLE_EIX': f'vwdts/SCALABLE/EIX/masterdata/in/MDIF_SCALABLE_EIX_{today_str}.csv',
                'MDIF_MWB_EIX': f'vwdts/SCALABLE/EIX/masterdata/in/MDIF_MWB_EIX_{today_str}.csv'
            }
            gettex_file_key = f'exchangeDataArchive/{today_gettex_str}/gettex_data.csv'

            eix_contents = {}
            for name, key in files_to_process_eix.items():
                content = download_file_content(key, bearer_token,
                                                'https://franz.agent-tool.scalable.capital/agent/s3-access/buckets/regds.data-external-reporting.scalable-prod/download')
                if content:
                    eix_contents[name] = content

            df_EIX = process_and_combine_content(eix_contents) if eix_contents else pd.DataFrame()

            gettex_content = download_file_content(gettex_file_key, bearer_token,
                                                   'https://franz.agent-tool.scalable.capital/agent/s3-access/buckets/financial-instrument-data.prod-b2c/download')

            if gettex_content is not None:
                df_final_gettex = process_final_file(gettex_content)
                if df_final_gettex is not None:
                    df_final_gettex.columns = df_final_gettex.iloc[0]
                    df_final = df_final_gettex[1:].copy()
                    df_final = df_final[df_final['instrument_type'].isin(['SHR', 'FDS', 'ETF', 'ETC', 'ETN'])]
                    df_final = df_final[['isin', 'shortname', 'marktsegment', 'instrument_type']]
                    merged_df = pd.merge(df_final, df_EIX, left_on='isin', right_on='ISIN', how='left', indicator=True)
                    df_result = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge', 'ISIN', 'Name'])
                    df_result = df_result[['isin', 'shortname', 'marktsegment', 'instrument_type']]

                    st.success("Processing complete!")
                    # Store the result in the session state
                    st.session_state['result_df'] = df_result
            else:
                st.error("Could not retrieve Gettex data. Aborting process.")

# --- START: Modified Section ---
# This block now handles displaying the results.
# It will run every time the script reruns (e.g., after clicking a button)
# as long as the result DataFrame exists in the session state.
if 'result_df' in st.session_state:
    st.dataframe(st.session_state['result_df'])

    st.download_button(
        label="📥 Download Result as CSV",
        data=convert_df_to_csv(st.session_state['result_df']),
        file_name=f"result_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
# --- END: Modified Section ---

st.divider()

st.header("2. Xetra New Listings Scraper")
st.markdown("Scrapes the Xetra website for newly listed ETFs & ETPs from the current and previous week.")

if st.button("Run Xetra Scraper"):
    with st.spinner("Scraping Xetra website..."):
        xetra_df = run_xetra_scraper()
        if xetra_df is not None and not xetra_df.empty:
            st.success("Found new listings!")
            st.dataframe(xetra_df)
        elif xetra_df is not None:
            st.info("No new Xetra listings found for the current or previous calendar week.")