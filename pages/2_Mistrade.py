import streamlit as st
import pandas as pd
import requests
import io
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qs
import pandas_market_calendars as mcal
import os

# --- Page Configuration ---
st.html("""
  <style>
    [alt=Logo] {
      height: 4.5rem;
    }
  </style>
        """)
st.logo("data/logo.png", size="large")

st.set_page_config(
    page_title="Mistrade Check & CSV Splitter",
    page_icon="🔎",
    layout="wide"
)

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
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

# --- 1. Fetch Data ---
st.header("Data Source")
# CSS to vertically align the button with the text inputs
st.markdown("""
<style>
    /* This targets the container of the button for better alignment */
    div[data-testid="column"] {
        display: flex;
        align-items: flex-end;
    }
</style>
""", unsafe_allow_html=True)

# Create columns for the inputs and the button
col1, col2, col3 = st.columns([4, 4, 2]) # Adjust ratios as needed

with col1:
    report_url = st.text_input(
        "Enter the Report URL:",
        help="Paste the full user-facing URL for the report. The app will automatically extract the required ID."
    )

with col2:
    bearer_token = st.text_input(
        "Enter Bearer Token:",
        type="password",
        help="Enter the Bearer Authentication token required to access the data."
    )

with col3:
    process_button = st.button("Fetch and Process Data")


# --- 2. Process Data (triggered by button) ---
if process_button:
    if not report_url or not bearer_token:
        st.warning("Please provide both the Report URL and the Bearer Token.")
        st.session_state['processed_data'] = None  # Clear previous results if any
    else:
        st.header("Processing Results")
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
                        condition3 = (df['theoretical_pnl_against_mid_gettex'].abs() / df['trade_volume']) <= 0.01

                        # A trade is "non-suspected" if ANY of the conditions are met (OR logic)
                        is_non_suspected = condition1 | condition2 | condition3

                        # Create the two new DataFrames
                        df_false = df[is_non_suspected]
                        df_true = df[~is_non_suspected]

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

# The URLs are long, so let's define them first for readability
# excel_url = "https://docs.google.com/spreadsheets/d/1OBM1X-x1PS1ZB1j1p26bZ-ZffSDttxl1/edit?usp=drive_link&ouid=101390197422307543832&rtpof=true&sd=true"
excel_url = "https://drive.google.com/uc?export=view&id=1OBM1X-x1PS1ZB1j1p26bZ-ZffSDttxl1"
gws_url = "https://drive.google.com/drive/folders/1ST0Thozebs-ytXW9f4I_UM2APMnJYBr3" # Same URL in your example
# Create two equal-width columns
col1, col2 = st.columns(2)

# Place the first info box in the first column
with col1:
    st.info(
        f"""
        [**Click here for the Excel template**]({excel_url})
        """,
        icon="📄"
    )
# Place the second info box in the second column
with col2:
    st.info(
        f"""
        [**Click here for Google Workspace Order**]({gws_url})
        """,
        icon="📁"
    )

with st.expander("Show/Hide VBA Code"):
    vba_code = """Sub ProcessDataAtIntervals()
    ' --- OPTIMIZATION & SETUP ---
    ' Turn off Excel features to speed up the macro
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual

    ' Set the worksheet you are working on.
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("mistradecheck")

    ' An array to hold all the time intervals you want to process.
    Dim timeIntervals As Variant
    timeIntervals = Array("00:02:00", "00:04:00", "00:06:00", "00:08:00", "00:10:00", "00:15:00", "00:20:00", "00:25:00", "00:30:00", "00:45:00", "01:25:00")

    ' Declare other necessary variables
    Dim singleTime As Variant
    Dim lastRow As Long
    Dim i As Long
    Dim waitUntil As Date

    ' --- MAIN LOOP (OUTER LOOP) ---
    For Each singleTime In timeIntervals
        
        ' 1. UPDATE THE TIME AND PROVIDE STATUS FEEDBACK
        Application.StatusBar = "Processing time: " & singleTime & ". Waiting for Bloomberg..."
        ws.Range("C2").Value = CDate(singleTime)

        ' 2. TRIGGER RECALCULATION FOR BLOOMBERG
        ws.Calculate
        
        ' 3. NEW RESPONSIVE WAIT METHOD
        ' This replaces Application.Wait to keep Excel from freezing.
        ' If data loads slowly, increase the "15" (seconds).
        waitUntil = Now + TimeValue("00:00:15")
        Do While Now < waitUntil
            DoEvents ' This command processes other events, keeping Excel responsive.
        Loop

        ' 4. NEW RULE: DETERMINE SCANNING LIMIT FROM COLUMN B
        Application.StatusBar = "Finding scan limit from Column B for time: " & singleTime & "..."
        ' Find the absolute last row with data in column B first
        lastRow = ws.Cells(ws.Rows.Count, "B").End(xlUp).Row
        
        ' Loop backwards from the bottom to find the first row that is NOT "NO FURTHER VERIFICATION"
        For i = lastRow To 2 Step -1
            If ws.Cells(i, "B").Value <> "NO FURTHER VERIFICATION" Then
                lastRow = i ' This is the new, shorter last row to check
                Exit For ' Stop searching once we've found it
            End If
            
            ' If the loop reaches the top and the first data row is also "NO FURTHER VERIFICATION"
            If i = 2 And ws.Cells(i, "B").Value = "NO FURTHER VERIFICATION" Then
                lastRow = 1 ' Set lastRow to 1 so the next loop doesn't run
            End If
        Next i
        
        ' 5. SCAN FOR "NO MISTRADE" WITHIN THE DETERMINED LIMIT
        Application.StatusBar = "Scanning up to row " & lastRow & " for NO MISTRADE..."
        
        ' Only run this loop if there are rows to check (lastRow > 1)
        If lastRow > 1 Then
            For i = 2 To lastRow
                ' DEBUG FIX: Use CStr to convert the cell value to a string first.
                ' This prevents a "Type Mismatch" error if the cell contains an error value like #VALUE!
                If CStr(ws.Cells(i, "AB").Value) = "NO MISTRADE" Then
                    ws.Rows(i).Value = ws.Rows(i).Value
                End If
                ' This DoEvents helps keep the inner loop from freezing on very large datasets
                If i Mod 100 = 0 Then DoEvents
            Next i
        End If
    Next singleTime

    ' --- CLEANUP ---
    ' Restore Excel's original settings
    Application.StatusBar = False ' Clears the status bar
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Application.Calculation = xlCalculationAutomatic

    MsgBox "Process Complete! All specified time intervals have been processed."

End Sub
"""
    st.text_area(
        label="Copy your VBA code from here:",
        value=vba_code,
        height=400,
        help="You can edit this code directly in the box and copy it for use in Excel."
    )

# --- 3. View Results and Download ---
if st.session_state['processed_data']:
    st.header("View Results and Download")

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