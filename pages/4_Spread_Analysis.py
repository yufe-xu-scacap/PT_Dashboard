import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Spread Analysis",
    page_icon="📊",
    layout="wide"
)


# --- Styling Function ---
def highlight_min_max(row):
    """
    Highlights the minimum value (in green) and maximum value (in red)
    in a row, EXCLUDING the '(Gettex)' column from the comparison.
    """

    def get_numeric_value(val):
        if isinstance(val, str):
            val = val.strip().replace('(', '').replace(')', '')
        return pd.to_numeric(val, errors='coerce')

    numeric_series = row.apply(get_numeric_value)
    comparison_series = numeric_series.drop('(Gettex)', errors='ignore')

    min_val = comparison_series.min()
    max_val = comparison_series.max()

    styles = []
    for col_name, val in numeric_series.items():
        style = ''
        if col_name in comparison_series.index and pd.notna(val):
            if val == max_val and val != min_val:
                style = 'color: red; font-weight: bold;'
            elif val == min_val:
                style = 'color: green; font-weight: bold;'
        styles.append(style)

    return styles


# --- MODIFIED: This function now returns the enabled venues ---
def display_venue_options(key_prefix: str):
    """
    Displays the venue checkboxes and returns a list of the enabled venues.
    A key_prefix is required to make widgets unique.
    """
    st.markdown("---")  # Visual separator
    st.markdown("**Compare average of selected venues vs. Gettex:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_tg = st.checkbox("Tradegate", value=True, key=f"{key_prefix}_tradegate")
    with col2:
        enable_ls = st.checkbox("Lang & Schwarz", value=True, key=f"{key_prefix}_ls")
    with col3:
        enable_quotrix = st.checkbox("Quotrix", value=True, key=f"{key_prefix}_quotrix")

    enabled_venues = []
    if enable_tg: enabled_venues.append("Tradegate")
    if enable_ls: enabled_venues.append("Lang & Schwarz")
    if enable_quotrix: enabled_venues.append("Quotrix")

    return enabled_venues


# --- NEW: Function for the spread comparison calculation ---
def calculate_lower_spread_count(df: pd.DataFrame, enabled_venues: list):
    """
    Calculates how many instruments have an average spread lower than Gettex.
    """
    if not enabled_venues or '(Gettex)' not in df.columns:
        return 0

    # Calculate the average of the selected venues for each row
    average_spread = df[enabled_venues].mean(axis=1)

    # Compare the average to Gettex and count how many are lower (sum of True values)
    lower_spread_count = (average_spread < df['(Gettex)']).sum()

    return lower_spread_count


# --- MODIFIED: Function now also returns the full processed dataframes ---
def process_spread_data(df: pd.DataFrame):
    """
    Cleans and analyzes the spread data from the uploaded Excel file.
    """
    try:
        df_processed = df.copy()
        df_processed = df_processed.replace('_', '', regex=True)
        df_processed = df_processed.rename(columns={
            'Formula Col. 6': 'ISIN', 'Formula Col. 7': 'Tradegate',
            'Formula Col. 8': 'Lang & Schwarz', 'Formula Col. 10': 'Quotrix',
            'Formelspalte 2': '(Gettex)'
        })
        df_processed = df_processed.iloc[1:]
        df_processed = df_processed.replace(['#ERROR', '', ' '], np.nan).dropna()

        cols_to_convert = ['Tradegate', 'Lang & Schwarz', 'Quotrix', '(Gettex)']
        for col in cols_to_convert:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        df_us = df_processed[df_processed['ISIN'].str.startswith('US')].copy()
        df_de = df_processed[df_processed['ISIN'].str.startswith('DE')].copy()
        us_instrument_count = len(df_us)
        de_instrument_count = len(df_de)

        cols_to_analyze = ['Tradegate', 'Lang & Schwarz', 'Quotrix', '(Gettex)']
        cols_for_comparison = ['Tradegate', 'Lang & Schwarz', 'Quotrix']

        summary_stats_us = df_us[cols_to_analyze].agg(['mean', 'median'])
        worst_counts_us = df_us[cols_for_comparison].idxmax(axis=1).value_counts().to_frame().T
        worst_counts_us.index = ['Worst Spread Count']
        final_us = pd.concat([summary_stats_us, worst_counts_us])
        final_us.loc[['mean', 'median']] = final_us.loc[['mean', 'median']].round(3)
        if '(Gettex)' in final_us.columns:
            final_us['(Gettex)'] = final_us['(Gettex)'].apply(lambda x: f'({x})' if pd.notna(x) else x)
        final_us.loc['Worst Spread Count'] = final_us.loc['Worst Spread Count'].fillna(0)

        summary_stats_de = df_de[cols_to_analyze].agg(['mean', 'median'])
        worst_counts_de = df_de[cols_for_comparison].idxmax(axis=1).value_counts().to_frame().T
        worst_counts_de.index = ['Worst Spread Count']
        final_de = pd.concat([summary_stats_de, worst_counts_de])
        final_de.loc[['mean', 'median']] = final_de.loc[['mean', 'median']].round(3)
        if '(Gettex)' in final_de.columns:
            final_de['(Gettex)'] = final_de['(Gettex)'].apply(lambda x: f'({x})' if pd.notna(x) else x)
        final_de.loc['Worst Spread Count'] = final_de.loc['Worst Spread Count'].fillna(0)

        return final_us, final_de, us_instrument_count, de_instrument_count, df_us, df_de

    except KeyError as e:
        st.error(
            f"Processing failed. A required column is missing: {e}. Please ensure the file has the correct format.")
        return None, None, 0, 0, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        return None, None, 0, 0, None, None


# --- Streamlit UI ---
st.title("📊 Spread Analysis")
st.markdown("Upload an Excel (`.xlsx`) file from your local machine to begin the analysis.")

st.header("1. Upload Your Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type='xlsx', help="Upload the Excel file with spread data.")

if uploaded_file is not None:
    st.header("2. Analysis Results")
    raw_df = pd.read_excel(uploaded_file, engine='openpyxl')

    with st.spinner("Analyzing data..."):
        # --- MODIFIED: Unpack the new full dataframes ---
        us_results, de_results, us_count, de_count, df_us, df_de = process_spread_data(raw_df)

    if us_results is not None and de_results is not None:
        st.subheader(f"US Stocks Summary (Total Instruments: {us_count})")
        if not us_results.empty:
            st.dataframe(
                us_results.style.apply(highlight_min_max, axis=1)
                .format('{:.0f}', subset=pd.IndexSlice['Worst Spread Count', :], na_rep='0')
            )
            # --- MODIFIED: Get enabled venues and calculate/display the new metric ---
            enabled_us_venues = display_venue_options(key_prefix="us")
            lower_count_us = calculate_lower_spread_count(df_us, enabled_us_venues)
            st.metric(label="Instruments with Lower Spread than Gettex", value=lower_count_us)
        else:
            st.warning("No US stocks (starting with ISIN 'US') found in the uploaded file.")

        st.divider()

        st.subheader(f"DE Stocks Summary (Total Instruments: {de_count})")
        if not de_results.empty:
            st.dataframe(
                de_results.style.apply(highlight_min_max, axis=1)
                .format('{:.0f}', subset=pd.IndexSlice['Worst Spread Count', :], na_rep='0')
            )
            # --- MODIFIED: Get enabled venues and calculate/display the new metric ---
            enabled_de_venues = display_venue_options(key_prefix="de")
            lower_count_de = calculate_lower_spread_count(df_de, enabled_de_venues)
            st.metric(label="Instruments with Lower Spread than Gettex", value=lower_count_de)
        else:
            st.warning("No German stocks (starting with ISIN 'DE') found in the uploaded file.")