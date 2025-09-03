import streamlit as st
import pandas as pd
import numpy as np
import os

st.html("""
  <style>
    [alt=Logo] {
      height: 4.5rem;
    }
  </style>
        """)
st.logo("data/logo.png", size="large")

# --- Page Configuration ---
st.set_page_config(
    page_title="PT Dashboard",
    page_icon="data/icon.png",
    layout="wide"
)


# --- Function to load index data (No Changes) ---
def load_index_data():
    file_path = os.path.join('data', 'Index.csv')
    try:
        index_df = pd.read_csv(file_path, header=0)
        index_df.rename(columns={'isin': 'ISIN', 'index': 'Index'}, inplace=True)
        if 'ISIN' in index_df.columns and 'Index' in index_df.columns:
            return index_df[['ISIN', 'Index']]
        else:
            st.error("The 'Index.csv' file must contain 'ISIN' and 'Index' columns.")
            return None
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading 'Index.csv': {e}")
        return None


# --- Styling Function (No Changes) ---
def highlight_min_max(row):
    def get_numeric_value(val):
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


# --- Venue Options Function (No Changes) ---
def display_venue_options(key_prefix: str):
    st.markdown("---")
    st.markdown("**Compare average of selected Exchanges vs. Gettex:**")
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


# --- Calculation Function (No Changes) ---
def calculate_lower_spread_count(df: pd.DataFrame, enabled_venues: list):
    if not enabled_venues or '(Gettex)' not in df.columns or df.empty:
        return 0
    average_spread = df[enabled_venues].mean(axis=1)
    lower_spread_count = (average_spread < df['(Gettex)']).sum()
    return lower_spread_count


# --- Data Processing Function (No Changes) ---
def process_spread_data(df: pd.DataFrame, index_df: pd.DataFrame):
    try:
        df_processed = df.copy()
        df_processed = df_processed.replace('_', '', regex=True)
        df_processed = df_processed.rename(columns={
            'Formula Col. 6': 'ISIN', 'Formula Col. 7': 'Tradegate',
            'Formula Col. 8': 'Lang & Schwarz', 'Formula Col. 10': 'Quotrix',
            'Formelspalte 2': '(Gettex)'
        })
        df_processed = df_processed.iloc[1:]
        df_processed = df_processed.replace(['#ERROR', '', ' '], np.nan)
        if index_df is not None:
            df_processed = pd.merge(df_processed, index_df, on='ISIN', how='left')
            df_processed['Index'].fillna('Other', inplace=True)
        else:
            df_processed['Index'] = 'Other'
        cols_to_check_na = ['ISIN', 'Tradegate', 'Lang & Schwarz', 'Quotrix', '(Gettex)']
        df_processed.dropna(subset=cols_to_check_na, inplace=True)
        cols_to_convert = ['Tradegate', 'Lang & Schwarz', 'Quotrix', '(Gettex)']
        for col in cols_to_convert:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # Drop rows where ISIN is not a string or is shorter than 2 characters
        df_processed = df_processed[df_processed['ISIN'].apply(lambda x: isinstance(x, str) and len(x) >= 2)]
        return df_processed
    except KeyError as e:
        st.error(f"Processing failed. A required column is missing from the uploaded file: {e}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        return None


# --- Summary Generation Function (No Changes) ---
def generate_summary(df_subgroup: pd.DataFrame):
    if df_subgroup.empty:
        return pd.DataFrame()
    cols_to_analyze = ['Tradegate', 'Lang & Schwarz', 'Quotrix', '(Gettex)']
    cols_for_comparison = ['Tradegate', 'Lang & Schwarz', 'Quotrix']
    cols_to_analyze = [col for col in cols_to_analyze if col in df_subgroup.columns]
    cols_for_comparison = [col for col in cols_for_comparison if col in df_subgroup.columns]

    summary_stats = df_subgroup[cols_to_analyze].agg(['mean', 'median'])
    worst_counts = df_subgroup[cols_for_comparison].idxmax(axis=1).value_counts().to_frame().T
    worst_counts.index = ['Worst Spread Count']

    final_summary = pd.concat([summary_stats, worst_counts])
    final_summary.loc[['mean', 'median']] = final_summary.loc[['mean', 'median']].round(3)
    final_summary.loc['Worst Spread Count'] = final_summary.loc['Worst Spread Count'].fillna(0)

    return final_summary


# --- Analysis display function (No Changes) ---
def display_country_analysis(country_code, country_df, formatters):
    """
    Displays the full analysis block for a given country's data.
    """
    st.markdown(f"#### {country_code} Stocks (Total: {len(country_df)})")

    if not country_df.empty:
        summary_df = generate_summary(country_df)
        st.dataframe(
            summary_df.style.apply(highlight_min_max, axis=1)
            .format(formatters, subset=pd.IndexSlice[['mean', 'median'], :], na_rep='-')
            .format('{:.0f}', subset=pd.IndexSlice['Worst Spread Count', :], na_rep='0')
            .set_properties(subset=['(Gettex)'], **{'color': 'grey'})
        )
        enabled_venues = display_venue_options(key_prefix=f"{country_code}_country")
        lower_count = calculate_lower_spread_count(country_df, enabled_venues)
        st.metric(label="Instruments with Lower Spread than Gettex", value=lower_count)
    else:
        st.warning(f"No data to display for {country_code} stocks.")


# --- Streamlit UI ---
st.title("📊 Spread Analysis")
st.markdown(
    "Upload an Excel (`.xlsx`) file to analyze spreads. The analysis is available by **Country** and by **Index** (from `data/Index.csv`).")

index_data = load_index_data()

st.header("1. Upload Your Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type='xlsx', help="Upload the Excel file with spread data.")

if uploaded_file is not None and index_data is not None:
    st.header("2. Analysis Results (Time Weighted Average Spread Pct)")
    raw_df = pd.read_excel(uploaded_file, engine='openpyxl')

    with st.spinner("Analyzing data..."):
        processed_df = process_spread_data(raw_df, index_data)

    if processed_df is not None and not processed_df.empty:

        mean_median_formatters = {
            'Tradegate': '{:.3f}',
            'Lang & Schwarz': '{:.3f}',
            'Quotrix': '{:.3f}',
            '(Gettex)': '({:.3f})'
        }

        tab_country, tab_index = st.tabs(["Analysis by Country Code", "Analysis by Index"])

        # --- MODIFIED: Country Analysis Tab with new sorting and layout ---
        with tab_country:
            st.subheader("Country Code-Based Summary")
            st.markdown("---")

            # 1. Get instrument counts for each country and sort descending
            country_counts = processed_df['ISIN'].str[:2].value_counts()
            all_codes_sorted_by_count = country_counts.index.tolist()

            # 2. Separate into priority ('DE', 'US') and other lists
            priority_order = ['DE', 'US']
            priority_codes_present = [code for code in priority_order if code in all_codes_sorted_by_count]
            other_codes_sorted = [code for code in all_codes_sorted_by_count if code not in priority_codes_present]

            # 3. Display the priority codes ('DE', 'US') in a 2-column top row
            if priority_codes_present:
                cols = st.columns(2)
                for i, code in enumerate(priority_codes_present):
                    with cols[i]:
                        country_df = processed_df[processed_df['ISIN'].str.startswith(code)].copy()
                        display_country_analysis(code, country_df, mean_median_formatters)

            # 4. Display all other codes in a 3-column grid below
            if other_codes_sorted:
                st.markdown("---")  # Visual separator

                # Create a 3-column layout
                col1, col2, col3 = st.columns(3)
                cols_cycle = [col1, col2, col3]

                for i, code in enumerate(other_codes_sorted):
                    # Determine which column to place the content in
                    current_col = cols_cycle[i % 3]
                    with current_col:
                        country_df = processed_df[processed_df['ISIN'].str.startswith(code)].copy()
                        display_country_analysis(code, country_df, mean_median_formatters)
                        st.markdown("---")  # Separator for visual clarity within columns

        # --- Index Analysis Tab (No Changes) ---
        with tab_index:
            st.subheader("Index-Based Summary")
            layout = [
                ['DE_large', 'DE_mid', 'DE_small'],
                ['EU_large', 'EU_total'],
                ['US_tech', 'US_large']
            ]
            available_indices = set(processed_df['Index'].unique())

            for row_group in layout:
                st.markdown("---")
                cols = st.columns(3)

                for i, index_name in enumerate(row_group):
                    if index_name in available_indices:
                        with cols[i]:
                            df_subgroup = processed_df[processed_df['Index'] == index_name].copy()
                            st.markdown(f"#### {index_name} (Total: {len(df_subgroup)})")
                            summary_table = generate_summary(df_subgroup)
                            st.dataframe(
                                summary_table.style.apply(highlight_min_max, axis=1)
                                .format(mean_median_formatters, subset=pd.IndexSlice[['mean', 'median'], :], na_rep='-')
                                .format('{:.0f}', subset=pd.IndexSlice['Worst Spread Count', :], na_rep='0')
                                .set_properties(subset=['(Gettex)'], **{'color': 'grey'})
                            )
                            key_prefix = index_name.replace(" ", "_").replace("-", "_")
                            enabled_venues = display_venue_options(key_prefix=key_prefix)
                            if enabled_venues:
                                lower_count = calculate_lower_spread_count(df_subgroup, enabled_venues)
                                st.metric(label="Instruments with Lower Spread than Gettex", value=f"{lower_count}")
                            else:
                                st.info("Select at least one venue to compare.")

    elif uploaded_file is not None:
        st.warning("Could not process the uploaded file. Please check the format and content.")