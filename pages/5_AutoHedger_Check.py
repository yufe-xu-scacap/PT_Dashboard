import streamlit as st
import pandas as pd
import numpy as np
from datetime import time, datetime
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('🛡️ Auto-Hedger Analysis')

# --- State Initialization ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

st.subheader("Define Auto-Hedger Enabled & Disabled Time Window")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your trade data (CSV)", type="csv")
with col2:
    start_time_str = st.text_input("Start Time (HH:MM:SS)", "09:49:00")
with col3:
    end_time_str = st.text_input("End Time (HH:MM:SS)", "11:44:00")
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button('Run Analysis')

st.subheader("Instruments")
instrument_input = st.text_area("Paste your list of instruments (space-separated)",
                                height=100)

if run_button:
    st.session_state.analysis_run = True
    if 'instrument_to_show' in st.session_state:
        del st.session_state.instrument_to_show

# --- Main Analysis and Display Block ---
if st.session_state.analysis_run:
    try:
        start_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
        end_time = datetime.strptime(end_time_str, "%H:%M:%S").time()

        if uploaded_file is not None and instrument_input:
            with st.spinner('Running analysis... This may take a moment.'):
                df_data = pd.read_csv(uploaded_file)
                autohedger_instruments = instrument_input.split()

                st.success(f"Processing {len(autohedger_instruments)} ISINs between {start_time} and {end_time}.")

                df_data["Execution Time"] = pd.to_datetime(df_data["Execution Time"]) + pd.Timedelta(hours=2)

                df_data = df_data[
                    (df_data['Execution Time'].dt.time >= start_time) &
                    (df_data['Execution Time'].dt.time <= end_time)
                    ]

                df_data['TradeType'] = np.select(
                    [
                        (df_data['Cpty'] == 'LOMSCALOFPS'),
                        (df_data['Cpty'] != 'LOMSCALOFPS') &
                        (~df_data['ExecutionWithinFirm'].str.startswith('SCLB', na=False))
                    ],
                    [
                        'ClientTrade',
                        'HedgeTrade'
                    ],
                    default='ManualTrade'
                )

                df_data = df_data[df_data["Instrument"].isin(autohedger_instruments)]
                df_data = df_data[df_data['Qty'] % 1 == 0]
                df_data = df_data.sort_values(by='Execution Time', ascending=True).reset_index(drop=True)


                # --- MODIFIED FUNCTION ---
                def filter_trade_block(group):
                    """
                    This function filters a group of trades to include everything from the first client trade
                    to the last hedge/manual trade.
                    """
                    try:
                        # Find the index of the first 'ClientTrade'
                        first_client_idx = (group['TradeType'] == 'ClientTrade').idxmax()

                        # Find the index of the last 'HedgeTrade' or 'ManualTrade'
                        is_hedge = (group['TradeType'] == 'HedgeTrade')

                        if not is_hedge.any():
                            # If no hedges, you might want to return nothing after the client trade,
                            # or handle it differently. Here, we return just the client trade.
                            return group.loc[first_client_idx:first_client_idx]

                        # The corrected logic to find the last index by reversing the boolean Series
                        last_hedge_idx = is_hedge[::-1].idxmax()

                        # Return the slice from the first client trade to the last hedge
                        return group.loc[first_client_idx:last_hedge_idx]
                    except ValueError:
                        # This triggers if no 'ClientTrade' is found in the group
                        return None


                df_data = df_data.groupby('Instrument', group_keys=False).apply(filter_trade_block).reset_index(
                    drop=True)


                def calculate_hedge(group, qty_col='Qty', price_col='Price', time_col='Execution Time',
                                    trade_type_col='TradeType'):
                    """
                    Calculates hedging metrics with a revised logic for Weighted Average Price (WAP).
                    WAP is now only updated for trades that increase the position size.
                    """
                    # --- Step 1 & 2: Initialization (Same as before) ---
                    initial_position = -group[qty_col].sum()
                    initial_price = group.iloc[0][price_col] if initial_position != 0 and not group.empty else np.nan
                    current_total_qty = initial_position
                    current_wap = initial_price
                    results = []
                    last_client_trade_time = group.loc[group[trade_type_col] == 'ClientTrade', time_col].max()
                    modified_trade_types = []

                    # --- Step 3: Loop Through Each Trade to Calculate Metrics ---
                    for index, row in group.iterrows():
                        trade_qty = row[qty_col]
                        trade_price = row[price_col]

                        # (Trade type re-classification logic remains the same)
                        original_trade_type = row[trade_type_col]
                        is_non_hedging_hedge = (
                                original_trade_type == 'HedgeTrade' and
                                (current_total_qty == 0 or np.sign(trade_qty) == np.sign(current_total_qty))
                        )
                        current_trade_type = 'ClientTrade' if is_non_hedging_hedge else original_trade_type
                        modified_trade_types.append(current_trade_type)

                        # Define the state *before* this trade.
                        pos_before_trade = current_total_qty
                        wap_before_trade = current_wap

                        # --- P&L Calculation (Same as before) ---
                        metrics = {'pnl': np.nan, 'pnl_fifo': np.nan, 'time_lag': pd.NaT, 'relative_return_bps': np.nan}
                        is_real_hedge = (
                                current_trade_type in ['HedgeTrade', 'ManualTrade'] and
                                pos_before_trade != 0 and
                                np.sign(trade_qty) != np.sign(pos_before_trade)
                        )
                        if is_real_hedge and not pd.isna(wap_before_trade):
                            pnl_per_share = (trade_price - wap_before_trade) * np.sign(pos_before_trade)
                            metrics['pnl'] = round(pnl_per_share * abs(trade_qty), 2)
                            if abs(wap_before_trade) > 1e-9:
                                metrics['relative_return_bps'] = (pnl_per_share / wap_before_trade) * 10000
                            if pd.notna(last_client_trade_time):
                                metrics['time_lag'] = row[time_col] - last_client_trade_time

                        if current_trade_type == 'ClientTrade':
                            last_client_trade_time = row[time_col]

                        # --- NEW: WAP and Position Update Logic ---
                        # The WAP is now only updated when a trade increases the position.
                        pos_after_trade = pos_before_trade + trade_qty

                        # Determine if the trade opens or increases the absolute position size.
                        is_increasing_trade = (pos_before_trade == 0 or np.sign(trade_qty) == np.sign(pos_before_trade))

                        # Determine if the position flipped signs (e.g., long to short).
                        is_crossing_zero = (
                                pos_before_trade != 0 and np.sign(pos_before_trade) != np.sign(pos_after_trade))

                        if pos_after_trade == 0:
                            # 1. FLATTENED: Position is now zero, so there's no WAP.
                            wap_after_trade = np.nan
                        elif is_crossing_zero:
                            # 2. FLIPPED: Position crossed zero. The new WAP is the price of the current trade,
                            # as it establishes the cost basis for the new position.
                            wap_after_trade = trade_price
                        elif is_increasing_trade:
                            # 3. INCREASED: Trade adds to the position. Calculate the new WAP.
                            if pos_before_trade == 0 or pd.isna(wap_before_trade):
                                # This is the first trade of a new position.
                                wap_after_trade = trade_price
                            else:
                                # Update the WAP with the new trade.
                                wap_after_trade = ((wap_before_trade * pos_before_trade) +
                                                   (trade_price * trade_qty)) / pos_after_trade
                        else:
                            # 4. REDUCED: Trade reduces the position (a "real hedge").
                            # The WAP of the remaining shares does NOT change.
                            wap_after_trade = wap_before_trade

                        # --- Finalize ---
                        current_total_qty = pos_after_trade
                        current_wap = wap_after_trade

                        metrics['Open Position'] = pos_after_trade
                        results.append(metrics)

                    # --- Step 4: Combine Results (Same as before) ---
                    new_cols = pd.DataFrame(results, index=group.index)
                    group[['pnl', 'pnl_fifo', 'time_lag', 'return(bps)', 'Open Position']] = new_cols

                    return group


                df_final = df_data.groupby('Instrument').apply(calculate_hedge)


                def format_timedelta_to_hms(td):
                    if pd.isna(td): return None
                    total_seconds = int(td.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f'{hours:02}:{minutes:02}:{seconds:02}'


                df_final['time_lag'] = df_final['time_lag'].apply(format_timedelta_to_hms)

                columns_to_keep = ['Instrument', 'Free Text', 'B/S', 'Qty', 'Open Position', 'Price', 'Premium',
                                   'Execution Time', 'TradeType', 'time_lag', 'pnl', 'return(bps)',
                                   'ExecutionWithinFirm', 'Cpty']

                df_final = df_final[[col for col in columns_to_keep if col in df_final.columns]]
                df_final.reset_index(drop=True, inplace=True)

                tab1, tab2 = st.tabs(["📊 Summary", "📄 Details"])

                # --- This block replaces the entire contents of `with tab1:` ---

                with tab1:
                    st.subheader('ISIN Summary')

                    # --- Create a summary of hedge counterparties ---
                    df_hedges = df_final[df_final['TradeType'] == 'HedgeTrade'].copy()
                    if not df_hedges.empty:
                        hedge_cptys = df_hedges.groupby('Instrument')['Cpty'].apply(
                            lambda x: ', '.join(x.unique())
                        ).reset_index(name='Hedge Counterparties')
                    else:
                        hedge_cptys = pd.DataFrame(columns=['Instrument', 'Hedge Counterparties'])

                    # --- Original aggregation logic ---
                    df_final['time_lag_td'] = pd.to_timedelta(df_final['time_lag'], errors='coerce')
                    agg_dict = {
                        'Name': ('Free Text', 'first'),
                        'autohedger_count': ('TradeType', lambda x: (x == 'HedgeTrade').sum()),
                        'client_trade_count': ('TradeType', lambda x: (x == 'ClientTrade').sum()),
                        'total_PnL': ('pnl', 'sum'),
                        'average_PnL': ('pnl', 'mean'),
                        'average_time_lag': ('time_lag_td', lambda x: x.dropna().dt.total_seconds().mean()),
                        'min_time_lag': ('time_lag_td', lambda x: x.dropna().dt.total_seconds().min()),
                        'max_time_lag': ('time_lag_td', lambda x: x.dropna().dt.total_seconds().max())
                    }
                    isin_summary = df_final.groupby('Instrument').agg(**agg_dict).rename(
                        columns={'Name': 'Instrument Name'})

                    # --- Merge the counterparty info into the main summary ---
                    isin_summary = isin_summary.reset_index()
                    isin_summary = pd.merge(isin_summary, hedge_cptys, on='Instrument', how='left')
                    isin_summary['Hedge Counterparties'].fillna('N/A', inplace=True)

                    # --- Original filtering and rounding ---
                    isin_summary = isin_summary[
                        (isin_summary['autohedger_count'] > 0) & (isin_summary['client_trade_count'] > 0)
                        ]
                    isin_summary = isin_summary.sort_values(by='autohedger_count', ascending=False)
                    isin_summary['average_PnL'] = isin_summary['average_PnL'].round(2)
                    isin_summary['average_time_lag'] = isin_summary['average_time_lag'].round(2)
                    isin_summary['total_PnL'] = isin_summary['total_PnL'].round(2)

                    # --- Reorder columns for better presentation ---
                    final_column_order = [
                        'Instrument', 'Instrument Name', 'Hedge Counterparties', 'autohedger_count',
                        'client_trade_count', 'total_PnL', 'average_PnL', 'average_time_lag',
                        'min_time_lag', 'max_time_lag'
                    ]
                    final_column_order = [col for col in final_column_order if col in isin_summary.columns]
                    isin_summary = isin_summary[final_column_order]

                    # --- NEW: Rename columns for a cleaner display ---
                    column_rename_map = {
                        'Instrument': 'ISIN',
                        'Instrument Name': 'Name',
                        'Hedge Counterparties': 'Hedge Counterparties',
                        'autohedger_count': 'Hedge Count',
                        'client_trade_count': 'Client Count',
                        'total_PnL': 'Total PnL',
                        'average_PnL': 'Average PnL',
                        'average_time_lag': 'Avg Lag (s)',
                        'min_time_lag': 'Min Lag (s)',
                        'max_time_lag': 'Max Lag (s)'
                    }
                    isin_summary.rename(columns=column_rename_map, inplace=True)


                    # --- MODIFIED: Styling must now use the NEW column names ---
                    def color_pnl(val):
                        if pd.isna(val):
                            return ''
                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                        return f'color: {color}'


                    styler = isin_summary.style
                    # Use the new names from column_rename_map here!
                    styler.format({
                        'Total PnL': '{:,.2f}',
                        'Average PnL': '{:,.2f}',
                        'Avg Lag (s)': '{:.2f}',
                        'Min Lag (s)': '{:.2f}',
                        'Max Lag (s)': '{:.2f}'
                    })
                    # Also use the new names for the subset here!
                    styler.applymap(color_pnl, subset=['Total PnL', 'Average PnL'])
                    st.dataframe(styler)

                    # --- Distribution plots (same as before) ---
                    st.markdown("---")
                    # ... (the rest of the tab1 code is unchanged) ...
                    df_final['time_lag_seconds'] = df_final['time_lag_td'].dt.total_seconds()
                    pnl_data = df_final['pnl'].dropna()
                    lag_data = df_final['time_lag_seconds'].dropna()

                    if not pnl_data.empty and not lag_data.empty:
                        st.subheader('Trade Performance Distributions')
                        pnl_lower_bound, pnl_upper_bound = pnl_data.quantile(0.01), pnl_data.quantile(0.99)
                        lag_upper_bound = lag_data.quantile(0.99)
                        pnl_plot_df = df_final[
                            (df_final['pnl'] >= pnl_lower_bound) & (df_final['pnl'] <= pnl_upper_bound)]
                        lag_plot_df = df_final[df_final['time_lag_seconds'] <= lag_upper_bound]
                        plot_col1, plot_col2 = st.columns(2)
                        with plot_col1:
                            fig1, ax1 = plt.subplots()
                            sns.histplot(ax=ax1, data=pnl_plot_df, x='pnl', kde=True, bins=40)
                            ax1.set_title('Distribution of PnL')
                            ax1.set_xlabel('PnL');
                            ax1.set_ylabel('Number of Trades')
                            st.pyplot(fig1)
                        with plot_col2:
                            fig2, ax2 = plt.subplots()
                            sns.histplot(ax=ax2, data=lag_plot_df, x='time_lag_seconds', kde=True, color='purple',
                                         bins=40)
                            ax2.set_title('Distribution of Time Lag')
                            ax2.set_xlabel('Time Lag (seconds)');
                            ax2.set_ylabel('Number of Trades')
                            st.pyplot(fig2)

                with tab2:
                    st.subheader("Check Trade Details by Instrument")

                    if not df_final.empty:
                        instrument_map = df_final[['Free Text', 'Instrument']].drop_duplicates()
                        instrument_dict = pd.Series(instrument_map.Instrument.values,
                                                    index=instrument_map['Free Text']).to_dict()
                        available_instrument_names = sorted(instrument_dict.keys())
                    else:
                        available_instrument_names = []

                    if available_instrument_names:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.selectbox("Select an Instrument:", options=available_instrument_names,
                                         key="instrument_selector_choice")
                        with col2:
                            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
                            if st.button("Show Details"):
                                st.session_state.instrument_to_show = st.session_state.instrument_selector_choice

                        if 'instrument_to_show' in st.session_state and st.session_state.instrument_to_show:
                            instrument_name_to_display = st.session_state.instrument_to_show
                            st.markdown(f"#### Details for: **{instrument_name_to_display}**")
                            selected_instrument_isin = instrument_dict[instrument_name_to_display]
                            instrument_details_df = df_final[df_final['Instrument'] == selected_instrument_isin].copy()
                            instrument_details_df['time_lag_td'] = pd.to_timedelta(instrument_details_df['time_lag'],
                                                                                   errors='coerce')

                            COLUMNS_TO_DISPLAY = [
                                'Execution Time', 'B/S', 'Qty', 'Open Position', 'Price',
                                'TradeType', 'pnl', 'return(bps)', 'time_lag'
                            ]
                            COLUMN_RENAME_MAP = {
                                'Execution Time': 'Time', 'B/S': 'Side', 'Qty': 'Quantity',
                                'Open Position': 'Pos. to Unwind', 'TradeType': 'Type',
                                'pnl': 'PnL', 'return(bps)': 'Return (bps)', 'time_lag': 'Hedge Lag'
                            }

                            display_df = instrument_details_df[
                                [col for col in COLUMNS_TO_DISPLAY if col in instrument_details_df.columns]].copy()
                            display_df.rename(columns=COLUMN_RENAME_MAP, inplace=True)


                            def style_pnl_color(val):
                                if pd.isna(val): return ''
                                color = 'green' if val > 0 else 'red' if val < 0 else ''
                                return f'color: {color}'


                            min_lag = instrument_details_df['time_lag_td'].dropna().min()
                            max_lag = instrument_details_df['time_lag_td'].dropna().max()


                            def highlight_min_max_lag(row, min_val, max_val, time_lag_col_name):
                                styles = [''] * len(row)
                                if pd.notna(row[time_lag_col_name]):
                                    current_lag_td = instrument_details_df.loc[row.name, 'time_lag_td']
                                    if pd.notna(current_lag_td):
                                        try:
                                            lag_idx = row.index.get_loc(time_lag_col_name)
                                            if current_lag_td == max_val:
                                                styles[lag_idx] = 'background-color: #F79880'
                                            elif current_lag_td == min_val:
                                                styles[lag_idx] = 'background-color: #6BDBCB'
                                        except KeyError:
                                            pass
                                return styles


                            styler = display_df.style

                            new_pnl_name = COLUMN_RENAME_MAP.get('pnl')
                            if new_pnl_name and new_pnl_name in display_df.columns:
                                styler.applymap(style_pnl_color, subset=[new_pnl_name])

                            new_time_lag_name = COLUMN_RENAME_MAP.get('time_lag')
                            if new_time_lag_name and new_time_lag_name in display_df.columns:
                                styler.apply(lambda r: highlight_min_max_lag(r, min_lag, max_lag, new_time_lag_name),
                                             axis=1)

                            original_formats = {
                                'Price': '{:.4f}', 'pnl': '{:.2f}', 'return(bps)': '{:.2f}',
                                'Open Position': '{:,.0f}', 'Qty': '{:,.0f}'
                            }
                            active_formats = {COLUMN_RENAME_MAP.get(k, k): v for k, v in original_formats.items() if
                                              k in COLUMNS_TO_DISPLAY}
                            styler.format(active_formats)

                            st.dataframe(styler, use_container_width=True)

                            st.markdown("---")
                            st.subheader("Trade Flow & PnL")

                            plot_df_1 = instrument_details_df.copy()
                            plot_df_1.rename(columns={'Open Position': 'Position'}, inplace=True)

                            plot_df_1['Execution Time'] = pd.to_datetime(plot_df_1['Execution Time'])
                            plot_df_1.sort_values('Execution Time', inplace=True)
                            plot_df_1['pnl'].fillna(0, inplace=True)
                            plot_df_1['Cumulative PnL'] = plot_df_1['pnl'].cumsum()

                            if not plot_df_1.empty:
                                last_row = plot_df_1.iloc[-1]
                                trade_date = last_row['Execution Time'].date()
                                end_timestamp = datetime.combine(trade_date, end_time)

                                if end_timestamp > last_row['Execution Time']:
                                    end_of_window_row = pd.DataFrame({
                                        'Execution Time': [end_timestamp],
                                        'Position': [last_row['Position']],
                                        'Cumulative PnL': [last_row['Cumulative PnL']]
                                    })
                                    plot_df_1 = pd.concat([plot_df_1, end_of_window_row], ignore_index=True)

                            fig_pos, ax_pos = plt.subplots(figsize=(12, 6))
                            ax_pnl = ax_pos.twinx()

                            ax_pos.plot(plot_df_1['Execution Time'], plot_df_1['Position'], color='royalblue',
                                        marker='o', linestyle='-', label='Position', drawstyle='steps-post')
                            ax_pnl.plot(plot_df_1['Execution Time'], plot_df_1['Cumulative PnL'], color='forestgreen',
                                        marker='.', linestyle='-', label='Cumulative PnL', drawstyle='steps-post')

                            if not plot_df_1.empty:
                                pos_min, pos_max = plot_df_1['Position'].min(), plot_df_1['Position'].max()
                                padding = (pos_max - pos_min) * 0.1 if (pos_max - pos_min) > 0 else 1.0
                                ax_pos.set_ylim(pos_min - padding, pos_max + padding)

                                pnl_min, pnl_max = plot_df_1['Cumulative PnL'].min(), plot_df_1['Cumulative PnL'].max()
                                padding = (pnl_max - pnl_min) * 0.1 if (pnl_max - pnl_min) > 0 else 1.0
                                ax_pnl.set_ylim(pnl_min - padding, pnl_max + padding)

                                y1_min, y1_max = ax_pos.get_ylim()
                                y2_min, y2_max = ax_pnl.get_ylim()

                                if y1_min < 0 < y1_max and y2_min < 0 < y2_max:
                                    ratio1 = y1_max / abs(y1_min)
                                    ratio2 = y2_max / abs(y2_min)

                                    if ratio1 < ratio2:
                                        ax_pos.set_ylim(y1_min, abs(y1_min) * ratio2)
                                    else:
                                        ax_pnl.set_ylim(y2_min, abs(y2_min) * ratio1)

                            ax_pos.set_xlabel('Execution Time')
                            ax_pos.set_ylabel('Position (Qty to Unwind)', color='royalblue')
                            ax_pos.tick_params(axis='y', labelcolor='royalblue')
                            ax_pos.axhline(0, color='black', linewidth=1.2)
                            ax_pnl.set_ylabel('Cumulative PnL (€)', color='forestgreen')
                            ax_pnl.tick_params(axis='y', labelcolor='forestgreen')
                            ax_pos.grid(True, linestyle=':', alpha=0.7)
                            date_format = DateFormatter("%H:%M")
                            ax_pos.xaxis.set_major_formatter(date_format)
                            lines1, labels1 = ax_pos.get_legend_handles_labels()
                            lines2, labels2 = ax_pnl.get_legend_handles_labels()
                            ax_pnl.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                            fig_pos.tight_layout()
                            st.pyplot(fig_pos)

                            st.markdown("#### Trade Flow & Return")


                            # --- NEW HELPER FUNCTION ---
                            # This function contains the logic to align the zero-point of two y-axes.
                            def align_yaxis_zero(ax1, ax2):
                                """
                                Adjusts the y-axis limits of two axes (ax1, ax2) to align their zero points.
                                This ensures that the y=0 line on both axes is at the same vertical position.
                                """
                                y1_min, y1_max = ax1.get_ylim()
                                y2_min, y2_max = ax2.get_ylim()

                                # Case 1: Both axes cross zero
                                if y1_min < 0 < y1_max and y2_min < 0 < y2_max:
                                    ratio1 = y1_max / -y1_min
                                    ratio2 = y2_max / -y2_min
                                    if ratio1 < ratio2:
                                        ax1.set_ylim(y1_min, -y1_min * ratio2)
                                    else:
                                        ax2.set_ylim(y2_min, -y2_min * ratio1)

                                # Case 2: Only the first axis (ax1) crosses zero
                                elif y1_min < 0 < y1_max:
                                    ratio1 = y1_max / -y1_min
                                    if y2_min >= 0:  # If ax2 is all non-negative
                                        ax2.set_ylim(-y2_max / ratio1, y2_max)
                                    else:  # If ax2 is all non-positive
                                        ax2.set_ylim(y2_min, -y2_min * ratio1)

                                # Case 3: Only the second axis (ax2) crosses zero
                                elif y2_min < 0 < y2_max:
                                    ratio2 = y2_max / -y2_min
                                    if y1_min >= 0:  # If ax1 is all non-negative
                                        ax1.set_ylim(-y1_max / ratio2, y1_max)
                                    else:  # If ax1 is all non-positive
                                        ax1.set_ylim(y1_min, -y1_min * ratio2)


                            plot_df_2 = instrument_details_df.dropna(subset=['time_lag_td']).copy()
                            plot_df_2_scatter = plot_df_2.dropna(subset=['return(bps)'])

                            if not plot_df_2.empty:
                                plot_df_2['Execution Time'] = pd.to_datetime(plot_df_2['Execution Time'])
                                left_edges, heights, widths, bar_color = plot_df_2['Execution Time'] - plot_df_2[
                                    'time_lag_td'], plot_df_2['Qty'], plot_df_2['time_lag_td'], 'skyblue'
                                fig_flow, ax_flow = plt.subplots(figsize=(12, 6))
                                ax_ret = ax_flow.twinx()
                                ax_flow.bar(left_edges, heights, width=widths, color=bar_color, align='edge',
                                            edgecolor='black', alpha=0.7)

                                # Set initial y-limits with padding
                                qty_data = heights
                                if not qty_data.empty:
                                    padding_qty = (qty_data.max() - qty_data.min()) * 0.1 if (
                                                                                                     qty_data.max() - qty_data.min()) > 0 else 1.0
                                    ax_flow.set_ylim(qty_data.min() - padding_qty, qty_data.max() + padding_qty)

                                if not plot_df_2_scatter.empty:
                                    ret_data = plot_df_2_scatter['return(bps)']
                                    padding_ret = (ret_data.max() - ret_data.min()) * 0.1 if (
                                                                                                     ret_data.max() - ret_data.min()) > 0 else 1.0
                                    ax_ret.set_ylim(ret_data.min() - padding_ret, ret_data.max() + padding_ret)

                                # --- APPLY THE ALIGNMENT ---
                                # This single function call replaces the previous if/else block.
                                align_yaxis_zero(ax_flow, ax_ret)

                                if not plot_df_2_scatter.empty:
                                    jitter_seconds = np.random.uniform(-2, 2, size=len(plot_df_2_scatter))
                                    jitter_timedelta = pd.to_timedelta(jitter_seconds, unit='s')
                                    jittered_time = plot_df_2_scatter['Execution Time'] + jitter_timedelta
                                    return_colors = ['forestgreen' if r >= 0 else 'crimson' for r in
                                                     plot_df_2_scatter['return(bps)']]
                                    ax_ret.scatter(x=jittered_time, y=plot_df_2_scatter['return(bps)'],
                                                   color=return_colors, marker='x', s=50, alpha=0.8)

                                ax_flow.set_xlabel('Execution Time')
                                ax_flow.set_ylabel('Trade Quantity', fontsize=12, color=bar_color)
                                ax_flow.tick_params(axis='y', labelcolor=bar_color)
                                ax_flow.axhline(0, color='black', linewidth=1.2)
                                ax_ret.set_ylabel('Return (bps)', fontsize=12)
                                quantity_patch = mpatches.Patch(color=bar_color, label='Trade Quantity')
                                pos_return_marker = Line2D([0], [0], color='forestgreen', marker='x', linestyle='None',
                                                           markersize=7, label='Positive Return (bps)')
                                neg_return_marker = Line2D([0], [0], color='crimson', marker='x', linestyle='None',
                                                           markersize=7, label='Negative Return (bps)')
                                handles = [quantity_patch]
                                if not plot_df_2_scatter.empty:
                                    handles.extend([pos_return_marker, neg_return_marker])
                                ax_ret.legend(handles=handles, loc='upper left')
                                date_format_flow = DateFormatter("%H:%M")
                                ax_flow.xaxis.set_major_formatter(date_format_flow)
                                fig_flow.tight_layout()
                                st.pyplot(fig_flow)

                    else:
                        st.info("No instruments with trade data found to display.")
        else:
            st.warning("Please upload a file and provide ISINs before running the analysis.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        import traceback

        st.error(traceback.format_exc())
        st.session_state.analysis_run = False