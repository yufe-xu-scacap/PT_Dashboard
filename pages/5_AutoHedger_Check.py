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
                # --- Data Processing (Unchanged) ---
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
                df_data = df_data.sort_values(by='Execution Time', ascending=True)


                def filter_trade_block(group):
                    try:
                        first_client_idx = (group['TradeType'] == 'ClientTrade').idxmax()
                        last_hedge_idx = (group['TradeType'] == 'HedgeTrade')[::-1].idxmax()
                        return group.loc[first_client_idx:last_hedge_idx]
                    except ValueError:
                        return None


                df_data = df_data.groupby('Instrument', group_keys=False).apply(filter_trade_block).reset_index(
                    drop=True)


                def calculate_hedge(group, qty_col='Qty', price_col='Price', time_col='Execution Time',
                                    trade_type_col='TradeType'):
                    initial_position = -group[qty_col].sum()
                    position_layers = deque()
                    current_total_qty = initial_position
                    current_wap = np.nan
                    is_initial_price_set = False

                    if abs(initial_position) > 1e-9:
                        position_layers.append({'qty': initial_position, 'price': np.nan})

                    last_client_trade_time = pd.NaT
                    results = []

                    for index, row in group.iterrows():
                        trade_qty = row[qty_col]
                        trade_price = row[price_col]
                        trade_type = row[trade_type_col]

                        if not is_initial_price_set and abs(initial_position) > 1e-9:
                            current_wap = trade_price
                            if position_layers:
                                position_layers[0]['price'] = trade_price
                            is_initial_price_set = True

                        metrics = {'pnl': np.nan, 'pnl_fifo': np.nan, 'time_lag': pd.NaT, 'relative_return_bps': np.nan}

                        is_real_hedge = (
                                trade_type == 'HedgeTrade' and
                                current_total_qty != 0 and
                                np.sign(trade_qty) != np.sign(current_total_qty)
                        )

                        if is_real_hedge:
                            if not pd.isna(current_wap):
                                pnl_per_share_wap = (trade_price - current_wap) * np.sign(current_total_qty)
                                metrics['pnl'] = round(pnl_per_share_wap * abs(trade_qty), 2)
                                if abs(current_wap) > 1e-9:
                                    relative_return = pnl_per_share_wap / current_wap
                                    metrics['relative_return_bps'] = relative_return * 10000

                            if pd.notna(last_client_trade_time):
                                metrics['time_lag'] = row[time_col] - last_client_trade_time

                        position_layers.append({'qty': trade_qty, 'price': trade_price})
                        new_total_qty = current_total_qty + trade_qty
                        if new_total_qty != 0:
                            if not pd.isna(current_wap):
                                current_wap = ((current_wap * current_total_qty) + (
                                        trade_price * trade_qty)) / new_total_qty
                            else:
                                current_wap = trade_price
                        else:
                            current_wap = 0.0
                            position_layers.clear()

                        current_total_qty = new_total_qty

                        if trade_type == 'ClientTrade':
                            last_client_trade_time = row[time_col]

                        results.append(metrics)

                    new_cols = pd.DataFrame(results, index=group.index)
                    group[['pnl', 'pnl_fifo', 'time_lag', 'return(bps)']] = new_cols
                    return group


                df_final = df_data.groupby('Instrument').apply(calculate_hedge)


                def format_timedelta_to_hms(td):
                    if pd.isna(td): return None
                    total_seconds = int(td.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f'{hours:02}:{minutes:02}:{seconds:02}'


                df_final['time_lag'] = df_final['time_lag'].apply(format_timedelta_to_hms)
                columns_to_keep = ['Instrument', 'Free Text', 'B/S', 'Qty', 'Price', 'Premium', 'Execution Time',
                                   'TradeType', 'time_lag', 'pnl', 'return(bps)', 'ExecutionWithinFirm', 'Cpty']
                df_final = df_final[columns_to_keep]
                df_final.reset_index(drop=True, inplace=True)

                tab1, tab2 = st.tabs(["📊 Summary", "📄 Details"])

                with tab1:
                    # Summary tab remains unchanged
                    st.subheader('ISIN Summary')
                    df_final['time_lag_td'] = pd.to_timedelta(df_final['time_lag'], errors='coerce')
                    isin_summary = df_final.groupby('Instrument').agg(
                        Name=('Free Text', 'first'),
                        autohedger_count=('TradeType', lambda x: (x == 'HedgeTrade').sum()),
                        client_trade_count=('TradeType', lambda x: (x == 'ClientTrade').sum()),
                        total_PnL=('pnl', 'sum'),
                        average_PnL=('pnl', 'mean'),
                        average_time_lag=('time_lag_td', lambda x: x.dt.total_seconds().mean()),
                        min_time_lag=('time_lag_td', lambda x: x.dt.total_seconds().min()),
                        max_time_lag=('time_lag_td', lambda x: x.dt.total_seconds().max())
                    ).rename(columns={'Name': 'Instrument Name'})
                    isin_summary = isin_summary[
                        (isin_summary['autohedger_count'] > 0) & (isin_summary['client_trade_count'] > 0)
                        ]
                    isin_summary = isin_summary.sort_values(by='autohedger_count', ascending=False)
                    isin_summary['average_PnL'] = isin_summary['average_PnL'].round(2)
                    isin_summary['average_time_lag'] = isin_summary['average_time_lag'].round(2)
                    isin_summary['total_PnL'] = isin_summary['total_PnL'].round(2)
                    st.dataframe(isin_summary.style.format({
                        'total_PnL': '{:,.2f}',
                        'average_PnL': '{:,.2f}',
                        'average_time_lag': '{:.2f}',
                        'min_time_lag': '{:.1f}',
                        'max_time_lag': '{:.1f}'
                    }))

                    st.markdown("---")
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


                            def style_pnl_color(val):
                                if pd.isna(val): return ''
                                color = 'green' if val > 0 else 'red' if val < 0 else ''
                                return f'color: {color}'


                            min_lag = instrument_details_df['time_lag_td'].min()
                            max_lag = instrument_details_df['time_lag_td'].max()


                            def highlight_min_max_lag(row, min_val, max_val):
                                styles = [''] * len(row)
                                if pd.notna(row['time_lag_td']) and pd.notna(min_val) and min_val != max_val:
                                    try:
                                        lag_idx = row.index.get_loc('time_lag')
                                        if row['time_lag_td'] == max_val:
                                            styles[lag_idx] = 'background-color: #F79880'
                                        elif row['time_lag_td'] == min_val:
                                            styles[lag_idx] = 'background-color: #6BDBCB'
                                    except KeyError:
                                        pass
                                return styles


                            styler = instrument_details_df.style \
                                .applymap(style_pnl_color, subset=['pnl']) \
                                .apply(lambda r: highlight_min_max_lag(r, min_lag, max_lag), axis=1)
                            styler.format({'Price': '{:.4f}', 'pnl': '{:.2f}', 'return(bps)': '{:.2f}'})
                            st.dataframe(styler, use_container_width=True)

                            st.markdown("---")
                            st.subheader("Visualizations")

                            # --- PLOT 1: Position & Cumulative PnL ---
                            st.markdown("#### Trade Position & Cumulative PnL")
                            plot_df_1 = instrument_details_df.copy()
                            plot_df_1['Execution Time'] = pd.to_datetime(plot_df_1['Execution Time'])
                            plot_df_1.sort_values('Execution Time', inplace=True)
                            plot_df_1['pnl'].fillna(0, inplace=True)
                            plot_df_1['Cumulative PnL'] = plot_df_1['pnl'].cumsum()
                            total_net_qty = plot_df_1['Qty'].sum()
                            plot_df_1['Position'] = plot_df_1['Qty'].cumsum() - total_net_qty

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

                            # --- NEW: Robust Scaling and Zero-Level Alignment ---
                            if not plot_df_1.empty:
                                # First, set initial padded limits to ensure full visibility
                                pos_min, pos_max = plot_df_1['Position'].min(), plot_df_1['Position'].max()
                                padding = (pos_max - pos_min) * 0.1 if (pos_max - pos_min) > 0 else 1.0
                                ax_pos.set_ylim(pos_min - padding, pos_max + padding)

                                pnl_min, pnl_max = plot_df_1['Cumulative PnL'].min(), plot_df_1['Cumulative PnL'].max()
                                padding = (pnl_max - pnl_min) * 0.1 if (pnl_max - pnl_min) > 0 else 1.0
                                ax_pnl.set_ylim(pnl_min - padding, pnl_max + padding)

                                # Second, align the zero-levels if both axes cross zero
                                y1_min, y1_max = ax_pos.get_ylim()
                                y2_min, y2_max = ax_pnl.get_ylim()

                                if y1_min < 0 < y1_max and y2_min < 0 < y2_max:
                                    ratio1 = y1_max / abs(y1_min)
                                    ratio2 = y2_max / abs(y2_min)

                                    if ratio1 < ratio2:
                                        # Adjust Position axis to match PnL ratio by extending the top
                                        ax_pos.set_ylim(y1_min, abs(y1_min) * ratio2)
                                    else:
                                        # Adjust PnL axis to match Position ratio by extending the top
                                        ax_pnl.set_ylim(y2_min, abs(y2_min) * ratio1)
                            # --- END NEW ---

                            ax_pos.set_xlabel('Execution Time')
                            ax_pos.set_ylabel('Position (Qty to Unwind)', color='royalblue')
                            ax_pos.tick_params(axis='y', labelcolor='royalblue')
                            ax_pos.axhline(0, color='black', linewidth=1.2)
                            ax_pnl.set_ylabel('Cumulative PnL ($)', color='forestgreen')
                            ax_pnl.tick_params(axis='y', labelcolor='forestgreen')
                            ax_pos.grid(True, linestyle=':', alpha=0.7)
                            date_format = DateFormatter("%H:%M:%S")
                            ax_pos.xaxis.set_major_formatter(date_format)
                            lines1, labels1 = ax_pos.get_legend_handles_labels()
                            lines2, labels2 = ax_pnl.get_legend_handles_labels()
                            ax_pnl.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                            fig_pos.tight_layout()
                            st.pyplot(fig_pos)

                            # --- PLOT 2: Trade Flow and Profitability ---
                            st.markdown("#### Trade Flow and Profitability")
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

                                # --- NEW: Robust Scaling and Zero-Level Alignment for Plot 2 ---
                                qty_data = heights
                                if not qty_data.empty:
                                    padding = (qty_data.max() - qty_data.min()) * 0.1 if (
                                                                                                     qty_data.max() - qty_data.min()) > 0 else 1.0
                                    ax_flow.set_ylim(qty_data.min() - padding, qty_data.max() + padding)

                                if not plot_df_2_scatter.empty:
                                    ret_data = plot_df_2_scatter['return(bps)']
                                    padding = (ret_data.max() - ret_data.min()) * 0.1 if (
                                                                                                     ret_data.max() - ret_data.min()) > 0 else 1.0
                                    ax_ret.set_ylim(ret_data.min() - padding, ret_data.max() + padding)

                                y1_min, y1_max = ax_flow.get_ylim()
                                y2_min, y2_max = ax_ret.get_ylim()

                                if y1_min < 0 < y1_max and y2_min < 0 < y2_max:
                                    ratio1 = y1_max / abs(y1_min)
                                    ratio2 = y2_max / abs(y2_min)

                                    if ratio1 < ratio2:
                                        ax_flow.set_ylim(y1_min, abs(y1_min) * ratio2)
                                    else:
                                        ax_ret.set_ylim(y2_min, abs(y2_min) * ratio1)
                                # --- END NEW ---

                                if not plot_df_2_scatter.empty:
                                    jitter_seconds = np.random.uniform(-2, 2, size=len(plot_df_2_scatter))
                                    jitter_timedelta = pd.to_timedelta(jitter_seconds, unit='s')
                                    jittered_time = plot_df_2_scatter['Execution Time'] + jitter_timedelta
                                    return_colors = ['forestgreen' if r >= 0 else 'crimson' for r in
                                                     plot_df_2_scatter['return(bps)']]
                                    ax_ret.scatter(x=jittered_time, y=plot_df_2_scatter['return(bps)'],
                                                   color=return_colors, marker='x', s=50, alpha=0.8)

                                ax_flow.set_xlabel('Execution Time', fontsize=12)
                                ax_flow.set_ylabel('Hedge Quantity', fontsize=12, color=bar_color)
                                ax_flow.tick_params(axis='y', labelcolor=bar_color)
                                ax_flow.axhline(0, color='black', linewidth=1.2)
                                ax_ret.set_ylabel('Return (bps)', fontsize=12)

                                quantity_patch = mpatches.Patch(color=bar_color, label='Hedge Quantity')
                                pos_return_marker = Line2D([0], [0], color='forestgreen', marker='x', linestyle='None',
                                                           markersize=7, label='Positive Return (bps)')
                                neg_return_marker = Line2D([0], [0], color='crimson', marker='x', linestyle='None',
                                                           markersize=7, label='Negative Return (bps)')
                                handles = [quantity_patch]
                                if not plot_df_2_scatter.empty: handles.extend([pos_return_marker, neg_return_marker])
                                ax_ret.legend(handles=handles, loc='upper left')
                                date_format_flow = DateFormatter("%H:%M:%S")
                                ax_flow.xaxis.set_major_formatter(date_format_flow)
                                fig_flow.autofmt_xdate()
                                fig_flow.tight_layout()
                                st.pyplot(fig_flow)

                    else:
                        st.info("No instruments with trade data found to display.")

        else:
            st.warning("Please upload a file and provide ISINs before running the analysis.")

    except ValueError:
        st.error("Invalid time format. Please use HH:MM:SS.")
        st.session_state.analysis_run = False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.analysis_run = False