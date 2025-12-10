import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Slope Trading Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 0;
    }

    .block-container {
        padding-top: 3rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-return {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .negative-return {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
    }
    
    .neutral-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Larger sidebar text */
    [data-testid="stSidebar"] {
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] h1 { font-size: 2rem !important; }
    [data-testid="stSidebar"] h2 { font-size: 1.6rem !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.4rem !important; }
    [data-testid="stSidebar"] h4 { font-size: 1.25rem !important; }

    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 24px;
        padding-right: 24px;
        background-color: #e9ecef;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] button,
    .stTabs [data-baseweb="tab"] div {
        font-size: 1.2rem !important;
    }

    .stTabs button[data-baseweb="tab"] {
        font-size: 1.2rem !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6;
        border-color: #adb5bd;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e9ecef;
        color: #667eea;
        border: 2px solid #667eea;
        text-decoration: underline;
    }

    .stTabs [aria-selected="true"]:hover {
        background-color: #dee2e6;
        border-color: #5a6fd6;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* YEARLY MINI CARDS */
    .yearly-container {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;        /* Prevent wrapping */
        justify-content: flex-start;
        align-items: flex-start;
        gap: 16px;
        overflow-x: auto;         /* Allow horizontal scroll on small screens */
        padding-bottom: 6px;
    }

    .yearly-card {
        background: #f8f9fa;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        min-width: 160px;
        max-width: 160px;
        border-left: 6px solid #888;
        flex-shrink: 0;           /* Prevent shrinking: stays same width */
    }

    .yearly-card.positive {
        border-color: #2ecc71;
    }

    .yearly-card.negative {
        border-color: #e74c3c;
    }

    .yearly-title {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .yearly-line {
        font-size: 0.75rem;
        margin: 0.15rem 0;
    }

</style>
""", unsafe_allow_html=True)


class SlopeTradingAnalyzer:
    def __init__(self):
        self.trade_logs_path = "./trade_logs"
        self.tickers_path = "./tickers"
        
    def load_available_branches(self):
        """Load all available trading branches from the trade_logs directory"""
        try:
            csv_files = glob.glob(os.path.join(self.trade_logs_path, "*.csv"))
            branches = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
            return sorted(branches)
        except Exception as e:
            st.error(f"Error loading branches: {e}")
            return []
    
    def load_branch_data(self, branch_name):
        """Load trading data for a specific branch"""
        try:
            file_path = os.path.join(self.trade_logs_path, f"{branch_name}.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.error(f"Error loading branch {branch_name}: {e}")
            return None
    
    def load_ticker_data(self, ticker):
        """Load price data for a specific ticker"""
        try:
            file_path = os.path.join(self.tickers_path, f"{ticker}.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return df
        except Exception as e:
            st.error(f"Error loading ticker {ticker}: {e}")
            return None
    
    def extract_ticker_from_branch(self, branch_name):
        """Extract ticker symbol from branch name"""
        # Pattern: XdD_RSI_TICKER_LTxx or XdD_RSI_TICKER_LTxx_and_...
        parts = branch_name.split('_')
        if len(parts) >= 3:
            return parts[2]  # TICKER is usually the 3rd part
        return None

    def get_unique_tickers(self, branches):
        """Extract unique tickers from branch names"""
        tickers = set()
        for branch in branches:
            ticker = self.extract_ticker_from_branch(branch)
            if ticker:
                tickers.add(ticker)
        return sorted(tickers)

    def get_branches_for_ticker(self, branches, ticker):
        """Get all branches for a specific ticker"""
        return [b for b in branches if self.extract_ticker_from_branch(b) == ticker]
    
    def calculate_slope(self, prices, window):
        """Calculate slope over a rolling window"""
        slopes = []
        for i in range(len(prices)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = prices[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    # Convert to percentage
                    slope_pct = (slope * (window-1) / y[0]) * 100 if y[0] != 0 else 0
                    slopes.append(slope_pct)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=prices.index)
    
    def apply_slope_filter(self, branch_data, ticker_data, slope_window, pos_threshold, neg_threshold, signal_type="Both"):
        """Apply slope filtering with Flag-based trading logic

        signal_type: "Both" (RSI + Slope), "RSI" (RSI only), "Slope" (Slope only)
        """
        # Merge branch data with ticker data
        merged = pd.merge(branch_data, ticker_data[['Date', 'Close', 'Volume']], on='Date', how='left')
        merged = merged.sort_values('Date')

        # Calculate slope
        merged['Slope'] = self.calculate_slope(merged['Close'], slope_window)

        # Initialize new columns
        merged['Flag'] = 0  # Flag for RSI activation
        merged['Entry_Signal'] = 0
        merged['Exit_Signal'] = 0
        merged['InTrade'] = 0

        # Flag and trading logic
        flag = 0
        in_position = False
        entry_price = None
        entry_date = None

        signals = []

        for i in range(len(merged)):
            row = merged.iloc[i]

            if pd.isna(row['Slope']):
                merged.at[merged.index[i], 'Flag'] = flag
                continue

            # Flag logic: Flag turns to 1 when RSI gets activated
            if row['Active'] == 1 and flag == 0:
                flag = 1

            if signal_type == "Both":
                # BOTH: RSI activates flag, then slope confirms entry
                # Entry condition: Slope is positive (green) AND Flag is 1
                if not in_position and flag == 1 and row['Slope'] > pos_threshold:
                    in_position = True
                    entry_price = row['Close']
                    entry_date = row['Date']
                    merged.at[merged.index[i], 'Entry_Signal'] = 1
                    merged.at[merged.index[i], 'InTrade'] = 1

                elif in_position:
                    merged.at[merged.index[i], 'InTrade'] = 1

                    # Exit condition: Slope falls below positive threshold
                    if row['Slope'] <= pos_threshold:
                        in_position = False
                        exit_price = row['Close']
                        exit_date = row['Date']
                        merged.at[merged.index[i], 'Exit_Signal'] = 1
                        merged.at[merged.index[i], 'InTrade'] = 0
                        flag = 0

                        if entry_price and entry_price != 0:
                            trade_return = (exit_price - entry_price) / entry_price * 100
                            days_held = (exit_date - entry_date).days
                            signals.append({
                                'Entry_Date': entry_date,
                                'Exit_Date': exit_date,
                                'Entry_Price': entry_price,
                                'Exit_Price': exit_price,
                                'Return_Pct': trade_return,
                                'Days_Held': days_held
                            })

            elif signal_type == "RSI":
                # RSI ONLY: Enter when RSI Active turns 1, exit when it turns 0
                if not in_position and row['Active'] == 1:
                    in_position = True
                    entry_price = row['Close']
                    entry_date = row['Date']
                    merged.at[merged.index[i], 'Entry_Signal'] = 1
                    merged.at[merged.index[i], 'InTrade'] = 1

                elif in_position:
                    merged.at[merged.index[i], 'InTrade'] = 1

                    # Exit when RSI Active turns 0
                    if row['Active'] == 0:
                        in_position = False
                        exit_price = row['Close']
                        exit_date = row['Date']
                        merged.at[merged.index[i], 'Exit_Signal'] = 1
                        merged.at[merged.index[i], 'InTrade'] = 0

                        if entry_price and entry_price != 0:
                            trade_return = (exit_price - entry_price) / entry_price * 100
                            days_held = (exit_date - entry_date).days
                            signals.append({
                                'Entry_Date': entry_date,
                                'Exit_Date': exit_date,
                                'Entry_Price': entry_price,
                                'Exit_Price': exit_price,
                                'Return_Pct': trade_return,
                                'Days_Held': days_held
                            })

            elif signal_type == "Slope":
                # SLOPE ONLY: Enter when slope > pos_threshold, exit when slope <= pos_threshold
                if not in_position and row['Slope'] > pos_threshold:
                    in_position = True
                    entry_price = row['Close']
                    entry_date = row['Date']
                    merged.at[merged.index[i], 'Entry_Signal'] = 1
                    merged.at[merged.index[i], 'InTrade'] = 1

                elif in_position:
                    merged.at[merged.index[i], 'InTrade'] = 1

                    # Exit when slope falls below threshold
                    if row['Slope'] <= pos_threshold:
                        in_position = False
                        exit_price = row['Close']
                        exit_date = row['Date']
                        merged.at[merged.index[i], 'Exit_Signal'] = 1
                        merged.at[merged.index[i], 'InTrade'] = 0

                        if entry_price and entry_price != 0:
                            trade_return = (exit_price - entry_price) / entry_price * 100
                            days_held = (exit_date - entry_date).days
                            signals.append({
                                'Entry_Date': entry_date,
                                'Exit_Date': exit_date,
                                'Entry_Price': entry_price,
                                'Exit_Price': exit_price,
                                'Return_Pct': trade_return,
                                'Days_Held': days_held
                            })

            # Store current flag value
            merged.at[merged.index[i], 'Flag'] = flag

        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df['Entry_Date'] = pd.to_datetime(signals_df['Entry_Date'])
            signals_df['Exit_Date'] = pd.to_datetime(signals_df['Exit_Date'])

        return merged, signals_df
    
    def calculate_performance_metrics(self, signals_df):
        """Calculate comprehensive performance metrics"""
        if signals_df.empty:
            return {}
            
        returns = signals_df['Return_Pct'].values
        
        # Basic metrics
        total_return = np.sum(returns)
        num_trades = len(returns)
        win_rate = len(returns[returns > 0]) / num_trades * 100 if num_trades > 0 else 0
        avg_return = np.mean(returns) if num_trades > 0 else 0
        avg_days_held = np.mean(signals_df['Days_Held']) if num_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        volatility = np.std(returns) if num_trades > 0 else 0
        
        # Time in market
        if not signals_df.empty:
            total_days_in_market = signals_df['Days_Held'].sum()
            date_range = (signals_df['Exit_Date'].max() - signals_df['Entry_Date'].min()).days
            time_in_market = (total_days_in_market / date_range * 100) if date_range > 0 else 0
        else:
            time_in_market = 0
            
        return {
            'Total_Return_Pct': total_return,
            'Win_Rate_Pct': win_rate,
            'Max_Drawdown_Pct': max_drawdown,
            'Num_Trades': num_trades,
            'Time_In_Market_Pct': time_in_market,
            'Avg_Days_Held': avg_days_held,
            'Avg_Return_Pct': avg_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Volatility_Pct': volatility
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        equity = np.cumprod(1 + np.array(returns) / 100.0)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity / running_max - 1.0) * 100
        return float(np.min(drawdown))
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio (annualized)"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = np.mean(returns) / 100.0 - risk_free_rate / 252  # Daily risk-free rate
        return (excess_returns * np.sqrt(252)) / (np.std(returns) / 100.0) if np.std(returns) != 0 else 0

    def compute_yearly_stats(self, signals_df):
        """Compute per-year stats: Return, Max DD, Trades, Avg Hold"""
        if signals_df.empty:
            return {}

        df = signals_df.copy()
        df["Year"] = df["Exit_Date"].dt.year

        yearly = {}
        for year, group in df.groupby("Year"):
            returns = group["Return_Pct"].values
            if len(returns) == 0:
                continue

            # Total return (sum of trade returns in %)
            total_return = float(np.sum(returns))

            # Max DD on equity curve for that year
            equity = np.cumprod(1 + returns / 100.0)
            running_max = np.maximum.accumulate(equity)
            dd = (equity / running_max - 1.0) * 100
            max_dd = float(np.min(dd))

            yearly[year] = {
                "Return": total_return,
                "MaxDD": max_dd,
                "Trades": int(len(group)),
                "AvgHold": float(group["Days_Held"].mean())
            }

        return yearly
    
    def create_price_chart(self, merged_data, signals_df, branch_name, slope_window, pos_threshold, neg_threshold):
        """Create a clean single-panel price chart with:
        - Price line
        - Colored slope segments (green/gray)
        - Entry markers
        - Exit markers with % return
        - Vertical entry/exit lines (FULL HEIGHT)
        - RSI activation markers
        - Flag activation markers
        Limited to last 5 years of data.
        """

        colors = {
            "price_base": "#e5e7eb",
            "slope_segment_green": "#10b981",
            "slope_segment_gray": "#6b7280",
            "rsi_entry": "#3b82f6",
            "rsi_exit_green": "#16a34a",
            "rsi_exit_gray": "#6b7280",
            "rsi_activation": "#8b5cf6",
            "flag_activation": "#8b5cf6",
        }

        df = merged_data.copy().set_index("Date")

        # Restrict to last 5 years MAX
        if len(df) > 0:
            last_date = df.index.max()
            start_date = last_date - pd.DateOffset(years=5)
            df = df[df.index >= start_date]

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="No data available for the last 5 years"
            )
            return fig

        # ======================================================
        # Figure: SINGLE PANEL
        # ======================================================
        fig = go.Figure()

        # Base price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                line=dict(color=colors["price_base"], width=2),
                name="Price",
                hoverinfo="skip"
            )
        )

        # ======================================================
        # SLOPE-BASED COLORED SEGMENTS
        # ======================================================
        slope_series = df["Slope"]

        slope_entry_idx = (slope_series > pos_threshold) & (slope_series.shift(1) <= pos_threshold)
        slope_exit_idx = (slope_series < neg_threshold) & (slope_series.shift(1) >= neg_threshold)

        slope_entry_dates = slope_series.index[slope_entry_idx].tolist()
        slope_exit_dates = slope_series.index[slope_exit_idx].tolist()

        slope_trades = []
        i = j = 0

        while i < len(slope_entry_dates) and j < len(slope_exit_dates):
            entry_date = slope_entry_dates[i]

            while j < len(slope_exit_dates) and slope_exit_dates[j] <= entry_date:
                j += 1

            if j < len(slope_exit_dates):
                exit_date = slope_exit_dates[j]

                if entry_date in df.index and exit_date in df.index:
                    entry_price = df.loc[entry_date, "Close"]
                    exit_price = df.loc[exit_date, "Close"]

                    slope_return = ((exit_price - entry_price) / entry_price) * 100

                    slope_trades.append({
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "return": slope_return
                    })

                j += 1
            i += 1

        # Draw colored segments
        for seg in slope_trades:
            mask = (df.index >= seg["entry_date"]) & (df.index <= seg["exit_date"])
            segment_df = df[mask]

            if len(segment_df) > 1:
                color = colors["slope_segment_green"] if seg["return"] > 0 else colors["slope_segment_gray"]

                fig.add_trace(
                    go.Scatter(
                        x=segment_df.index,
                        y=segment_df["Close"],
                        mode="lines",
                        line=dict(color=color, width=4),
                        name="Slope Segment",
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

        # ======================================================
        # ENTRY + EXIT MARKERS
        # ======================================================
        shapes = []  # for vertical lines

        if not signals_df.empty:
            mask_signals = signals_df["Entry_Date"] >= df.index.min()
            s_df = signals_df[mask_signals].copy()

            if not s_df.empty:
                # Full-height ENTRY lines
                for d in s_df["Entry_Date"]:
                    shapes.append(dict(
                        type="line",
                        x0=d, x1=d,
                        y0=0, y1=1,
                        xref="x",
                        yref="paper",
                        layer="below",
                        line=dict(color="rgba(59,130,246,0.70)", width=2, dash="solid")
                    ))

                # Full-height EXIT lines
                for d in s_df["Exit_Date"]:
                    shapes.append(dict(
                        type="line",
                        x0=d, x1=d,
                        y0=0, y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="rgba(22,163,74,0.40)", width=1.2, dash="dot")
                    ))

                # Entry markers
                fig.add_trace(
                    go.Scatter(
                        x=s_df["Entry_Date"],
                        y=s_df["Entry_Price"],
                        mode="markers+text",
                        marker=dict(color=colors["rsi_entry"], size=14, symbol="line-ns",
                                    line=dict(color=colors["rsi_entry"], width=3)),
                        text=["ENTRY"] * len(s_df),
                        textposition="top center",
                        name="RSI Entry",
                        hoverinfo="skip"
                    )
                )

                # Exit markers
                exit_colors = [colors["rsi_exit_green"] if r > 0 else colors["rsi_exit_gray"]
                            for r in s_df["Return_Pct"]]

                fig.add_trace(
                    go.Scatter(
                        x=s_df["Exit_Date"],
                        y=s_df["Exit_Price"],
                        mode="markers+text",
                        marker=dict(color=exit_colors, size=14, symbol="triangle-down",
                                    line=dict(color="white", width=1)),
                        text=[f"{r:+.1f}%" for r in s_df["Return_Pct"]],
                        textposition="bottom center",
                        name="Exit",
                        hoverinfo="skip"
                    )
                )


        # ======================================================
        # FINAL CLEAN LAYOUT
        # ======================================================

        last_date = df.index.max()
        start_date = df.index.min()

        fig.update_layout(
            shapes=shapes,   # <-- vertical lines added here
            template="plotly_white",
            hovermode=False,
            height=750,
            margin=dict(l=60, r=40, t=100, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),

            # X-AXIS
            xaxis=dict(
                range=[start_date, last_date],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.15)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(0,0,0,0.40)",
                linewidth=1.2,
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=4, label="4Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ],
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=12)
                ),
                rangeslider=dict(
                    visible=True,
                    thickness=0.05,
                    bgcolor="rgba(230,230,230,0.4)"
                ),
                type="date"
            ),

            # Y-AXIS
            yaxis=dict(
                title="Price",
                tickformat="$,.2f",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.12)",
                gridwidth=1.2,
                zeroline=False,
                showline=True,
                linecolor="rgba(0,0,0,0.40)",
                linewidth=1.2,
            ),
        )

        return fig

    def compute_branch_cagr(self, total_return_pct, years):
        """Compute CAGR from total return percentage and years."""
        if years <= 0 or total_return_pct is None:
            return 0.0
        total_return_decimal = total_return_pct / 100.0
        if total_return_decimal <= -1:
            return -100.0
        cagr = ((1 + total_return_decimal) ** (1 / years) - 1) * 100
        return cagr

    def compute_all_branch_overviews(self, branches, slope_window, pos_threshold, neg_threshold, signal_type):
        """Compute overview metrics for all branches."""
        overview_data = []
        for branch in branches:
            branch_data = self.load_branch_data(branch)
            if branch_data is None:
                continue

            ticker = self.extract_ticker_from_branch(branch)
            if not ticker:
                continue

            ticker_data = self.load_ticker_data(ticker)
            if ticker_data is None:
                continue

            merged_data, signals_df = self.apply_slope_filter(
                branch_data, ticker_data, slope_window, pos_threshold, neg_threshold, signal_type
            )

            if signals_df.empty:
                continue

            metrics = self.calculate_performance_metrics(signals_df)
            total_return = metrics.get('Total_Return_Pct', 0)

            first_entry = signals_df['Entry_Date'].min()
            last_exit = signals_df['Exit_Date'].max()
            days_span = (last_exit - first_entry).days
            years_float = days_span / 365.25 if days_span > 0 else 0

            period_str = f"{first_entry.strftime('%Y-%m-%d')} to {last_exit.strftime('%Y-%m-%d')}"

            cagr = self.compute_branch_cagr(total_return, years_float)

            overview_data.append({
                'Ticker': ticker,
                'Branch': clean_branch_name(branch),
                'Branch_Raw': branch,
                'Period': period_str,
                'Return %': round(total_return, 2),
                'Max DD %': round(metrics.get('Max_Drawdown_Pct', 0), 2),
                'CAGR %': round(cagr, 2),
                'Years': round(years_float, 1),
                'TIM %': round(metrics.get('Time_In_Market_Pct', 0), 2),
                'Win Rate %': round(metrics.get('Win_Rate_Pct', 0), 2),
                'Trades': metrics.get('Num_Trades', 0)
            })

        return pd.DataFrame(overview_data)


# Initialize the analyzer
@st.cache_resource
def init_analyzer():
    return SlopeTradingAnalyzer()

def clean_branch_name(branch_raw: str) -> str:
    """
    Convert '15D_RSI_A_LT33_daily_trade_log' → '15D RSI A LT33'
    """
    # Remove suffix
    branch_raw = branch_raw.split("_daily_trade_log")[0]
    # Replace underscores with spaces
    pretty = branch_raw.replace("_", " ")

    return pretty


def display_metrics_row(metrics, yearly_dict):
    """Display total metrics and yearly breakdown in a single row with scrollable years."""
    if not metrics:
        st.warning("No metrics available.")
        return

    available_years = sorted(yearly_dict.keys(), reverse=True) if yearly_dict else []

    total_return = metrics.get('Total_Return_Pct', 0)
    win_rate = metrics.get('Win_Rate_Pct', 0)
    max_dd = metrics.get('Max_Drawdown_Pct', 0)
    num_trades = metrics.get('Num_Trades', 0)
    avg_hold = metrics.get('Avg_Days_Held', 0)

    # Create columns: Total card area (with toggle) + yearly cards area
    col_total, col_years = st.columns([1, 4])

    with col_total:
        # Small toggle checkbox above Total card
        collapsed = st.checkbox("Collapse others", key="metrics_collapsed")

        # Total card HTML
        total_html = f'''
        <div style="background: #f8f9fa; padding: 1rem 1.5rem; border-left: 6px solid #667eea; border-radius: 8px;">
        <div style="font-weight: 700; font-size: 1.3rem; margin-bottom: 0.4rem;">Total</div>
        <div style="font-size: 1.1rem;">Return: {total_return:.2f}%</div>
        <div style="font-size: 1.1rem;">Win Rate: {win_rate:.2f}%</div>
        <div style="font-size: 1.1rem;">Max DD: {max_dd:.2f}%</div>
        <div style="font-size: 1.1rem;">Trades: {num_trades}</div>
        <div style="font-size: 1.1rem;">Avg Hold: {avg_hold:.1f} days</div>
        </div>'''
        st.markdown(total_html, unsafe_allow_html=True)

    # Only show yearly cards if not collapsed
    if not collapsed:
        with col_years:
            year_cards_html = ""
            for year in available_years:
                stats = yearly_dict[year]
                border_color = "#2ecc71" if stats["Return"] > 0 else "#e74c3c"
                year_cards_html += f'<div style="background: #f8f9fa; padding: 1rem 1.5rem; border-left: 6px solid {border_color}; border-radius: 8px; min-width: 180px; flex-shrink: 0; display: inline-block; margin-right: 16px;"><div style="font-weight: 700; font-size: 1.3rem; margin-bottom: 0.4rem;">{year}</div><div style="font-size: 1.1rem;">Return: {stats["Return"]:.2f}%</div><div style="font-size: 1.1rem;">Max DD: {stats["MaxDD"]:.2f}%</div><div style="font-size: 1.1rem;">Trades: {stats["Trades"]}</div><div style="font-size: 1.1rem;">Avg Hold: {stats["AvgHold"]:.1f} days</div></div>'

            st.markdown(f'<div style="display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px;">{year_cards_html}</div>', unsafe_allow_html=True)




def main():
    analyzer = init_analyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("<h4 style='text-align: center; font-size: 18px; line-height: 1.4;'>Advanced RSI + Slope Filter<br>Backtesting System</h4>", unsafe_allow_html=True)
        st.header("🔧 Configuration")
        
        # Slope parameters
        st.subheader("Slope Parameters")
        slope_window = st.slider("Slope Window (days)", 5, 30, 15)
        pos_threshold = st.slider("Positive Threshold (%)", 0.0, 20.0, 5.0, 0.5)
        neg_threshold = st.slider("Negative Threshold (%)", -10.0, 10.0, 0.0, 0.5)
        
        st.divider()
        
        # Branch selection (ALL branches selected internally)
        st.subheader("Branch Selection")

        # Signal type filter
        signal_type = st.radio(
            "Signal Type",
            ["Both", "RSI", "Slope"],
            horizontal=True
        )

        available_branches = analyzer.load_available_branches()

        # Step 1: Get unique tickers and let user select one
        unique_tickers = analyzer.get_unique_tickers(available_branches)

        if not unique_tickers:
            st.error("No trading branches found in ./trade_logs/")
            st.stop()

        # Choose selection method
        selection_method = st.radio(
            "Selection Method",
            ["Filter by Ticker", "Search All Branches"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if selection_method == "Filter by Ticker":
            # Two columns for ticker and branch selection
            col_ticker, col_branch = st.columns(2)

            with col_ticker:
                selected_ticker = st.selectbox(
                    "Select Ticker",
                    unique_tickers
                )

            # Filter branches for selected ticker
            branches_for_ticker = analyzer.get_branches_for_ticker(available_branches, selected_ticker)

            # Create mapping from pretty name → raw filename
            branch_display_map = {
                clean_branch_name(b): b
                for b in branches_for_ticker
            }

            with col_branch:
                branch_pretty = st.selectbox(
                    "Select Branch",
                    list(branch_display_map.keys())
                )

            branch_to_analyze = branch_display_map[branch_pretty]

        else:
            # Direct search across all branches
            all_branch_display_map = {
                clean_branch_name(b): b
                for b in available_branches
            }

            branch_pretty = st.selectbox(
                "Search All Branches",
                list(all_branch_display_map.keys())
            )

            branch_to_analyze = all_branch_display_map[branch_pretty]

        st.divider()
        st.markdown("### Analysis Options")
        show_yearly = st.checkbox("Show Yearly Breakdown", value=True)
        show_individual_charts = st.checkbox("Show Individual Charts", value=True)
    
    # Use ALL branches for backend overall analysis
    all_branches = available_branches

    # Tabs for different views (Individual first)
    tab_individual, tab_overall, tab_reports, tab_overviews = st.tabs(
        ["Individual Analysis", "Overall Results", "Detailed Reports", "Branch Overviews"]
    )

    # ---------------------------------------------------------
    # INDIVIDUAL ANALYSIS TAB
    # ---------------------------------------------------------
    with tab_individual:
        if branch_to_analyze:
            branch_data = analyzer.load_branch_data(branch_to_analyze)
            ticker = analyzer.extract_ticker_from_branch(branch_to_analyze)
            
            if branch_data is not None and ticker:
                ticker_data = analyzer.load_ticker_data(ticker)
                
                if ticker_data is not None:
                    # Apply slope filter
                    merged_data, signals_df = analyzer.apply_slope_filter(
                        branch_data, ticker_data, slope_window, pos_threshold, neg_threshold, signal_type
                    )
                    
                    # Calculate metrics
                    metrics = analyzer.calculate_performance_metrics(signals_df)
                    yearly = analyzer.compute_yearly_stats(signals_df)

                    # Display metrics in single row: Total | 2025 | 2024 | 2023 | 2022 | 2021
                    st.subheader(f"Performance Metrics - {branch_pretty}")
                    display_metrics_row(metrics, yearly)

                    st.divider()
                    
                    # Price chart (last 5 years max)
                    if show_individual_charts:
                        st.subheader("Price Chart with Slope Segments & Signals (Last 5 Years Max)")
                        chart = analyzer.create_price_chart(
                            merged_data, signals_df, branch_to_analyze, 
                            slope_window, pos_threshold, neg_threshold
                        )
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Trade details
                    if not signals_df.empty:
                        st.subheader("Trade Details")
                        
                        trade_display = signals_df.copy()
                        trade_display['Entry_Date'] = trade_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                        trade_display['Exit_Date'] = trade_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                        trade_display = trade_display.round(2)
                        
                        st.dataframe(trade_display, use_container_width=True)

    # ---------------------------------------------------------
    # OVERALL RESULTS TAB
    # ---------------------------------------------------------
    with tab_overall:
        st.header("Overall Performance Summary (All Branches)")
        
        all_results = []
        results_df = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, branch in enumerate(all_branches):
            status_text.text(f'Processing {branch}... ({i+1}/{len(all_branches)})')
            
            # Load branch data
            branch_data = analyzer.load_branch_data(branch)
            if branch_data is None:
                continue
            
            # Get ticker
            ticker = analyzer.extract_ticker_from_branch(branch)
            if not ticker:
                continue
            
            # Load ticker data
            ticker_data = analyzer.load_ticker_data(ticker)
            if ticker_data is None:
                continue
            
            # Apply slope filter
            merged_data, signals_df = analyzer.apply_slope_filter(
                branch_data, ticker_data, slope_window, pos_threshold, neg_threshold, signal_type
            )

            # Calculate metrics
            metrics = analyzer.calculate_performance_metrics(signals_df)
            metrics['Branch'] = branch
            metrics['Ticker'] = ticker
            all_results.append(metrics)
            
            progress_bar.progress((i + 1) / len(all_branches))
        
        status_text.text('Analysis complete!')
        progress_bar.empty()
        status_text.empty()
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_return = results_df['Total_Return_Pct'].mean()
                st.metric("Average Total Return", f"{avg_return:.2f}%")
            
            with col2:
                avg_win_rate = results_df['Win_Rate_Pct'].mean()
                st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
            
            with col3:
                avg_trades = results_df['Num_Trades'].mean()
                st.metric("Average Trades", f"{avg_trades:.1f}")
            
            with col4:
                avg_drawdown = results_df['Max_Drawdown_Pct'].mean()
                st.metric("Average Max DD", f"{avg_drawdown:.2f}%")
            
            st.divider()
            
            # Results table
            st.subheader("Branch Performance Comparison")
            
            display_df = results_df.copy().round(2)
            
            def color_returns(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else 'red'
                return f'background-color: {color}; color: white'
            
            styled_df = display_df.style.applymap(
                color_returns, 
                subset=['Total_Return_Pct', 'Max_Drawdown_Pct']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Performance visualization
            st.subheader("Performance Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter = px.scatter(
                    results_df, 
                    x='Win_Rate_Pct', 
                    y='Total_Return_Pct',
                    hover_data=['Branch', 'Num_Trades'],
                    title='Return vs Win Rate',
                    labels={'Win_Rate_Pct': 'Win Rate (%)', 'Total_Return_Pct': 'Total Return (%)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                fig_hist = px.histogram(
                    results_df, 
                    x='Total_Return_Pct',
                    title='Return Distribution',
                    labels={'Total_Return_Pct': 'Total Return (%)'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------------------------------------
    # DETAILED REPORTS TAB
    # ---------------------------------------------------------
    with tab_reports:
        st.header("Detailed Reports")
        
        # Use results_df from Overall tab if available
        if 'results_df' in locals() and results_df is not None:
            st.subheader("Export Results")
            
            if st.button("Generate CSV Report"):
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"slope_trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            if show_yearly:
                st.subheader("Yearly Performance Breakdown (Note: per-branch yearly stats shown in Individual Analysis)")
                st.info("Global yearly aggregation across all branches can be added here if needed.")
            
            st.subheader("Current Parameters Summary")
            param_df = pd.DataFrame({
                'Parameter': ['Slope Window', 'Positive Threshold', 'Negative Threshold'],
                'Value': [f"{slope_window} days", f"{pos_threshold}%", f"{neg_threshold}%"]
            })
            st.table(param_df)
        else:
            st.info("Run the Overall Results tab at least once to populate the reports.")

    # ---------------------------------------------------------
    # BRANCH OVERVIEWS TAB
    # ---------------------------------------------------------
    with tab_overviews:
        st.header("Branch Overviews")

        with st.spinner("Computing metrics for all branches..."):
            overview_df = analyzer.compute_all_branch_overviews(
                all_branches, slope_window, pos_threshold, neg_threshold, signal_type
            )

        if not overview_df.empty:
            st.subheader(f"All Branches ({len(overview_df)} total)")

            display_df = overview_df.drop(columns=['Branch_Raw'])

            html_table = display_df.to_html(index=False, border=0, escape=False)
            styled_html = f"""
            <div style="overflow-x: auto;">
            <style>
            .overview-table {{ font-size: 1.15rem !important; width: 100%; border-collapse: collapse; }}
            .overview-table th {{ background-color: #f0f2f6; padding: 12px 10px; text-align: left; border-bottom: 2px solid #ccc; font-weight: 600; }}
            .overview-table td {{ padding: 10px; border-bottom: 1px solid #eee; }}
            .overview-table tr:hover {{ background-color: #f5f5f5; }}
            </style>
            {html_table.replace('class="dataframe"', 'class="dataframe overview-table"')}
            </div>
            """
            st.markdown(styled_html, unsafe_allow_html=True)
        else:
            st.warning("No branch data available.")

if __name__ == "__main__":
    main()
