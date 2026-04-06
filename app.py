# =========================================================
# app.py — BESSelligence Streamlit App
# Main entry point. Run with: streamlit run app.py
# =========================================================
# import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
# import shap
import plotly.graph_objects as go


# =========================================================
# Page configuration — must be the first Streamlit call
# =========================================================
st.set_page_config(
    page_title="BESSelligence | Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
)

# =========================================================
# Load model artifacts (cached so they load only once)
# =========================================================

@st.cache_data
def load_features():
    """Load the full feature dataset for SHAP computation."""
    df = pd.read_csv("data/df_features.csv", parse_dates=["timestamp"])
    # Keep only the test set period
    df = df[df["timestamp"] >= "2025-07-01"].reset_index(drop=True)
    return df

df_features = load_features()

@st.cache_resource
def load_model():
    """Load the trained CatBoost model from disk."""
    import joblib
    model = joblib.load("models/catboost_wattwise.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return model, feature_cols

@st.cache_data
def load_test_predictions():
    """Load the test set with actual vs predicted prices."""
    df = pd.read_csv("data/test_predictions.csv", parse_dates=["timestamp"])
    return df

model, feature_cols = load_model()
df_test = load_test_predictions()

# =========================================================
# Sidebar — project info
# =========================================================
with st.sidebar:
    st.title("⚡ BESSelligence")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "BESSelligence forecasts **day-ahead electricity prices** "
        "for the German market (EPEX SPOT) using a CatBoost model "
        "trained on ENTSO-E generation, weather, and fuel price data."
    )
    st.markdown("---")

    st.markdown("### Model")
    st.markdown(
        "- **Algorithm:** CatBoost Regressor\n"
        "- **Features:** 40\n"
        "- **Train period:** 2019 – 2024\n"
        "- **Test period:** Jul 2025 – Mar 2026\n"
    )
    st.markdown("---")

    # =========================================================
    # Battery parameter sliders
    # =========================================================
    st.markdown("### 🔋 Battery Parameters")
    st.markdown("Adjust the battery configuration used in the optimizer.")

    SOC_MAX_MWH = st.slider(
        "Battery Capacity [MWh]",
        min_value=1.0, max_value=10.0, value=4.0, step=0.5,
    )
    MAX_CHARGE_MW = st.slider(
        "Max Charge Power [MW]",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    )
    MAX_DISCHARGE_MW = st.slider(
        "Max Discharge Power [MW]",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    )
    ETA_CHARGE    = 0.95 
    ETA_DISCHARGE = 0.95  

    # SOC init and final are locked to 50% of capacity
    SOC_INIT_MWH  = SOC_MAX_MWH * 0.5
    SOC_FINAL_MWH = SOC_MAX_MWH * 0.5
    SOC_MIN_MWH   = 0.0
    DT            = 1.0

    st.caption(f"Initial & final SOC locked at 50% → {SOC_INIT_MWH:.1f} MWh")

    st.markdown("---")
    st.caption("Neue Fische Data Science Bootcamp · 2025")
    st.caption("Data available through March 2026.")

# =========================================================
# Main area — tabs
# =========================================================
st.title("⚡ BESSelligence — Electricity Price Forecasting")
st.markdown("Day-ahead price forecasting for the German electricity market")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Historical Predictions",
    "📊 Model Metrics",
    "🔍 Feature Importance (SHAP)",
    "🔋 Battery Optimizer",
])

# =========================================================
# Placeholder content
# =========================================================
with tab1:
    st.subheader("Actual vs. Predicted Electricity Prices")
    st.markdown("CatBoost model predictions on the **test set (Jul 2025 – Mar 2026)**")

    # =========================================================
    # Date range filter
    # =========================================================
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input(
            "From",
            value=df_test["timestamp"].min().date(),
            min_value=df_test["timestamp"].min().date(),
            max_value=df_test["timestamp"].max().date(),
            format="DD.MM.YYYY",  # add this
        )
    with col2:
        date_to = st.date_input(
            "To",
            value=df_test["timestamp"].max().date(),
            min_value=df_test["timestamp"].min().date(),
            max_value=df_test["timestamp"].max().date(),
            format="DD.MM.YYYY",  # add this
        )

    # Filter dataframe based on selected date range
    mask = (df_test["timestamp"].dt.date >= date_from) & (df_test["timestamp"].dt.date <= date_to)
    df_filtered = df_test.loc[mask].copy()

    # =========================================================
    # Main line chart — actual vs predicted
    # =========================================================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"],
        y=df_filtered["price"],
        name="Actual",
        line=dict(color="#1f77b4", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"],
        y=df_filtered["predicted"],
        name="Predicted",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (EUR/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # Quick stats below the chart
    # =========================================================
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    mae_filtered = np.mean(np.abs(df_filtered["price"] - df_filtered["predicted"]))
    rmse_filtered = np.sqrt(np.mean((df_filtered["price"] - df_filtered["predicted"]) ** 2))
    r2_filtered = 1 - np.sum((df_filtered["price"] - df_filtered["predicted"]) ** 2) / \
                      np.sum((df_filtered["price"] - df_filtered["price"].mean()) ** 2)

    c1.metric("MAE", f"{mae_filtered:.2f} EUR/MWh")
    c2.metric("RMSE", f"{rmse_filtered:.2f} EUR/MWh")
    c3.metric("R²", f"{r2_filtered:.3f}")
    

with tab2:
    st.subheader("Model Performance Metrics")
    st.markdown("Evaluation on the **test set (Jul 2025 – Mar 2026)** — data the model has never seen.")

    # =========================================================
    # Compute metrics on the full test set
    # =========================================================
    y_true = df_test["price"]
    y_pred = df_test["predicted"]

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    # =========================================================
    # Metric cards — top row
    # =========================================================
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f} EUR/MWh", help="Mean Absolute Error — average prediction error per hour")
    c2.metric("RMSE", f"{rmse:.2f} EUR/MWh", help="Root Mean Squared Error — penalizes large errors more")
    c3.metric("R²", f"{r2:.3f}", help="How much price variance the model explains (1.0 = perfect)")

    st.markdown("---")

    # =========================================================
    # Model vs Naive benchmark comparison
    # =========================================================
    st.markdown("### Model vs. Naive Benchmark")
    st.markdown("Naive benchmark predicts yesterday's price for the same hour (D-1 / D-7 rule).")

    # Compute naive benchmark — same logic as in Ati_model.ipynb
    df_bench = df_test.copy()
    df_bench["naive"] = np.where(
        df_bench["timestamp"].dt.dayofweek.isin([1, 2, 3, 4, 6]),
        df_bench["price"].shift(24),
        df_bench["price"].shift(168),
    )
    df_bench = df_bench.dropna(subset=["naive"])

    naive_mae  = np.mean(np.abs(df_bench["price"] - df_bench["naive"]))
    naive_rmse = np.sqrt(np.mean((df_bench["price"] - df_bench["naive"]) ** 2))
    naive_r2   = 1 - np.sum((df_bench["price"] - df_bench["naive"]) ** 2) / \
                     np.sum((df_bench["price"] - df_bench["price"].mean()) ** 2)

    # Build comparison dataframe
    df_comparison = pd.DataFrame({
        "Metric": ["MAE (EUR/MWh)", "RMSE (EUR/MWh)", "R²"],
        "CatBoost": [f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.3f}"],
        "Naive Benchmark": [f"{naive_mae:.2f}", f"{naive_rmse:.2f}", f"{naive_r2:.3f}"],
    })

    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================
    # Actual vs predicted scatter plot
    # =========================================================
    st.markdown("### Actual vs. Predicted Scatter")

    fig_scatter = go.Figure()

    # Scatter points
    fig_scatter.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(color="#1f77b4", opacity=0.3, size=4),
        name="Predictions",
    ))

    # Perfect prediction line
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        name="Perfect prediction",
    ))

    fig_scatter.update_layout(
        xaxis_title="Actual Price (EUR/MWh)",
        yaxis_title="Predicted Price (EUR/MWh)",
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Feature Importance (SHAP)")
    st.markdown("SHAP values show how much each feature **contributes to the model's predictions**.")

    # =========================================================
    # Compute SHAP values (cached so it runs only once)
    # =========================================================
    @st.cache_data
    def compute_shap(_model, feature_cols):
        """Compute SHAP values on a sample of the test set for performance."""
        import shap

        # Use a sample of 500 rows — full test set would be too slow in the app
        df_sample = df_features[feature_cols].sample(500, random_state=42)

        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(df_sample)

        return shap_values, df_sample

    shap_values, df_sample = compute_shap(model, feature_cols)

    # =========================================================
    # Bar chart — mean absolute SHAP value per feature
    # =========================================================
    

    # Compute mean absolute SHAP value for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Build a dataframe and sort by importance
    df_shap = pd.DataFrame({
        "feature": feature_cols,
        "importance": mean_shap,
    }).sort_values("importance", ascending=True)

    # Plot horizontal bar chart
    fig_shap = go.Figure()

    fig_shap.add_trace(go.Bar(
        x=df_shap["importance"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(color="#1f77b4"),
    ))

    fig_shap.update_layout(
        xaxis_title="Mean |SHAP value| (EUR/MWh)",
        yaxis_title="",
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # =========================================================
    # Top 5 most important features — plain text explanation
    # =========================================================
    st.markdown("### Top 5 Most Important Features")

    # =========================================================
    # SHAP explanation text
    # =========================================================
    st.markdown("### How to Read This Chart")
    st.markdown("""
    **What is SHAP?**  
    SHAP (SHapley Additive exPlanations) is a method for explaining machine learning model predictions.
    It tells us *how much* each feature pushes the predicted price **up or down** compared to the average prediction.

    **How to read the bar chart:**
    - Each bar shows the **average absolute impact** of that feature on the predicted price (in EUR/MWh)
    - A longer bar = the feature has a **bigger influence** on the model's output
    - This is averaged across 500 randomly sampled predictions from the test set

    **Example:**  
    If `price_lag_24h` has a SHAP value of 8.5 EUR/MWh, it means that on average,  
    yesterday's price at the same hour shifts today's prediction by ±8.5 EUR/MWh.

    **Key features explained:**
    - `price_lag_24h` / `price_lag_168h` — yesterday's and last week's price at the same hour
    - `residual_load` — electricity demand minus renewable generation (higher = more thermal plants needed = higher price)
    - `total_wind_forecast` — more wind = more cheap renewable energy = lower price
    - `price_volatility_24h` — how much prices fluctuated in the last 24 hours
    - `is_crisis_period` — flag for the 2021–2023 energy crisis (extreme price regime)
    """)

    top5 = df_shap.sort_values("importance", ascending=False).head(5)

    for i, row in top5.iterrows():
        st.markdown(f"**{row['feature']}** — avg impact: `{row['importance']:.2f}` EUR/MWh")

with tab4:
    st.subheader("Battery Energy Storage Optimizer")
    st.markdown(
        "Select a day from the test set. The optimizer finds the best **charge/discharge schedule** "
        "to maximize arbitrage profit using our CatBoost price forecast."
    )

    # =========================================================
    # Day selector — only dates available in the test set
    # =========================================================
    available_dates = sorted(df_test["timestamp"].dt.date.unique())

    selected_date = st.selectbox(
        "Select a day to optimize",
        options=available_dates,
        index=len(available_dates) - 2,  # default: second to last day
    )

    # =========================================================
    # Filter test data for selected day
    # =========================================================
    day_mask = df_test["timestamp"].dt.date == selected_date
    day_df = df_test.loc[day_mask].sort_values("timestamp").reset_index(drop=True)

    if len(day_df) < 24:
        st.warning("Selected day has fewer than 24 hours of data. Please select another day.")
    else:
        # Extract price vectors for the selected day
        prices_actual   = day_df["price"].to_numpy()
        prices_forecast = day_df["predicted"].to_numpy()
        hours           = np.arange(24)

        # =========================================================
        # Core battery optimization function using scipy linprog
        # =========================================================
        def run_battery_optimization(prices, soc_max, soc_min, soc_init, soc_final,
                                      max_charge, max_discharge, eta_c, eta_d, dt):
            """
            Solve the battery arbitrage linear program for a 24-hour price vector.
            All battery parameters are passed as arguments so Streamlit re-runs
            the optimization whenever sidebar sliders change.
            Returns: charge [MWh], discharge [MWh], soc [MWh] arrays
            """
            from scipy.optimize import linprog

            n = 24
            n_vars = 3 * n  # [charge x24 | discharge x24 | soc x24]

            # Variable index offsets
            charge_start    = 0
            discharge_start = n
            soc_start       = 2 * n

            # Objective: minimize (price * charge - price * discharge)
            # i.e. maximize profit from selling high and buying low
            c_obj = np.concatenate([prices, -prices, np.zeros(n)])

            # Variable bounds
            charge_bounds    = [(0, max_charge * dt)] * n
            discharge_bounds = [(0, max_discharge * dt)] * n
            soc_bounds       = [(soc_min, soc_max)] * n
            bounds           = charge_bounds + discharge_bounds + soc_bounds

            # Equality constraints: SOC balance equations
            A_eq = []
            b_eq = []

            # Hour 0: SOC_0 = soc_init + eta_c * charge_0 - discharge_0 / eta_d
            row = np.zeros(n_vars)
            row[charge_start]    = -eta_c
            row[discharge_start] =  1 / eta_d
            row[soc_start]       =  1
            A_eq.append(row)
            b_eq.append(soc_init)

            # Hours 1-23: SOC_t = SOC_(t-1) + eta_c * charge_t - discharge_t / eta_d
            for t in range(1, n):
                row = np.zeros(n_vars)
                row[charge_start + t]    = -eta_c
                row[discharge_start + t] =  1 / eta_d
                row[soc_start + t]       =  1
                row[soc_start + (t - 1)] = -1
                A_eq.append(row)
                b_eq.append(0)

            # Final SOC constraint: SOC_23 = soc_final
            row = np.zeros(n_vars)
            row[soc_start + (n - 1)] = 1
            A_eq.append(row)
            b_eq.append(soc_final)

            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)

            # Solve the linear program using HiGHS solver
            result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            # Extract optimal schedule from solution vector
            x = result.x
            charge_opt    = x[charge_start:discharge_start]
            discharge_opt = x[discharge_start:soc_start]
            soc_opt       = x[soc_start:]

            return charge_opt, discharge_opt, soc_opt

        # =========================================================
        # Run optimization for both forecast and perfect foresight
        # =========================================================
        charge_fcast, discharge_fcast, soc_fcast = run_battery_optimization(
            prices_forecast, SOC_MAX_MWH, SOC_MIN_MWH, SOC_INIT_MWH, SOC_FINAL_MWH,
            MAX_CHARGE_MW, MAX_DISCHARGE_MW, ETA_CHARGE, ETA_DISCHARGE, DT
        )
        charge_perfect, discharge_perfect, soc_perfect = run_battery_optimization(
            prices_actual, SOC_MAX_MWH, SOC_MIN_MWH, SOC_INIT_MWH, SOC_FINAL_MWH,
            MAX_CHARGE_MW, MAX_DISCHARGE_MW, ETA_CHARGE, ETA_DISCHARGE, DT
        )

        # Compute hourly cashflows using actual prices (what really happened)
        cashflow_fcast   = prices_actual * discharge_fcast   - prices_actual * charge_fcast
        cashflow_perfect = prices_actual * discharge_perfect - prices_actual * charge_perfect

        # Compute cumulative revenue through the day
        cumrev_fcast   = np.cumsum(cashflow_fcast)
        cumrev_perfect = np.cumsum(cashflow_perfect)

        # Daily profit summary
        profit_fcast   = cashflow_fcast.sum()
        profit_perfect = cashflow_perfect.sum()

        # =========================================================
        # Profit summary cards
        # =========================================================
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast-based Profit", f"€ {profit_fcast:,.2f}")
        m2.metric("Perfect Foresight Profit", f"€ {profit_perfect:,.2f}")
        m3.metric("Profit Gap", f"€ {profit_perfect - profit_fcast:,.2f}",
                  delta=f"{((profit_fcast / profit_perfect) * 100):.1f}% captured" if profit_perfect > 0 else "N/A")

        st.markdown("---")

        # =========================================================
        # 4 charts — same layout as in the notebook
        # =========================================================
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Battery State of Charge",
                "Charge / Discharge Power",
                "Electricity Price",
                "Cumulative Revenue",
            ),
            vertical_spacing=0.08,
        )

        # --- Row 1: State of Charge ---
        fig.add_trace(go.Scatter(
            x=hours, y=soc_fcast,
            name="SOC (forecast)", line=dict(color="#f0c040", width=2)
        ), row=1, col=1)

        # --- Row 2: Charge / Discharge bars ---
        fig.add_trace(go.Bar(
            x=hours, y=discharge_fcast,
            name="Discharge", marker_color="#4caf50"
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=hours, y=[-c for c in charge_fcast],
            name="Charge", marker_color="#f44336"
        ), row=2, col=1)

        # --- Row 3: Electricity price ---
        fig.add_trace(go.Scatter(
            x=hours, y=prices_actual,
            name="Actual price", line=dict(color="#f0c040", width=2)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=hours, y=prices_forecast,
            name="Forecast price", line=dict(color="#90caf9", width=1.5, dash="dot")
        ), row=3, col=1)

        # --- Row 4: Cumulative revenue ---
        fig.add_trace(go.Scatter(
            x=hours, y=cumrev_fcast,
            name="Forecast revenue", line=dict(color="#4caf50", width=2)
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=hours, y=cumrev_perfect,
            name="Perfect foresight", line=dict(color="#90caf9", width=1.5, dash="dash")
        ), row=4, col=1)

        # Layout
        fig.update_layout(
            height=850,
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            barmode="relative",
        )
        fig.update_yaxes(title_text="SOC [MWh]",    row=1, col=1)
        fig.update_yaxes(title_text="Power [MWh]",  row=2, col=1)
        fig.update_yaxes(title_text="EUR/MWh",      row=3, col=1)
        fig.update_yaxes(title_text="Revenue [EUR]", row=4, col=1)
        fig.update_xaxes(title_text="Hour of Day",  row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)