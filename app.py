import streamlit as st
import pandas as pd
import numpy as np

from model import (
    FEATURES,
    calculate_mortgage,
    get_property_persona,
    load_data,
    predict_price,
    train_model,
)

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="HomeVal AI - Smart Real Estate Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
/* Import modern clean font */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu, footer {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

/* Header & Badges */
.hero-container {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 20px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #e0e7ff;
    color: #4338ca;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #1e1b4b;
    margin: 0;
    letter-spacing: -0.8px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #4b5563;
    margin: 0.35rem 0 0;
}

/* Preset Action Bar */
.preset-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Modern Card Container */
.panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.35rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.03);
}
.panel-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Result Hero Card */
.valuation-card {
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 55%, #7c3aed 100%);
    color: #ffffff;
    border-radius: 20px;
    padding: 1.85rem;
    text-align: center;
    box-shadow: 0 16px 36px -6px rgba(79, 70, 229, 0.35);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.2rem;
}
.valuation-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.persona-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 0.75rem;
}
.valuation-label {
    font-size: 0.82rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.9;
    font-weight: 600;
}
.valuation-price {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.2px;
    line-height: 1.1;
    margin: 0.35rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.valuation-range {
    font-size: 0.98rem;
    font-weight: 500;
    opacity: 0.95;
    background: rgba(255, 255, 255, 0.15);
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 8px;
    margin-top: 0.3rem;
}
.metrics-ribbon {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    margin-top: 1.1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 0.88rem;
}
.metrics-ribbon span {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

/* Specs Bar */
.specs-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 1.2rem;
}
.spec-chip {
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #e2e8f0;
    padding: 0.35rem 0.75rem;
    border-radius: 10px;
    font-size: 0.84rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

/* Value Drivers Rows */
.driver-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.driver-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.55rem 0;
    font-size: 0.88rem;
}
.driver-name {
    width: 145px;
    font-weight: 500;
    color: #374151;
}
.driver-track {
    flex: 1;
    height: 9px;
    background: #f1f5f9;
    border-radius: 999px;
    overflow: hidden;
}
.driver-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.driver-amt {
    width: 95px;
    text-align: right;
    font-weight: 700;
}
.up { color: #16a34a; }
.down { color: #dc2626; }

/* Mortgage Summary Card */
.mortgage-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem;
    margin-top: 0.8rem;
}
.mortgage-total {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1e1b4b;
    margin: 0.2rem 0;
}
.mortgage-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.84rem;
    color: #4b5563;
    padding: 0.2rem 0;
}

/* Comparison Cards */
.compare-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.05);
    height: 100%;
}
.compare-price {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4338ca;
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)


# ----------------- Model & Data Caching -----------------
@st.cache_resource
def get_model():
    df = load_data()
    model, metrics, test = train_model(df)
    return df, model, metrics, test


df, model, metrics, test = get_model()
avg_price = float(df["price"].mean())
avg_home = df[FEATURES].mean()

# ----------------- Currency Settings -----------------
CURRENCIES = {
    "USD ($)": {"symbol": "$", "rate": 1.0},
    "EUR (€)": {"symbol": "€", "rate": 0.92},
    "GBP (£)": {"symbol": "£", "rate": 0.79},
    "INR (₹)": {"symbol": "₹", "rate": 83.5},
}

# ----------------- Presets & Session Defaults -----------------
DEFAULTS = dict(area=1800, bedrooms=3, bathrooms=2, age=10, distance=8.0, garage=1)
PRESETS = {
    "🏠 Cozy Starter": dict(area=850, bedrooms=2, bathrooms=1, age=20, distance=14.0, garage=0),
    "🏡 Suburban Family": dict(area=2100, bedrooms=3, bathrooms=2, age=8, distance=7.5, garage=1),
    "🏰 Executive Estate": dict(area=3600, bedrooms=5, bathrooms=4, age=3, distance=4.0, garage=2),
    "🌆 Urban Loft": dict(area=1100, bedrooms=2, bathrooms=1, age=5, distance=2.5, garage=1),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("saved_properties", [])
st.session_state.setdefault("currency_choice", "USD ($)")


def apply_preset(name):
    st.session_state.update(PRESETS[name])


def reset_defaults():
    st.session_state.update(DEFAULTS)


curr_info = CURRENCIES[st.session_state.currency_choice]
curr_symbol = curr_info["symbol"]
curr_rate = curr_info["rate"]


def fmt_curr(val):
    converted = val * curr_rate
    return f"{curr_symbol}{converted:,.0f}"


# ----------------- Hero Banner -----------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ Machine Learning · Real Estate Valuation Engine</div>
        <h1 class="hero-title">🏡 Smart Home Price Estimator & Insights</h1>
        <p class="hero-subtitle">
            Get instant statistical appraisals, explainable value drivers, affordability breakdowns, and multi-scenario comparisons.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- Top Controls & Presets -----------------
col_presets, col_curr = st.columns([3.2, 1], gap="medium")

with col_presets:
    st.markdown('<div class="preset-label">⚡ Quick Presets (Click to load):</div>', unsafe_allow_html=True)
    p_cols = st.columns(len(PRESETS) + 1)
    for col, name in zip(p_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    p_cols[-1].button("🔄 Reset", on_click=reset_defaults, use_container_width=True, help="Reset all inputs to baseline")

with col_curr:
    st.markdown('<div class="preset-label">💱 Display Currency:</div>', unsafe_allow_html=True)
    st.selectbox(
        "Currency",
        options=list(CURRENCIES.keys()),
        key="currency_choice",
        label_visibility="collapsed",
    )

# ----------------- Main Navigation Tabs -----------------
tab_estimate, tab_compare, tab_diagnostics, tab_data = st.tabs([
    "🏡 Valuation & Breakdown",
    "⚖️ Compare Scenarios",
    "📈 Model Diagnostics",
    "📊 Market Data Explorer",
])

# =====================================================================
# TAB 1: VALUATION & BREAKDOWN
# =====================================================================
with tab_estimate:
    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        # Panel 1: Space & Rooms
        st.markdown(
            '<div class="panel-header">📐 Living Space & Layout</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.slider(
                "Living Area (sq ft)",
                min_value=500,
                max_value=4500,
                step=50,
                key="area",
                help="Total finished indoor living space in square feet.",
            )
            c_bed, c_bath, c_gar = st.columns(3)
            with c_bed:
                st.radio("Bedrooms", options=[1, 2, 3, 4, 5, 6], horizontal=True, key="bedrooms")
            with c_bath:
                st.radio("Bathrooms", options=[1, 2, 3, 4, 5], horizontal=True, key="bathrooms")
            with c_gar:
                st.radio(
                    "Garage Bays",
                    options=[0, 1, 2, 3],
                    horizontal=True,
                    key="garage",
                    help="Vehicle capacity in private garage. 0 means street/uncovered parking.",
                )

        # Panel 2: Location & Age
        st.markdown(
            '<div class="panel-header">📍 Location & Structure</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.slider(
                "Property Age (years)",
                min_value=0,
                max_value=50,
                step=1,
                key="age",
                help="0 indicates new construction. Older homes typically have depreciation adjustments.",
            )
            st.slider(
                "Distance to City Centre (km)",
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                key="distance",
                help="Proximity to downtown commercial hub. Proximity usually carries premium valuation.",
            )

        # Panel 3: Interactive Mortgage Estimator
        st.markdown(
            '<div class="panel-header">💳 Monthly Mortgage Estimator</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                down_pct = st.slider("Down Payment (%)", 0, 50, 20, step=5) / 100.0
            with mc2:
                loan_yrs = st.selectbox("Loan Term", [30, 15], index=0)
            with mc3:
                int_rate = st.slider("Interest Rate (%)", 3.0, 10.0, 6.5, step=0.1) / 100.0

    # Calculate Current Valuation
    s = st.session_state
    inputs = dict(
        area_sqft=s.area,
        bedrooms=s.bedrooms,
        bathrooms=s.bathrooms,
        age_years=s.age,
        distance_km=s.distance,
        garage=s.garage,
    )
    price = predict_price(model, **inputs)
    mae = metrics["MAE"]
    low, high = max(price - mae, 0), price + mae
    diff_pct = (price - avg_price) / avg_price * 100
    direction = "above" if diff_pct >= 0 else "below"
    persona = get_property_persona(s.area, s.bedrooms, s.bathrooms, price, s.distance)
    price_per_sqft = price / s.area

    # Mortgage calculation
    mortgage = calculate_mortgage(
        price=price,
        down_payment_pct=down_pct,
        interest_rate=int_rate,
        loan_years=loan_yrs,
    )

    with right_col:
        # Result Hero Card
        st.markdown(
            f"""
            <div class="valuation-card">
                <div class="persona-tag">{persona}</div>
                <div class="valuation-label">Estimated Market Valuation</div>
                <div class="valuation-price">{fmt_curr(price)}</div>
                <div class="valuation-range">
                    Range: {fmt_curr(low)} – {fmt_curr(high)} (±{fmt_curr(mae)})
                </div>
                <div class="metrics-ribbon">
                    <span>📊 <b>{abs(diff_pct):.1f}%</b> {direction} community avg</span>
                    <span>📐 <b>{fmt_curr(price_per_sqft)}</b> / sq ft</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Specs Bar
        st.markdown(
            f"""
            <div class="specs-bar">
                <span class="spec-chip">📐 {s.area:,} sq ft</span>
                <span class="spec-chip">🛏️ {s.bedrooms} Beds</span>
                <span class="spec-chip">🛁 {s.bathrooms} Baths</span>
                <span class="spec-chip">🚗 {s.garage} Garage</span>
                <span class="spec-chip">⏳ {s.age} yrs old</span>
                <span class="spec-chip">📍 {s.distance:.1f} km to center</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Save to Compare Button
        def save_current_property():
            curr_props = st.session_state["saved_properties"]
            item = {
                "Persona": persona,
                "Price": price,
                "Area": s.area,
                "Bedrooms": s.bedrooms,
                "Bathrooms": s.bathrooms,
                "Garage": s.garage,
                "Age": s.age,
                "Distance": s.distance,
                "Monthly": mortgage["total_monthly"],
            }
            if len(curr_props) >= 4:
                curr_props.pop(0)
            curr_props.append(item)
            st.session_state["saved_properties"] = curr_props
            st.toast("Property configuration saved to Compare tab!", icon="✅")

        st.button("📌 Save this Property to Compare", on_click=save_current_property, use_container_width=True)

        # Why this price? Value Drivers
        st.markdown(
            f"""
            <div class="panel-header" style="margin-top: 1.2rem;">
                🔍 Why this price? (Value Drivers vs Average Home)
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels = {
            "area_sqft": "Living Area",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "age_years": "Property Age",
            "distance_km": "Distance to Center",
            "garage": "Garage Capacity",
        }
        contrib = {f: model.coef_[i] * (inputs[f] - avg_home[f]) for i, f in enumerate(FEATURES)}
        biggest = max(abs(v) for v in contrib.values()) or 1.0

        rows_html = '<div class="driver-card">'
        for f, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            width = max(min(abs(v) / biggest * 100, 100), 3)
            color = "#10b981" if v >= 0 else "#ef4444"
            sign = "+" if v >= 0 else "-"
            cls = "up" if v >= 0 else "down"
            rows_html += f"""
            <div class="driver-row">
                <span class="driver-name">{labels[f]}</span>
                <div class="driver-track">
                    <div class="driver-bar" style="width:{width:.0f}%; background:{color};"></div>
                </div>
                <span class="driver-amt {cls}">{sign}{fmt_curr(abs(v))}</span>
            </div>
            """
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)

        # Affordability Breakdown Card
        st.markdown(
            f"""
            <div class="mortgage-box">
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #6366f1;">
                    Estimated Monthly Ownership Cost
                </div>
                <div class="mortgage-total">{fmt_curr(mortgage['total_monthly'])} <span style="font-size:0.9rem; font-weight:500; color:#6b7280;">/ month</span></div>
                <div class="mortgage-row">
                    <span>Principal & Interest ({loan_yrs}yr @ {int_rate*100:.1f}%)</span>
                    <b>{fmt_curr(mortgage['monthly_principal_interest'])}</b>
                </div>
                <div class="mortgage-row">
                    <span>Property Taxes (est. 1.2%/yr)</span>
                    <b>{fmt_curr(mortgage['monthly_tax'])}</b>
                </div>
                <div class="mortgage-row">
                    <span>Home Insurance (est.)</span>
                    <b>{fmt_curr(mortgage['monthly_insurance'])}</b>
                </div>
                <div class="mortgage-row" style="border-top:1px dashed #cbd5e1; margin-top:0.4rem; padding-top:0.4rem;">
                    <span>Down Payment Required ({down_pct*100:.0f}%)</span>
                    <b>{fmt_curr(mortgage['down_payment'])}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
# TAB 2: COMPARE SCENARIOS
# =====================================================================
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Property Comparison")
    st.caption("Compare multiple house configurations you saved from the Valuation tab.")

    saved = st.session_state["saved_properties"]

    if not saved:
        st.info("No properties saved yet! Go to the **Valuation & Breakdown** tab and click **'Save this Property to Compare'**, or load standard presets below.")
        if st.button("📥 Load All Presets into Comparison"):
            st.session_state["saved_properties"] = [
                {
                    "Persona": name,
                    "Price": predict_price(model, area_sqft=p["area"], bedrooms=p["bedrooms"],
                                           bathrooms=p["bathrooms"], age_years=p["age"],
                                           distance_km=p["distance"], garage=p["garage"]),
                    "Area": p["area"],
                    "Bedrooms": p["bedrooms"],
                    "Bathrooms": p["bathrooms"],
                    "Garage": p["garage"],
                    "Age": p["age"],
                    "Distance": p["distance"],
                    "Monthly": calculate_mortgage(
                        predict_price(model, area_sqft=p["area"], bedrooms=p["bedrooms"],
                                      bathrooms=p["bathrooms"], age_years=p["age"],
                                      distance_km=p["distance"], garage=p["garage"])
                    )["total_monthly"],
                }
                for name, p in PRESETS.items()
            ]
            st.rerun()
    else:
        # Comparison Cards
        cols = st.columns(len(saved))
        for idx, (col, item) in enumerate(zip(cols, saved)):
            with col:
                st.markdown(
                    f"""
                    <div class="compare-card">
                        <div style="font-weight:700; color:#374151; font-size:1rem; margin-bottom:0.3rem;">
                            {item['Persona']}
                        </div>
                        <div class="compare-price">{fmt_curr(item['Price'])}</div>
                        <div style="font-size:0.86rem; color:#6b7280; margin-bottom:0.75rem;">
                            {fmt_curr(item['Price'] / item['Area'])} / sq ft
                        </div>
                        <hr style="border:none; border-top:1px solid #f1f5f9; margin:0.6rem 0;" />
                        <div style="text-align:left; font-size:0.85rem; line-height:1.7;">
                            <div>📐 <b>Area:</b> {item['Area']:,} sq ft</div>
                            <div>🛏️ <b>Layout:</b> {item['Bedrooms']} Bed / {item['Bathrooms']} Bath</div>
                            <div>🚗 <b>Garage:</b> {item['Garage']} space(s)</div>
                            <div>⏳ <b>Age:</b> {item['Age']} years</div>
                            <div>📍 <b>Distance:</b> {item['Distance']} km</div>
                            <div style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px solid #f1f5f9; color:#4338ca; font-weight:700;">
                                💳 Est. {fmt_curr(item['Monthly'])}/mo
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Saved Properties"):
            st.session_state["saved_properties"] = []
            st.rerun()

# =====================================================================
# TAB 3: MODEL DIAGNOSTICS & ACCURACY
# =====================================================================
with tab_diagnostics:
    st.markdown("### 📈 Model Performance & Statistical Evaluation")
    st.caption("Cross-validation and error diagnostics evaluated on an independent 20% hold-out test set.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Typical Error (MAE)", fmt_curr(metrics["MAE"]),
              help="Mean Absolute Error: average magnitude of appraisal discrepancy.")
    m2.metric("Accuracy Score (R²)", f"{metrics['R2'] * 100:.1f}%",
              help="Proportion of total housing price variance accounted for by the regression model.")
    m3.metric("Rel. Percentage Error (MAPE)", f"{metrics['MAPE']:.1f}%",
              help="Mean Absolute Percentage Error relative to actual market transactions.")
    m4.metric("Hold-out Test Records", f"{len(test):,} homes")

    diag_left, diag_right = st.columns([1.1, 0.9], gap="large")

    with diag_left:
        st.markdown("#### 🎯 Predicted vs. Actual Sale Price")
        st.caption("Points cluster tight to the diagonal line, confirming consistent unbiased estimation.")
        st.scatter_chart(
            test,
            x="Actual price",
            y="Predicted price",
            color="#4f46e5",
            use_container_width=True,
        )

    with diag_right:
        st.markdown("#### ⚖️ Marginal Value Weights (Model Coefficients)")
        st.caption("How much each specific unit adjustment quantitatively impacts estimated price:")

        impact_data = []
        plain = {
            "area_sqft": "Each extra square foot",
            "bedrooms": "Each additional bedroom",
            "bathrooms": "Each additional bathroom",
            "garage": "Each additional garage bay",
            "age_years": "Each year of property age",
            "distance_km": "Each km further from city center",
        }
        for i, f in enumerate(FEATURES):
            c = model.coef_[i]
            direction_str = "Appreciation (+)" if c >= 0 else "Depreciation (-)"
            impact_data.append({
                "Property Feature": plain[f],
                "Impact Direction": direction_str,
                "Marginal Value (USD)": f"{'+' if c >= 0 else '-'}${abs(c):,.2f}",
            })
        st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)

# =====================================================================
# TAB 4: MARKET DATA EXPLORER
# =====================================================================
with tab_data:
    st.markdown("### 📊 Market Transactions Dataset")
    st.caption("Explore the underlying training properties with interactive filters.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        min_p, max_p = int(df["price"].min()), int(df["price"].max())
        price_filter = st.slider("Filter by Price Range", min_p, max_p, (min_p, max_p), step=10000)
    with f_col2:
        beds_filter = st.multiselect(
            "Filter by Bedrooms",
            options=sorted(df["bedrooms"].unique().tolist()),
            default=sorted(df["bedrooms"].unique().tolist()),
        )
    with f_col3:
        min_a, max_a = int(df["area_sqft"].min()), int(df["area_sqft"].max())
        area_filter = st.slider("Filter by Area (sq ft)", min_a, max_a, (min_a, max_a), step=100)

    # Filtered View
    filtered_df = df[
        (df["price"] >= price_filter[0]) & (df["price"] <= price_filter[1]) &
        (df["bedrooms"].isin(beds_filter if beds_filter else df["bedrooms"].unique())) &
        (df["area_sqft"] >= area_filter[0]) & (df["area_sqft"] <= area_filter[1])
    ]

    # Quick Stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Filtered Homes", f"{len(filtered_df):,}")
    s2.metric("Median Price", fmt_curr(float(filtered_df["price"].median()) if len(filtered_df) else 0))
    s3.metric("Avg Price / sqft", fmt_curr(float((filtered_df["price"] / filtered_df["area_sqft"]).mean()) if len(filtered_df) else 0))
    s4.metric("Avg Distance", f"{filtered_df['distance_km'].mean():.1f} km" if len(filtered_df) else "N/A")

    st.dataframe(filtered_df, use_container_width=True, height=360)

    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        st.download_button(
            "📥 Download Filtered CSV",
            filtered_df.to_csv(index=False),
            file_name="houses_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.caption("⚠️ Note: This dataset is synthetic and calibrated for instructional real-estate regression modeling.")
