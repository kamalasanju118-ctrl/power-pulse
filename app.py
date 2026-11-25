"""
Power Pulse Website (Multi-City, with a home screen of clickable tiles)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Power Pulse", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #dbeafe 0%, #eef2ff 45%, #fce7f3 100%); }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp span { color: #0f172a; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #e0e7ff 100%);
        border-right: 1px solid #c7d2fe;
    }

    div[data-testid="stMetric"] {
        background: #ffffff; border-radius: 16px; padding: 18px 22px;
        box-shadow: 0 6px 18px rgba(30, 64, 175, 0.12); border-left: 6px solid #3b82f6;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff; border-radius: 14px; padding: 16px;
        box-shadow: 0 6px 18px rgba(30, 64, 175, 0.10);
    }
    div[data-testid="stDataFrame"] {
        background: #ffffff; border-radius: 14px; padding: 8px;
        box-shadow: 0 6px 18px rgba(30, 64, 175, 0.10);
    }
    div[data-testid="stAlert"] { border-radius: 14px; }

    .hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #fce7f3 100%);
        border-radius: 20px; padding: 28px 34px; margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(30, 64, 175, 0.30);
    }
    .hero h1 { color: #ffffff !important; margin: 0; font-size: 2.2rem; }
    .hero p { color: #dbeafe !important; margin: 6px 0 0 0; font-size: 1rem; }

    /* big desktop-style tiles on the home screen */
    .st-key-tiles button {
        height: 130px; width: 100%; border-radius: 20px; background: #ffffff;
        border: 2px solid #bfdbfe; box-shadow: 0 8px 20px rgba(30, 64, 175, 0.15);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .st-key-tiles button p { font-size: 1.35rem; font-weight: 700; color: #1e3a8a !important; }
    .st-key-tiles button:hover {
        transform: translateY(-4px); border-color: #3b82f6;
        box-shadow: 0 14px 28px rgba(30, 64, 175, 0.28);
    }
    .st-key-menu button { width: 100%; text-align: left; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("power_pulse_model.pkl"), joblib.load("feature_cols.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("power_pulse_features.csv", parse_dates=["timestamp"])


model, feature_cols = load_model()
all_df = load_data()

PAGES = [
    ("predict", "⚡", "Live Prediction", "Risk of a power cut in the next hour"),
    ("charts", "📈", "Charts", "Voltage, frequency, load and risk trends"),
    ("cities", "🏙️", "City Comparison", "Compare outages across cities"),
    ("history", "📋", "Outage History", "List of past power cut events"),
    ("about", "ℹ️", "About the Model", "How the prediction works"),
]
PAGE_TITLES = {key: f"{icon} {title}" for key, icon, title, _ in PAGES}

if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page):
    st.session_state.page = page


page = st.session_state.page

# ---------------- Sidebar: menu, city, time ----------------
st.sidebar.header("Menu")
with st.sidebar.container(key="menu"):
    st.button("🏠 Home", key="menu_home", on_click=go, args=("home",))
    for key, icon, title, _ in PAGES:
        st.button(f"{icon} {title}", key=f"menu_{key}", on_click=go, args=(key,))

st.sidebar.divider()
city = st.sidebar.selectbox("City", sorted(all_df["city"].unique()), key="city")
df = all_df[all_df["city"] == city].reset_index(drop=True)

if page in ("predict", "charts"):
    row_idx = st.sidebar.slider(
        "Timestamp index (simulates a live sensor feed)",
        0, len(df) - 1, len(df) - 500, key=f"slider_{city}",
    )
else:
    row_idx = len(df) - 500

current_row = df.iloc[row_idx]

# ---------------- Header ----------------
subtitle = (
    "Choose an option below to open it"
    if page == "home"
    else f"{PAGE_TITLES.get(page, '')} - {city}"
)
st.markdown(
    f"""
    <div class="hero">
        <h1>⚡ Power Pulse</h1>
        <p>{subtitle}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page != "home":
    st.button("← Back to Home", key="back_home", on_click=go, args=("home",))

# ---------------- Pages ----------------
if page == "home":
    st.write("Predicts the probability of a power cut in the next 1 hour from voltage, frequency, load and time-of-day patterns. Click a tile to open it.")
    with st.container(key="tiles"):
        for start in range(0, len(PAGES), 3):
            cols = st.columns(3)
            for col, (key, icon, title, desc) in zip(cols, PAGES[start:start + 3]):
                with col:
                    st.button(f"{icon}  {title}", key=f"tile_{key}", on_click=go, args=(key,))
                    st.caption(desc)

elif page == "predict":
    X_current = current_row[feature_cols].to_frame().T.astype(float)
    risk_pct = round(model.predict_proba(X_current)[0][1] * 100, 1)

    st.write(f"Selected time: {current_row['timestamp']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Voltage", f"{current_row['voltage']} V")
    c2.metric("Frequency", f"{current_row['frequency']} Hz")
    c3.metric("Load", f"{current_row['load_kw']} kW")

    st.subheader(f"Power cut risk in {city} (next 1 hour)")
    if risk_pct >= 70:
        st.error(f"HIGH RISK - {risk_pct}% chance of a power cut in the next hour")
    elif risk_pct >= 40:
        st.warning(f"MODERATE RISK - {risk_pct}% chance of a power cut")
    else:
        st.success(f"LOW RISK - {risk_pct}% chance of a power cut")
    st.progress(min(int(risk_pct), 100))
    st.caption("Use the slider in the left sidebar to move through time.")

elif page == "charts":
    window = df.iloc[max(0, row_idx - 30):min(len(df), row_idx + 10)].set_index("timestamp")
    st.write(f"Selected time: {current_row['timestamp']} (use the sidebar slider to move through time)")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Voltage", "Frequency", "Load", "Risk over time", "Outages by hour"])
    with tab1:
        st.line_chart(window["voltage"])
    with tab2:
        st.line_chart(window["frequency"])
    with tab3:
        st.line_chart(window["load_kw"])
    with tab4:
        risk_series = model.predict_proba(window[feature_cols].astype(float))[:, 1] * 100
        st.line_chart(pd.Series(risk_series, index=window.index, name="Risk %"))
    with tab5:
        by_hour = df[df["power_cut"] == 1]["hour"].value_counts().reindex(range(24), fill_value=0)
        by_hour_df = pd.DataFrame({"hour": by_hour.index.astype(str).str.zfill(2), "outage readings": by_hour.values})
        st.bar_chart(by_hour_df, x="hour", y="outage readings")

elif page == "cities":
    st.subheader("Outage readings in the dataset, by city")
    city_counts = all_df[all_df["power_cut"] == 1]["city"].value_counts().reset_index()
    city_counts.columns = ["city", "outage readings"]
    st.bar_chart(city_counts, x="city", y="outage readings", horizontal=True)

    summary = all_df.groupby("city").agg(
        readings=("power_cut", "size"),
        outage_readings=("power_cut", "sum"),
        avg_voltage=("voltage", "mean"),
    ).round(2)
    st.dataframe(summary)

elif page == "history":
    outages = df[df["power_cut"] == 1][["timestamp", "voltage", "current", "frequency"]]
    st.write(f"Outage readings in {city}: {len(outages)}")
    st.dataframe(outages)

elif page == "about":
    st.write(
        "Model: Random Forest classifier. Target: power cut in the next 1 hour "
        "(6 readings ahead at 10-minute intervals). Features: voltage, current, frequency, load, "
        "rolling averages and spreads, change rates, lag values, peak-hour, weekend and monsoon flags, and the city."
    )
    st.write(
        "Data is simulated from typical load-shedding patterns for each city because official "
        "outage logs are not public. Each city has its own outage rate, voltage level and monsoon effect."
    )
