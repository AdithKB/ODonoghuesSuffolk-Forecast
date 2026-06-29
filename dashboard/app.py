"""
O'Donoghues Suffolk Street — Demand Forecast Dashboard
Operational decision-support tool for kitchen and bar management.
"""

import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
from src.model import (
    BaselinePredictor, assign_shift_labels,
    label_from_thresholds, SAFE_FOR_NEXT_DAY, TARGETS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="O'Donoghues | Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens + CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #000000;
    --surface: #0A0A0A;
    --surface-hover: #111111;
    --border: #1F1F1F;
    --border-dim: #141414;
    --text-pri: #FAFAFA;
    --text-sec: #A1A1AA;
    --text-ter: #52525B;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', -apple-system, sans-serif;

    --c-quiet: #10B981;
    --c-normal: #3B82F6;
    --c-busy: #F59E0B;
    --c-slammed: #EF4444;
    
    --bg-quiet: rgba(16, 185, 129, 0.1);
    --bg-normal: rgba(59, 130, 246, 0.1);
    --bg-busy: rgba(245, 158, 11, 0.1);
    --bg-slammed: rgba(239, 68, 68, 0.1);
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text-pri) !important;
}

[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stHeader"] { background-color: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stToolbar"] { position: relative !important; }
[data-testid="stToolbar"]::after {
    content: "O'Donoghues";
    position: absolute; left: 50%; top: 50%;
    transform: translate(-50%, -50%);
    color: #FAFAFA; font-family: 'Inter', sans-serif;
    font-size: 0.875rem; font-weight: 600; letter-spacing: 0.04em;
    pointer-events: none;
}


.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

hr { border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; border-bottom: none !important; }

/* ── Streamlit Overrides ── */
[data-testid="stDateInput"] input { background-color: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text-pri) !important; font-family: var(--sans) !important; border-radius: 4px; }
[data-baseweb="input"] { background-color: transparent !important; border-color: transparent !important; }
[data-testid="stBaseButton-secondary"] { background-color: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text-pri) !important; font-family: var(--sans) !important; border-radius: 4px !important; transition: all 0.1s ease !important; font-weight: 500 !important; }
[data-testid="stBaseButton-secondary"]:hover { background-color: var(--surface-hover) !important; border-color: var(--text-ter) !important; color: white !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { display: none !important; }
div[data-baseweb="slider"] div { background-color: var(--border) !important; }
div[data-baseweb="slider"] div[role="slider"] { background-color: var(--text-pri) !important; border: none !important; }
[data-testid="stCheckbox"] label span { color: var(--text-pri) !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; background: var(--bg) !important; border-radius: 6px !important; box-shadow: none !important; }
[data-testid="stExpander"] summary { font-weight: 500 !important; color: var(--text-sec) !important; background: var(--bg) !important; padding: 0.5rem 0.75rem !important;}
[data-testid="stExpander"] summary:hover { color: var(--text-pri) !important; }
[data-testid="stWidgetLabel"] p { font-size: 0.75rem !important; color: var(--text-sec) !important; font-weight: 500 !important; }

/* ── Forecast Drivers List ── */
.driver-row { display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; border-bottom: 1px solid var(--border-dim); }
.driver-row:last-child { border-bottom: none; }
.driver-label { flex: 1 1 0; font-size: 0.8rem; color: var(--text-sec); min-width: 0; }
.driver-bar-wrap { flex: 0 0 80px; height: 3px; background: var(--surface); border-radius: 2px; overflow: hidden; }
.driver-bar { height: 100%; background: #52525B; border-radius: 2px; }
.driver-pct { flex: 0 0 3.5rem; text-align: right; font-size: 0.75rem; color: var(--text-pri); font-family: var(--mono); }

/* ── Custom HTML Table ── */
.custom-table { width: 100%; border-collapse: collapse; font-family: var(--sans); font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 1rem; }
.custom-table th { text-align: right; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); color: var(--text-sec); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.65rem;}
.custom-table th:first-child { text-align: left; padding-left: 0.5rem; }
.custom-table th:nth-child(2) { text-align: left; }
.custom-table td { text-align: right; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-dim); color: var(--text-pri); font-family: var(--mono); font-weight: 500;}
.custom-table td:first-child { text-align: left; font-family: var(--mono); padding-left: 0.5rem; }
.custom-table td:nth-child(2) { font-family: var(--sans); color: var(--text-ter); text-align: left; }
.custom-table tr:last-child td { border-bottom: none; }
.custom-table tr:hover { background-color: var(--surface-hover); }

/* ── UI Components ── */
.hero-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-bottom: 1.5rem;
}
.hero-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-sec); margin-bottom: 0.5rem;}
.hero-date { font-size: 2.25rem; font-weight: 600; color: var(--text-pri); letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.5rem;}
.hero-meta { font-size: 0.85rem; color: var(--text-ter); }

.pressure-banner {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.pressure-banner.quiet   { border-left-color: var(--c-quiet);   }
.pressure-banner.normal  { border-left-color: var(--c-normal);  }
.pressure-banner.busy    { border-left-color: var(--c-busy);    }
.pressure-banner.slammed { border-left-color: var(--c-slammed); }
.pb-label { font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
.pressure-banner.quiet .pb-label   { color: var(--c-quiet);   }
.pressure-banner.normal .pb-label  { color: var(--c-normal);  }
.pressure-banner.busy .pb-label    { color: var(--c-busy);    }
.pressure-banner.slammed .pb-label { color: var(--c-slammed); }
.pb-desc { font-size: 0.85rem; color: var(--text-sec); }

.shift-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    height: 100%;
}
.sc-header { display: flex; justify-content: space-between; align-items: flex-start; }
.sc-title { font-weight: 600; font-size: 0.9rem; color: var(--text-pri); letter-spacing: 0.02em; }
.sc-subtitle { font-size: 0.7rem; color: var(--text-ter); margin-top: 0.2rem; }

.sc-badge { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.2rem 0.5rem; border-radius: 4px; }
.sc-badge.quiet   { color: var(--c-quiet);   background: var(--bg-quiet);   }
.sc-badge.normal  { color: var(--c-normal);  background: var(--bg-normal);  }
.sc-badge.busy    { color: var(--c-busy);    background: var(--bg-busy);    }
.sc-badge.slammed { color: var(--c-slammed); background: var(--bg-slammed); }

.sc-metrics { display: flex; gap: 1.5rem; }
.sc-metric { display: flex; flex-direction: column; gap: 0.25rem; }
.sc-metric-label { font-size: 0.65rem; color: var(--text-ter); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;}
.sc-metric-value { font-family: var(--mono); font-size: 1.5rem; font-weight: 500; color: var(--text-pri); line-height: 1; }

.sc-footer { padding-top: 1rem; border-top: 1px solid var(--border-dim); font-size: 0.75rem; color: var(--text-sec); line-height: 1.4; }
.sc-footer strong { color: var(--text-pri); font-weight: 500; }

.panel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
}
.panel-title { font-size: 0.85rem; font-weight: 600; color: var(--text-pri); margin-bottom: 1rem; letter-spacing: 0.02em; border-bottom: 1px solid var(--border-dim); padding-bottom: 0.75rem; }
.sig-row { display: flex; align-items: center; padding: 0.35rem 0; font-size: 0.8rem; color: var(--text-pri); border-bottom: 1px solid var(--border-dim); }
.sig-row:last-child { border-bottom: none; }
.sig-pip { width: 6px; height: 6px; border-radius: 50%; margin-right: 0.75rem; flex-shrink: 0; }
.sig-pip.high    { background: var(--c-slammed); box-shadow: 0 0 8px rgba(239, 68, 68, 0.4); }
.sig-pip.medium  { background: var(--c-busy);    box-shadow: 0 0 8px rgba(245, 158, 11, 0.4); }
.sig-pip.low     { background: var(--text-ter);  }
.sig-group-hdr { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-ter); margin: 0.75rem 0 0.25rem 0; }
.sig-group-hdr:first-child { margin-top: 0; }

/* ── Responsive / Mobile ── */
.table-responsive { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; padding-top: 4.5rem !important; }
    .hero-block { padding: 1rem; margin-bottom: 1rem; gap: 0.5rem; }
    .hero-date { font-size: 1.75rem; line-height: 1.1; }
    .hero-meta { font-size: 0.75rem; line-height: 1.4; }
    
    .pressure-banner { flex-direction: column; align-items: flex-start; gap: 0.5rem; padding: 0.75rem 1rem; margin-bottom: 1.5rem; }
    
    .shift-card { padding: 1rem; gap: 1rem; }
    .sc-metrics { gap: 1rem; justify-content: space-between; }
    .sc-metric-value { font-size: 1.25rem; }
    
    .panel-card { padding: 1rem; margin-bottom: 1rem; }
    
    .custom-table th, .custom-table td { padding: 0.5rem; font-size: 0.75rem; white-space: nowrap; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHIFT_PREP = {
    "quiet":   ("Normal prep",   "No extra staffing needed."),
    "normal":  ("Standard prep", "Follow usual levels."),
    "busy":    ("Increase prep", "Higher than usual — check stock."),
    "slammed": ("Maximum prep",  "All hands on deck."),
}
SHIFT_HOURS = {
    "Lunch":    {"range": (12, 14), "subtitle": "12:00 – 15:00 · Food service"},
    "Evening":  {"range": (17, 20), "subtitle": "17:00 – 21:00 · Dinner service"},
    "Late bar": {"range": (21, 25), "subtitle": "21:00 – close · Bar only"},
}

HUMAN_LABELS = {
    "st_patricks_week_flag":       "St. Patrick's Festival week",
    "major_sports_event_flag":     "Major sports event",
    "aviva_event_flag":            "Aviva Stadium event",
    "croke_park_event_flag":       "Croke Park event",
    "nearby_venue_event_flag":     "Nearby venue event",
    "event_impact_score":          "Event impact score",
    "bloomsday_flag":              "Bloomsday",
    "bloomsday_week_flag":         "Bloomsday week",
    "summer_tourism_flag":         "Summer tourism season",
    "christmas_market_flag":       "Christmas markets",
    "new_years_eve_flag":          "New Year's Eve",
    "college_term_flag":           "TCD term time",
    "city_event_flag":             "City event",
    "special_event_flag":          "Special / private event",
    "bank_holiday_flag":           "Bank holiday",
    "school_holiday_flag":         "School holidays",
    "cruise_ship_flag":            "Cruise ship in port",
    "payday_period_flag":          "Payday period",
    "failte_event_count":          "Failte Ireland events near pub",
    "is_friday_saturday":          "Friday / Saturday",
    "is_weekend":                  "Weekend",
    "is_live_music_window":        "Live music tonight",
    "airport_arrivals_zscore":     "Airport arrivals vs average",
    "orders_count_lag_168h":       "Same slot last week (orders)",
    "food_tickets_count_lag_168h": "Same slot last week (food)",
    "event_intensity":             "Combined event pressure",
    "tourism_pressure":            "Tourism signal",
    "weekend_x_music":             "Weekend + live music",
    "orders_count_roll_mean_24":   "24h rolling average",
    "suffolk_footfall_lag_24h":    "Suffolk St footfall (yesterday)",
    "suffolk_footfall_lag_168h":   "Suffolk St footfall (last week)",
    "suffolk_footfall_roll_24h":   "Suffolk St footfall (24h avg)",
}

# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_features() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/features.parquet")
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    return df.sort_values("timestamp_hour").reset_index(drop=True)

@st.cache_resource(show_spinner="Loading models...")
def load_models() -> dict:
    m = {}
    for target in TARGETS:
        xm = xgb.XGBRegressor()
        xm.load_model(f"models/xgb_{target}.json")
        bl = joblib.load(f"models/baseline_{target}.pkl")
        m[target] = {"xgb": xm, "baseline": bl}
    return m

@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    fi = {}
    for target in TARGETS:
        p = Path(f"models/feature_importance_{target}.csv")
        if p.exists():
            fi[target] = pd.read_csv(p)
    return fi

# ---------------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------------
def forecast_for_date(df, models, date, overrides):
    day = df[df["timestamp_hour"].dt.date == date.date()].copy()
    if day.empty:
        return pd.DataFrame()
    for col, val in (overrides or {}).items():
        if col in day.columns:
            day[col] = val
    feat = [c for c in SAFE_FOR_NEXT_DAY if c in day.columns]
    X = day[feat].fillna(0)
    result = day[["timestamp_hour"]].copy()
    result["hour"] = day["timestamp_hour"].dt.hour
    for t in TARGETS:
        result[f"{t}_xgb"]      = np.maximum(models[t]["xgb"].predict(X), 0).round(1)
        result[f"{t}_baseline"] = np.maximum(models[t]["baseline"].predict(day, t), 0).round(1)
    for col in overrides or {}:
        if col in day.columns:
            result[col] = day[col].values
    return result

def shift_kpis(forecast, history):
    out = {}
    for name, info in SHIFT_HOURS.items():
        h0, h1 = info["range"]
        if h1 > 24:
            mask = (forecast["hour"] >= h0) | (forecast["hour"] <= h1 - 24)
        else:
            mask = forecast["hour"].between(h0, h1)
        s = forecast[mask]
        if s.empty:
            continue
        orders  = s["orders_count_xgb"].sum()
        tickets = s["food_tickets_count_xgb"].sum()
        peak_h  = int(s.loc[s["orders_count_xgb"].idxmax(), "hour"])
        if not history.empty:
            hist_s = history[history["hour"].between(h0, min(h1, 23))]
            if not hist_s.empty:
                day_totals = hist_s.groupby(hist_s["timestamp_hour"].dt.date)["orders_count"].sum()
                q = np.percentile(day_totals.values, [25, 65, 88]) if len(day_totals) > 3 else [20, 40, 70]
            else:
                q = [20, 40, 70]
        else:
            q = [20, 40, 70]
        thr = {"quiet": float(q[0]), "normal": float(q[1]), "busy": float(q[2])}
        label = label_from_thresholds(float(orders), thr)
        out[name] = {
            "orders": int(orders), "tickets": int(tickets),
            "peak_hour": peak_h, "label": label,
            "subtitle": info["subtitle"],
        }
    return out

# ---------------------------------------------------------------------------
# UI Components Renderers
# ---------------------------------------------------------------------------
def render_header(forecast_date):
    st.markdown(
        f"""
        <div class="hero-block">
            <div class="hero-title">O'Donoghues Forecast</div>
            <div class="hero-date">{forecast_date.strftime('%A, %d %B %Y')}</div>
            <div class="hero-meta">15 Suffolk St, Dublin &middot; Kitchen 09:00–21:00 &middot; Bar until 02:00 Fri/Sat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_pressure_banner(kpis):
    labels = [v["label"] for v in kpis.values()]
    if "slammed" in labels:
        banner = ("slammed", "High pressure day",  "Maximum prep. Full staffing across all shifts.")
    elif labels.count("busy") >= 2:
        banner = ("busy",    "Busy day",            "Increase prep. Review staffing for all shifts.")
    elif "busy" in labels:
        banner = ("busy",    "Moderate pressure",   "One busy shift expected. Targeted prep recommended.")
    else:
        banner = ("quiet",   "Normal trading day",  "No elevated pressure expected.")

    st.markdown(
        f'<div class="pressure-banner {banner[0]}">'
        f'  <span class="pb-label">{banner[1]}</span>'
        f'  <span class="pb-desc">{banner[2]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_shift_card(name: str, info: dict):
    label = info["label"]
    prep_title, prep_desc = SHIFT_PREP[label]
    is_food = name != "Late bar"

    food_html = (
        f'<div class="sc-metric">'
        f'  <span class="sc-metric-label">Food</span>'
        f'  <span class="sc-metric-value">{info["tickets"]}</span>'
        f'</div>'
    ) if is_food else (
        f'<div class="sc-metric" style="visibility:hidden;">'
        f'  <span class="sc-metric-label">-</span>'
        f'  <span class="sc-metric-value">-</span>'
        f'</div>'
    )

    html = (
        f'<div class="shift-card">'
        f'  <div class="sc-header">'
        f'    <div>'
        f'      <div class="sc-title">{name}</div>'
        f'      <div class="sc-subtitle">{info["subtitle"]}</div>'
        f'    </div>'
        f'    <div class="sc-badge {label}">{label}</div>'
        f'  </div>'
        f'  <div class="sc-metrics">'
        f'    <div class="sc-metric">'
        f'      <span class="sc-metric-label">Orders</span>'
        f'      <span class="sc-metric-value">{info["orders"]}</span>'
        f'    </div>'
        f'    {food_html}'
        f'    <div class="sc-metric">'
        f'      <span class="sc-metric-label">Peak</span>'
        f'      <span class="sc-metric-value">{info["peak_hour"]:02d}:00</span>'
        f'    </div>'
        f'  </div>'
        f'  <div class="sc-footer">'
        f'    <strong>{prep_title}</strong> &mdash; {prep_desc}'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_hourly_chart(forecast: pd.DataFrame):
    st.markdown('<div class="panel-title" style="margin-top:2rem; margin-bottom:0.5rem; border:none;">Hourly Breakdown</div>', unsafe_allow_html=True)
    
    hours_str = [f"{int(h):02d}:00" for h in forecast["hour"]]
    
    fig = go.Figure()

    shade = [
        ("12:00", "15:00", "rgba(255,255,255,0.015)",  "LUNCH"),
        ("17:00", "21:00", "rgba(255,255,255,0.025)",  "EVENING"),
        ("21:00", "01:00", "rgba(255,255,255,0.015)",  "LATE BAR"),
    ]
    for x0, x1, colour, label in shade:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=colour, layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=8, annotation_font_color="#3F3F46",
            annotation_font_family="Inter, sans-serif",
        )

    fig.add_trace(go.Bar(
        x=hours_str, y=forecast["orders_count_xgb"],
        name="Orders", marker_color="#3B82F6", opacity=0.9,
        marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=hours_str, y=forecast["food_tickets_count_xgb"],
        name="Food Tickets", mode="lines+markers",
        line=dict(color="#D97706", width=2), marker=dict(size=5, color="#D97706", line=dict(width=1, color="#0A0A0A")),
    ))
    
    baseline_vals = forecast["orders_count_baseline"].tolist()
    
    fig.add_trace(go.Scatter(
        x=hours_str, y=baseline_vals,
        name="Same Slot Last Week", mode="lines",
        line=dict(color="#3F3F46", width=1, dash="dash"), opacity=0.5,
        connectgaps=False
    ))
    
    fig.add_vline(
        x="21:00", line_width=1, line_dash="dash", line_color="#3F3F46"
    )
    fig.add_annotation(
        x="21:00", y=1, yref="paper",
        text="Kitchen close", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(size=8, color="#71717A", family="Inter, sans-serif"),
        xshift=4, yshift=-4
    )

    tick_vals = hours_str
    tick_text = [h if int(h.split(":")[0]) % 2 == 0 else "" for h in tick_vals]

    fig.update_layout(
        template="plotly_dark",
        height=340,
        dragmode=False,
        margin=dict(l=0, r=0, t=10, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
            font=dict(size=10, color="#A1A1AA", family="Inter, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=hours_str,
            tickvals=tick_vals, ticktext=tick_text, title=None,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
            gridcolor="#141414", showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=None, gridcolor="#141414", zeroline=False,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_signals_panel(forecast, forecast_date):
    wd = forecast_date.weekday()
    wd_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd_intensity = {0:"low",1:"low",2:"low",3:"low",4:"medium",5:"medium",6:"low"}

    signals = []
    flags = {
        "st_patricks_week_flag":    ("St. Patrick's Festival week",  "high"),
        "new_years_eve_flag":       ("New Year's Eve",               "high"),
        "bloomsday_flag":           ("Bloomsday (16 Jun)",           "high"),
        "aviva_event_flag":         ("Aviva Stadium event",          "high"),
        "croke_park_event_flag":    ("Croke Park event",             "high"),
        "major_sports_event_flag":  ("Major sports event",           "high"),
        "bloomsday_week_flag":      ("Bloomsday festival week",      "medium"),
        "summer_tourism_flag":      ("Summer tourism peak",          "medium"),
        "christmas_market_flag":    ("Christmas markets season",     "medium"),
        "nearby_venue_event_flag":  ("Nearby venue event",           "medium"),
        "city_event_flag":          ("City event",                   "medium"),
        "special_event_flag":       ("Special / private event",      "medium"),
        "bank_holiday_flag":        ("Bank holiday",                 "medium"),
        "cruise_ship_flag":         ("Cruise ship in port",          "medium"),
        "college_term_flag":        ("TCD term time",                "low"),
        "school_holiday_flag":      ("School holidays",              "low"),
        "payday_period_flag":       ("Payday period",                "low"),
    }
    for col, (text, level) in flags.items():
        if col in forecast.columns and forecast[col].astype(float).any():
            signals.append((text, level))

    if "failte_event_count" in forecast.columns:
        n = int(pd.to_numeric(forecast["failte_event_count"], errors="coerce").fillna(0).iloc[0])
        if n > 0:
            level = "high" if n >= 5 else "medium" if n >= 2 else "low"
            signals.append((f"{n} Failte Ireland events within 5km", level))

    signals.append((wd_names[wd], wd_intensity[wd]))

    if "is_live_music_window" in forecast.columns and forecast["is_live_music_window"].astype(float).any():
        time_str = "22:00 – 01:00" if wd in (4, 5) else "21:30 – 23:30"
        signals.append((f"Live music ({time_str})", "high"))

    if "rain_mm" in forecast.columns:
        rain = forecast["rain_mm"].mean()
        if rain > 5:
            signals.append((f"Heavy rain ({rain:.0f}mm) — indoor trade elevated", "medium"))
        elif rain > 1:
            signals.append((f"Light rain ({rain:.0f}mm)", "low"))
        else:
            signals.append(("Dry conditions", "low"))

    if "airport_arrivals_zscore" in forecast.columns:
        z = float(forecast["airport_arrivals_zscore"].iloc[0])
        if z > 1.0:
            signals.append(("Airport arrivals above average", "medium"))
        elif z < -1.0:
            signals.append(("Airport arrivals below average", "low"))

    html_parts = ['<div class="panel-card"><div class="panel-title">Today\'s Signals</div>']
    
    if not signals:
        html_parts.append('<div style="color:var(--text-sec);font-size:0.85rem;">No elevated signals today.</div>')
    else:
        high_sigs   = [(t, l) for t, l in signals if l == "high"]
        medium_sigs = [(t, l) for t, l in signals if l == "medium"]
        low_sigs    = [(t, l) for t, l in signals if l == "low"]

        def render_group(header, items):
            if not items: return ""
            rows = "".join(f'<div class="sig-row"><span class="sig-pip {l}"></span>{t}</div>' for t, l in items)
            return f'<div class="sig-group-hdr">{header}</div>{rows}'

        html_parts.append(render_group("High pressure", high_sigs))
        html_parts.append(render_group("Moderate", medium_sigs))
        html_parts.append(render_group("Context", low_sigs))
    
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def render_feature_importance_panel(fi):
    if "orders_count" not in fi:
        st.markdown('<div class="panel-card"><div class="panel-title">Forecast Drivers</div><div style="color:var(--text-sec);font-size:0.85rem;">No data.</div></div>', unsafe_allow_html=True)
        return

    df = fi["orders_count"].head(7).copy()
    df["label"] = df["feature"].map(HUMAN_LABELS).fillna(df["feature"].str.replace("_", " ").str.title())
    df = df.sort_values("importance_pct", ascending=False)
    max_pct = df["importance_pct"].max() or 1

    rows = ""
    for _, row in df.iterrows():
        bar_w = int(row["importance_pct"] / max_pct * 100)
        rows += (
            f'<div class="driver-row">'
            f'<span class="driver-label">{row["label"]}</span>'
            f'<div class="driver-bar-wrap"><div class="driver-bar" style="width:{bar_w}%"></div></div>'
            f'<span class="driver-pct">{row["importance_pct"]:.1f}%</span>'
            f'</div>'
        )
    st.markdown(f'<div class="panel-card"><div class="panel-title">Forecast Drivers</div>{rows}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df     = load_features()
    models = load_models()
    fi     = load_feature_importance()

    available    = sorted(df["timestamp_hour"].dt.date.unique())
    day_counts   = df.groupby(df["timestamp_hour"].dt.date)["timestamp_hour"].count()
    full_days    = sorted(day_counts[day_counts >= 12].index)

    import datetime
    today = datetime.date.today()
    if today in set(available):
        default_date = today
    else:
        default_date = full_days[-1] if full_days else available[-1]

    # ── Sidebar Controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h3 style='color: var(--text-pri); margin-bottom: 1rem; font-weight: 600; font-size: 1.1rem;'>Settings & Overrides</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Forecast Date</div>", unsafe_allow_html=True)
        sel_date = st.date_input("Forecast Date", value=default_date, min_value=available[0], max_value=available[-1], label_visibility="collapsed")
        forecast_date = pd.Timestamp(sel_date)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Overrides</div>", unsafe_allow_html=True)
        live_music    = st.toggle("Live music",      value=True)
        special_event = st.toggle("Special event",   value=False)
        major_sports  = st.toggle("Sports event",    value=False)
        cruise        = st.toggle("Cruise ship",     value=False)
        st_pats       = st.toggle("St. Patrick's week", value=False)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Weather</div>", unsafe_allow_html=True)
        rain = st.slider("Rain (mm)",        0.0, 20.0, 1.0, 0.5)
        temp = st.slider("Temperature (°C)", 0.0, 25.0, 12.0, 0.5)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Data Sync</div>", unsafe_allow_html=True)
        if st.button("Refresh Live Data", use_container_width=True):
            with st.spinner("Fetching..."):
                subprocess.run([sys.executable, "src/refresh_data.py"], capture_output=True, cwd=Path(__file__).parent.parent)
                st.cache_data.clear()
                st.rerun()

    overrides = {
        "live_music_flag":         int(live_music),
        "is_live_music_window":    int(live_music),
        "special_event_flag":      int(special_event),
        "major_sports_event_flag": int(major_sports),
        "cruise_ship_flag":        int(cruise),
        "st_patricks_week_flag":   int(st_pats),
        "rain_mm": rain, "temp_c": temp,
    }

    forecast = forecast_for_date(df, models, forecast_date, overrides)
    if forecast.empty:
        st.warning("No data for this date.")
        return

    history = df[df["timestamp_hour"].dt.date < forecast_date.date()].copy()
    history["timestamp_hour"] = pd.to_datetime(history["timestamp_hour"])
    kpis = shift_kpis(forecast, history)

    # ── Header & Banner ──────────────────────────────────────────────────────
    render_header(forecast_date)
    render_pressure_banner(kpis)

    # ── Shift cards ──────────────────────────────────────────────────────────
    cols = st.columns(3)
    for col, (name, info) in zip(cols, kpis.items()):
        with col:
            render_shift_card(name, info)

    # ── Chart ────────────────────────────────────────────────────────────────
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    render_hourly_chart(forecast)

    # ── Context Panels ───────────────────────────────────────────────────────
    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        render_signals_panel(forecast, forecast_date)
    with right:
        render_feature_importance_panel(fi)

    # ── Hourly table ─────────────────────────────────────────────────────────
    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    with st.expander("View Full Hourly Data"):
        def _shift_label(h):
            if 12 <= h <= 14: return "Lunch"
            if 17 <= h <= 20: return "Evening"
            if h >= 21 or h <= 1: return "Late bar"
            return ""

        out = forecast[["timestamp_hour", "hour", "orders_count_xgb", "food_tickets_count_xgb", "orders_count_baseline"]].copy()
        out["Shift"]            = out["hour"].apply(_shift_label)
        out["Time"]             = out["timestamp_hour"].dt.strftime("%H:%M")
        out["Orders"]           = out["orders_count_xgb"].round(0).astype(int)
        out["Food"]             = out["food_tickets_count_xgb"].round(0).astype(int)
        out["Baseline"]         = out["orders_count_baseline"].round(0).astype(int)

        table_rows = []
        for _, row in out.iterrows():
            table_rows.append(
                f"<tr>"
                f"<td>{row['Time']}</td>"
                f"<td>{row['Shift']}</td>"
                f"<td>{row['Orders']}</td>"
                f"<td>{row['Food']}</td>"
                f"<td>{row['Baseline']}</td>"
                f"</tr>"
            )
        
        table_html = f"""
        <div class="table-responsive">
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Shift</th>
                    <th>Orders</th>
                    <th>Food</th>
                    <th>Baseline</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        
        csv_out = out[["Time", "Shift", "Orders", "Food", "Baseline"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_out, file_name=f"odonoghues_{forecast_date.strftime('%Y-%m-%d')}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
