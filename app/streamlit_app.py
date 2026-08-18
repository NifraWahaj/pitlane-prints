"""
streamlit_app.py
----------------
F1 Telemetry Lens — Driver Style Fingerprinting Dashboard

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots
from scipy.signal import sawtooth
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="F1 Telemetry Lens",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────
BG        = "#08080d"
SURFACE   = "#101017"
SURFACE_2 = "#15151e"
BORDER    = "#23232f"
TEXT      = "#e8e8f2"
TEXT_DIM  = "#8b8ba6"
TEXT_MUTE = "#5a5a72"
ACCENT    = "#e10600"

DRIVER_COLORS = {
    "VER": "#3b82f6", "HAM": "#a78bfa", "ALO": "#f87171",
    "LEC": "#f97316", "SAI": "#facc15", "NOR": "#34d399",
}
DRIVER_NAMES = {
    "VER": "Max Verstappen", "HAM": "Lewis Hamilton", "ALO": "Fernando Alonso",
    "LEC": "Charles Leclerc", "SAI": "Carlos Sainz",  "NOR": "Lando Norris",
}
FEATURE_COLS = [
    "brake_duration_ratio", "throttle_smoothness", "full_throttle_ratio",
    "coasting_ratio", "gear_change_freq", "speed_at_throttle_lift",
    "mean_corner_speed", "speed_variance", "throttle_brake_overlap",
]
FEATURE_LABELS = [
    "Brake Duration", "Throttle Smoothness", "Full Throttle", "Coasting",
    "Gear Changes", "Braking Speed", "Corner Speed", "Speed Variance", "Trail Braking",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "Inter"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {BG}; color: {TEXT}; }}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
.block-container {{
    padding-top: 2.25rem !important;
    padding-bottom: 4rem !important;
    max-width: 1320px;
}}

/* ── Masthead ── */
.masthead {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0.35rem;
}}
.mast-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.015em;
    line-height: 0.95;
    color: {TEXT};
    margin: 0;
}}
.mast-title span {{ color: {ACCENT}; }}
.mast-tag {{
    font-size: 0.86rem;
    color: {TEXT_DIM};
    margin-top: 9px;
    max-width: 640px;
    line-height: 1.6;
}}
.mast-meta {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {TEXT_MUTE};
    text-align: right;
    line-height: 2;
    white-space: nowrap;
}}
.mast-meta b {{ color: {TEXT_DIM}; font-weight: 600; }}

/* ── Metric cards ── */
.metric {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    height: 100%;
}}
.metric-label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {TEXT_MUTE};
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.metric-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.01em;
}}
.metric-note {{
    font-size: 0.7rem;
    color: {TEXT_MUTE};
    margin-top: 5px;
    line-height: 1.4;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.6rem;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {TEXT_MUTE};
    background: transparent;
    padding: 0.75rem 1.4rem;
    border-radius: 0;
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color: {TEXT} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; height: 2px; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

/* ── Panel ── */
.panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
}}
.panel-flush {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.4rem 0.6rem 0.1rem;
}}

/* ── Section label ── */
.slabel {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 0.7rem;
}}
.intro {{
    font-size: 0.85rem;
    color: {TEXT_DIM};
    line-height: 1.65;
    max-width: 900px;
    margin-bottom: 1.2rem;
}}
.intro b {{ color: {TEXT}; font-weight: 600; }}
.note {{
    font-size: 0.78rem;
    color: {TEXT_MUTE};
    line-height: 1.6;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid {BORDER};
}}
.note b {{ color: {TEXT_DIM}; font-weight: 600; }}

/* ── Verdict card ── */
.vcard {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
}}
.vcard-kicker {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.64rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {TEXT_MUTE};
}}
.vcard-code {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.1rem;
    font-weight: 800;
    line-height: 1;
    margin: 6px 0 3px;
}}
.vcard-name {{ font-size: 0.82rem; color: {TEXT_DIM}; }}
.verdict {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid {BORDER};
}}
.verdict-sub {{ font-size: 0.75rem; color: {TEXT_MUTE}; margin-top: 4px; letter-spacing: 0; }}

/* ── Probability bars ── */
.pbar-row {{ margin-bottom: 0.7rem; }}
.pbar-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    margin-bottom: 4px;
}}
.pbar-track {{ background: {SURFACE_2}; border-radius: 3px; height: 6px; overflow: hidden; }}
.pbar-fill {{ height: 6px; border-radius: 3px; }}

/* ── Key/value table ── */
.kv {{ width: 100%; border-collapse: collapse; font-size: 0.79rem; }}
.kv td {{ padding: 6px 2px; border-bottom: 1px solid {BORDER}; }}
.kv td:first-child {{ color: {TEXT_MUTE}; }}
.kv td:last-child {{
    text-align: right;
    font-weight: 600;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
}}

/* ── Empty state ── */
.empty {{
    background: {SURFACE};
    border: 1px dashed {BORDER};
    border-radius: 10px;
    padding: 1.6rem;
    font-size: 0.83rem;
    color: {TEXT_MUTE};
    line-height: 1.6;
}}

/* ── Streamlit widget polish ── */
.stButton > button {{
    border-radius: 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid {BORDER};
}}
[data-testid="stExpander"] details {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
[data-testid="stExpander"] summary {{ font-size: 0.84rem; }}
div[role="radiogroup"] label {{ font-size: 0.83rem; }}
.stSelectbox label, .stMultiSelect label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {TEXT_MUTE} !important;
}}
hr {{ border-color: {BORDER}; margin: 1.8rem 0; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading telemetry and models...")
def load_all_data():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    tag           = "all_races"
    features_dir  = os.path.join(ROOT, config["data"]["features_dir"])
    processed_dir = os.path.join(ROOT, config["data"]["processed_dir"])
    models_dir    = os.path.join(ROOT, "outputs", "models")

    def opt_csv(name):
        path = os.path.join(features_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    bundle = dict(
        config      = config,
        features_df = pd.read_csv(os.path.join(features_dir, f"{tag}_features.csv")),
        umap_df     = pd.read_csv(os.path.join(features_dir, f"{tag}_umap_coords.csv")),
        history_df  = pd.read_csv(os.path.join(features_dir, f"{tag}_cnn_history.csv")),
        fi_df       = pd.read_csv(os.path.join(features_dir, f"{tag}_feature_importance.csv")),
        xgb_le      = joblib.load(os.path.join(models_dir, f"{tag}_label_encoder.pkl")),
        oof_df      = opt_csv(f"{tag}_oof_predictions.csv"),
        embed_df    = opt_csv(f"{tag}_embeddings.csv"),
        contrast_df = opt_csv(f"{tag}_contrastive_embeddings.csv"),
    )

    # Raw telemetry keyed by race tag then driver. Loaded for every configured
    # race so the blind test can show the exact lap it scored, and the explorer
    # can browse any race.
    raw = {}
    for race_cfg in config["races"]:
        rtag = f"{race_cfg['year']}_{race_cfg['race'].lower()}"
        raw[rtag] = {}
        for driver in config["drivers"]:
            path = os.path.join(processed_dir, rtag, f"{driver}.parquet")
            raw[rtag][driver] = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    bundle["raw"] = raw

    return bundle


@st.cache_data(show_spinner=False)
def compute_headline_metrics(oof_df, embed_df):
    """
    Derive the headline numbers from the saved artefacts rather than
    hardcoding them, so they can never drift out of sync with a re-run
    of the pipeline.
    """
    out = {}
    if oof_df is not None:
        truth = oof_df["Driver"]
        if "oof_pred_intrack" in oof_df.columns:
            out["intrack"] = float((oof_df["oof_pred_intrack"] == truth).mean())
        if "oof_pred_crosscircuit" in oof_df.columns:
            out["cross"] = float((oof_df["oof_pred_crosscircuit"] == truth).mean())
    if embed_df is not None:
        dims = [c for c in embed_df.columns if c.startswith("dim_")]
        if dims:
            out["silhouette"] = float(silhouette_score(embed_df[dims].values,
                                                       embed_df["Driver"].values))
    return out


@st.cache_data(show_spinner=False)
def compute_silhouette(df):
    if df is None:
        return None
    dims = [c for c in df.columns if c.startswith("dim_")]
    if not dims:
        return None
    return float(silhouette_score(df[dims].values, df["Driver"].values, metric="cosine"))


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def base_layout(height=None, legend=True):
    kw = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family=FONT, size=11),
        margin=dict(l=8, r=8, t=10, b=8),
    )
    if height:
        kw["height"] = height
    if legend:
        kw["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                            orientation="h", yanchor="bottom", y=1.0,
                            xanchor="right", x=1.0)
    return kw


def make_radar(features_df, drivers):
    means = features_df.groupby("Driver")[FEATURE_COLS].mean()
    norm  = (means - means.min()) / (means.max() - means.min() + 1e-8)
    fig = go.Figure()
    for d in drivers:
        if d not in norm.index:
            continue
        vals = norm.loc[d].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1],
            theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
            fill="toself", name=d,
            line=dict(color=DRIVER_COLORS.get(d, "#888"), width=2),
            fillcolor=DRIVER_COLORS.get(d, "#888"), opacity=0.14,
            hovertemplate=f"<b>{d}</b> · %{{theta}}: %{{r:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **base_layout(430),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], gridcolor=BORDER, linecolor=BORDER,
                            tickfont=dict(color=TEXT_MUTE, size=8),
                            tickvals=[0.25, 0.5, 0.75, 1.0]),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(color=TEXT_DIM, size=10)),
        ),
    )
    return fig


def make_umap(umap_df, highlight=None):
    fig = go.Figure()
    has_race = "Race" in umap_df.columns
    for d in sorted(umap_df["Driver"].unique()):
        sub    = umap_df[umap_df["Driver"] == d]
        dimmed = highlight is not None and d != highlight
        color  = DRIVER_COLORS.get(d, "#888")
        if has_race:
            custom = np.stack([sub["Race"], sub["LapNumber"]], axis=-1)
            htmpl  = f"<b>{d}</b><br>%{{customdata[0]}} · Lap %{{customdata[1]}}<extra></extra>"
        else:
            custom = sub["LapNumber"]
            htmpl  = f"<b>{d}</b> · Lap %{{customdata}}<extra></extra>"
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"], mode="markers", name=d,
            marker=dict(color=color, size=7 if not dimmed else 5,
                        opacity=0.8 if not dimmed else 0.08,
                        line=dict(width=0)),
            customdata=custom, hovertemplate=htmpl,
        ))
        if not dimmed:
            fig.add_annotation(
                x=sub["umap_x"].mean(), y=sub["umap_y"].mean(), text=d,
                font=dict(color=color, size=13, family="Barlow Condensed"),
                showarrow=False, bgcolor="rgba(8,8,13,0.65)", borderpad=3,
            )
    fig.update_layout(**base_layout(430))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def get_lap_df(raw, driver, lap_num):
    return raw[driver][raw[driver]["LapNumber"] == lap_num].reset_index(drop=True)


def make_telemetry(raw, driver, lap_num, compare=None, height=400):
    """Four stacked channel traces. `compare` overlays a second driver."""
    lap_df = get_lap_df(raw, driver, lap_num)
    if lap_df.empty:
        return None

    series = [(driver, lap_df, DRIVER_COLORS.get(driver, "#888"), 1.0)]
    if compare:
        cmp_df = get_lap_df(raw, compare, lap_num)
        if not cmp_df.empty:
            series.append((compare, cmp_df, DRIVER_COLORS.get(compare, "#888"), 0.55))

    channels = [("Throttle", "Throttle %"), ("Brake", "Brake"),
                ("Speed", "Speed km/h"), ("nGear", "Gear")]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.3, 0.14, 0.35, 0.21])

    for row, (ch, label) in enumerate(channels, 1):
        for name, df, color, alpha in series:
            if ch not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=np.arange(len(df)), y=df[ch], mode="lines", name=name,
                line=dict(color=color, width=1.3),
                opacity=alpha, legendgroup=name,
                showlegend=(row == 1 and len(series) > 1),
                hovertemplate=f"<b>{name}</b> · {label}: %{{y}}<extra></extra>",
            ), row=row, col=1)
        fig.update_yaxes(title_text=label, row=row, col=1, gridcolor=BORDER,
                         zerolinecolor=BORDER, title_font=dict(size=9, color=TEXT_MUTE),
                         tickfont=dict(size=8, color=TEXT_MUTE))

    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER,
                     tickfont=dict(size=8, color=TEXT_MUTE))
    fig.update_layout(**base_layout(height, legend=len(series) > 1))
    fig.update_xaxes(title_text="Telemetry sample", row=4, col=1,
                     title_font=dict(size=9, color=TEXT_MUTE))
    return fig


def make_feature_importance(fi_df):
    df = fi_df.sort_values("importance")
    labels = dict(zip(FEATURE_COLS, FEATURE_LABELS))
    fig = go.Figure(go.Bar(
        x=df["importance"], y=[labels.get(f, f) for f in df["feature"]],
        orientation="h",
        marker=dict(color=df["importance"],
                    colorscale=[[0, "#2a2a3d"], [0.5, "#7a2530"], [1, ACCENT]],
                    showscale=False),
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="outside", textfont=dict(size=9, color=TEXT_MUTE),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(340, legend=False))
    fig.update_xaxes(title_text="Gain importance", gridcolor=BORDER,
                     title_font=dict(size=9, color=TEXT_MUTE),
                     tickfont=dict(size=8, color=TEXT_MUTE),
                     range=[0, df["importance"].max() * 1.2])
    fig.update_yaxes(tickfont=dict(size=9, color=TEXT_DIM), gridcolor=BORDER)
    return fig


def make_training_curve(history_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["train_acc"] * 100, mode="lines",
        name="Train", line=dict(color="#3b82f6", width=2),
        hovertemplate="Epoch %{x} · Train %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["val_acc"] * 100, mode="lines",
        name="Validation (held-out circuits)",
        line=dict(color=ACCENT, width=2, dash="dot"),
        hovertemplate="Epoch %{x} · Val %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**base_layout(340))
    fig.update_xaxes(title_text="Epoch", gridcolor=BORDER,
                     title_font=dict(size=9, color=TEXT_MUTE),
                     tickfont=dict(size=8, color=TEXT_MUTE))
    fig.update_yaxes(title_text="Accuracy %", gridcolor=BORDER, range=[0, 105],
                     title_font=dict(size=9, color=TEXT_MUTE),
                     tickfont=dict(size=8, color=TEXT_MUTE))
    return fig


# ─────────────────────────────────────────────
# Sonification — turn a lap's telemetry into audio.
# RPM -> pitch, Throttle -> volume, Brake -> percussive thump.
# Exposes the same style differences the XGBoost features capture
# (throttle smoothness, braking frequency, trail braking) through
# your ears instead of your eyes.
# ─────────────────────────────────────────────
SAMPLE_RATE = 22050


def sonify_lap(lap_df, sample_rate: int = SAMPLE_RATE) -> tuple:
    """Returns (audio: float32 np.array in [-1, 1], sample_rate)."""
    n = len(lap_df)
    if n < 2:
        return None, sample_rate

    lap_time = lap_df["LapTime_s"].iloc[0] if "LapTime_s" in lap_df.columns else np.nan
    if pd.isna(lap_time) or lap_time <= 0:
        lap_time = 90.0
    duration = float(np.clip(lap_time / 8.0, 5.0, 18.0))

    rpm = (lap_df["RPM"].ffill().bfill().values.astype(float)
           if "RPM" in lap_df.columns else np.full(n, 10000.0))
    throttle = (lap_df["Throttle"].fillna(0).values.astype(float)
                if "Throttle" in lap_df.columns else np.full(n, 50.0))
    brake = ((lap_df["Brake"].fillna(0).values.astype(float) > 0).astype(float)
             if "Brake" in lap_df.columns else np.zeros(n))

    t_control = np.linspace(0, duration, n)
    n_samples = int(sample_rate * duration)
    t_audio   = np.linspace(0, duration, n_samples)

    rpm_i      = np.interp(t_audio, t_control, rpm)
    throttle_i = np.interp(t_audio, t_control, throttle)

    rpm_lo, rpm_hi = np.percentile(rpm[rpm > 0], [5, 95]) if np.any(rpm > 0) else (4000, 12000)
    if rpm_hi <= rpm_lo:
        rpm_hi = rpm_lo + 1000
    freq = np.interp(rpm_i, [rpm_lo, rpm_hi], [90, 340])

    # Continuous phase (cumulative) avoids clicks from frequency jumps.
    phase  = 2 * np.pi * np.cumsum(freq) / sample_rate
    engine = 0.65 * sawtooth(phase) + 0.35 * sawtooth(2 * phase + 0.4)

    amp   = np.interp(throttle_i, [0, 100], [0.06, 0.8])
    audio = engine * amp

    # Percussive thump on every brake-on edge, detected on the original
    # unsmoothed signal — interpolation would blur out short brake stabs.
    brake_onsets = np.where(np.diff(brake, prepend=0) > 0)[0]
    thump_len = int(0.09 * sample_rate)
    thump_env = np.exp(-np.linspace(0, 12, thump_len))
    rng = np.random.default_rng(42)
    for onset_idx in brake_onsets:
        t0  = int(t_control[onset_idx] / duration * n_samples)
        end = min(t0 + thump_len, n_samples)
        seg = end - t0
        if seg <= 0:
            continue
        audio[t0:end] += 0.55 * rng.uniform(-1, 1, seg) * thump_env[:seg]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.9
    return audio.astype(np.float32), sample_rate


# ─────────────────────────────────────────────
# Small render helpers
# ─────────────────────────────────────────────
def metric_card(col, color, label, value, note):
    col.markdown(
        f"<div class='metric' style='border-top:2px solid {color};'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{color}'>{value}</div>"
        f"<div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def slabel(text):
    st.markdown(f"<div class='slabel'>{text}</div>", unsafe_allow_html=True)


def race_label(tag):
    year, _, name = tag.partition("_")
    return f"{name.title()} {year}"


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
try:
    D = load_all_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}\n\nRun the pipeline scripts first.")
    st.stop()

config      = D["config"]
features_df = D["features_df"]
umap_df     = D["umap_df"]
history_df  = D["history_df"]
fi_df       = D["fi_df"]
xgb_le      = D["xgb_le"]
oof_df      = D["oof_df"]
raw         = D["raw"]

all_drivers  = config["drivers"]
race_tags    = [f"{r['year']}_{r['race'].lower()}" for r in config["races"]]
seasons      = sorted({r["year"] for r in config["races"]})
metrics      = compute_headline_metrics(oof_df, D["embed_df"])

# ─────────────────────────────────────────────
# Masthead
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='masthead'>
  <div>
    <div class='mast-title'>F1 TELEMETRY <span>LENS</span></div>
    <div class='mast-tag'>
      Can a model recognise a driver from nothing but how they use the throttle, brake
      and gearbox? Every lap here is reduced to a style fingerprint &mdash; no lap times,
      no team labels, no circuit hints.
    </div>
  </div>
  <div class='mast-meta'>
    <b>{len(config['races'])}</b> races &nbsp;·&nbsp; {seasons[0]}&ndash;{seasons[-1]}<br>
    <b>{len(all_drivers)}</b> drivers &nbsp;·&nbsp; <b>{len(features_df):,}</b> laps<br>
    XGBoost &nbsp;·&nbsp; 1D-CNN &nbsp;·&nbsp; UMAP
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

cols = st.columns(5, gap="small")
metric_card(cols[0], ACCENT, "Same-circuit acc",
            f"{metrics['intrack']*100:.1f}%" if "intrack" in metrics else "n/a",
            "XGBoost, stratified 5-fold")
metric_card(cols[1], "#f97316", "Cross-circuit acc",
            f"{metrics['cross']*100:.1f}%" if "cross" in metrics else "n/a",
            "Entire circuits held out")
metric_card(cols[2], "#a78bfa", "Silhouette",
            f"{metrics['silhouette']:.2f}" if "silhouette" in metrics else "n/a",
            "CNN embeddings, 32-dim")
metric_card(cols[3], "#facc15", "Random baseline",
            f"{100/len(all_drivers):.1f}%", f"1 in {len(all_drivers)} drivers")
metric_card(cols[4], "#3b82f6", "Laps analysed",
            f"{len(features_df):,}", "After quality filtering")

st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

tab_overview, tab_blind, tab_telemetry, tab_model = st.tabs(
    ["Fingerprints", "Blind test", "Telemetry & audio", "Model"]
)

# ═════════════════════════════════════════════
# TAB 1 — Fingerprints
# ═════════════════════════════════════════════
with tab_overview:
    st.markdown(
        "<div class='intro'>Two views of the same question. On the left, nine "
        "hand-crafted style metrics averaged per driver &mdash; interpretable, but "
        "coarse. On the right, a 1D-CNN's own 32-dimensional representation of every "
        "individual lap, projected to 2D. Nobody told the network what to look for; the "
        "clusters are what it decided mattered.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        slabel("Style fingerprint")
        picked = st.multiselect(
            "Drivers on the radar", options=all_drivers, default=all_drivers,
            format_func=lambda d: f"{d} — {DRIVER_NAMES.get(d, d)}",
        )
        if picked:
            st.plotly_chart(make_radar(features_df, picked), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(
                "<div class='note'>Each axis is one metric, min-max normalised across "
                "all six drivers, so the <b>shape</b> is the signature rather than the "
                "size. Overlapping polygons are exactly the pairs the classifier "
                "confuses most.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='empty'>Pick at least one driver to draw the "
                        "radar.</div>", unsafe_allow_html=True)

    with right:
        slabel("Learned embedding space")
        focus = st.selectbox(
            "Isolate a driver", options=["Show all"] + all_drivers,
            format_func=lambda d: d if d == "Show all" else f"{d} — {DRIVER_NAMES.get(d, d)}",
        )
        st.plotly_chart(
            make_umap(umap_df, None if focus == "Show all" else focus),
            width="stretch", config={"displayModeBar": False},
        )
        sil_txt = (f"Silhouette <b>{metrics['silhouette']:.2f}</b> across all 12 races. "
                   if "silhouette" in metrics else "")
        st.markdown(
            f"<div class='note'>One dot per lap, UMAP-projected from the CNN's 32-dim "
            f"embedding. {sil_txt}Clusters this separated across circuits the encoder "
            f"never trained on is the strongest evidence here that something "
            f"driver-specific is being learned &mdash; with the caveat in the Model "
            f"tab about how much of it is really the car.</div>",
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════
# TAB 2 — Blind test
# ═════════════════════════════════════════════
with tab_blind:
    if oof_df is None:
        st.markdown(
            "<div class='empty'>No out-of-fold prediction file found. Run "
            "<code>python src/models/baseline.py</code> to generate "
            "<code>all_races_oof_predictions.csv</code>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='intro'>Draw a lap at random and see whether the classifier "
            "gets it. The prediction is a genuine <b>out-of-fold</b> result recorded "
            "during cross-validation &mdash; the model that scored this lap never saw "
            "it in training. It is not the final model re-scoring data it already "
            "memorised, which would make the whole demo meaningless. Everything "
            "feeding the decision is shown below so you can check it rather than "
            "take it on trust.</div>",
            unsafe_allow_html=True,
        )

        ctrl_left, ctrl_right = st.columns([2.2, 1], gap="large")
        with ctrl_left:
            mode = st.radio(
                "Evaluation split",
                options=["Same circuit", "Held-out circuit"],
                horizontal=True,
                captions=[
                    f"Stratified 5-fold — {metrics.get('intrack', 0)*100:.1f}% overall",
                    f"GroupKFold by race — {metrics.get('cross', 0)*100:.1f}% overall",
                ],
            )
        with ctrl_right:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            draw = st.button("Draw a random lap", width="stretch", type="primary")

        cross = mode == "Held-out circuit"
        pred_col = "oof_pred_crosscircuit" if cross else "oof_pred_intrack"
        prob_pre = "oof_prob_crosscircuit" if cross else "oof_prob_intrack"

        missing_split = pred_col not in oof_df.columns
        if missing_split:
            st.markdown(
                "<div class='empty'>This split is not present in the prediction file. "
                "Re-run <code>baseline.py</code> with data from five or more "
                "races.</div>", unsafe_allow_html=True)

    if oof_df is not None and not missing_split:
        # Draw one on first visit so the tab is never empty.
        if draw or "blind_idx" not in st.session_state:
            st.session_state["blind_idx"] = int(oof_df.sample(1).index[0])

        lap = oof_df.loc[st.session_state["blind_idx"]]
        truth = lap["Driver"]
        pred  = lap[pred_col]
        hit   = pred == truth
        probs = {c: lap[f"{prob_pre}_{c}"] for c in xgb_le.classes_
                 if f"{prob_pre}_{c}" in lap.index}

        tcolor = DRIVER_COLORS.get(truth, TEXT)
        vcolor = "#22c55e" if hit else ACCENT

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        card, bars = st.columns([1, 1.7], gap="large")

        with card:
            st.markdown(f"""
            <div class='vcard' style='border-left:3px solid {tcolor};'>
              <div class='vcard-kicker'>{lap['Race']} {int(lap['Season'])} &nbsp;·&nbsp; Lap {int(lap['LapNumber'])}</div>
              <div class='vcard-code' style='color:{tcolor};'>{truth}</div>
              <div class='vcard-name'>{DRIVER_NAMES.get(truth, truth)}
                &nbsp;·&nbsp; {lap['LapTime_s']:.2f}s</div>
              <div class='verdict' style='color:{vcolor};'>
                {'Identified correctly' if hit else 'Misidentified'}
                <div class='verdict-sub'>Model said {pred}
                  ({DRIVER_NAMES.get(pred, pred)})</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with bars:
            slabel("Out-of-fold confidence")
            rows = ""
            for cls, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
                c = DRIVER_COLORS.get(cls, "#888")
                lead = cls == pred
                rows += (
                    f"<div class='pbar-row'>"
                    f"<div class='pbar-head'>"
                    f"<span style='color:{c};font-weight:{700 if lead else 400};'>"
                    f"{cls} — {DRIVER_NAMES.get(cls, cls)}</span>"
                    f"<span style='color:{TEXT_DIM};font-variant-numeric:tabular-nums;'>"
                    f"{p*100:.1f}%</span></div>"
                    f"<div class='pbar-track'><div class='pbar-fill' style='background:{c};"
                    f"width:{max(p*100, 0.6):.1f}%;opacity:{1 if lead else 0.35};'></div></div>"
                    f"</div>"
                )
            st.markdown(f"<div class='panel'>{rows}</div>", unsafe_allow_html=True)

        with st.expander("Audit this prediction — the exact inputs and raw trace"):
            match = features_df[
                (features_df["Driver"] == truth)
                & (features_df["Race"] == lap["Race"])
                & (features_df["Season"] == lap["Season"])
                & (features_df["LapNumber"] == lap["LapNumber"])
            ]
            acol, bcol = st.columns([1, 1.5], gap="large")
            with acol:
                slabel("The nine values fed to the model")
                if match.empty:
                    st.markdown("<div class='empty'>Feature row not found — the "
                                "features file looks out of sync with the prediction "
                                "file.</div>", unsafe_allow_html=True)
                else:
                    r = match.iloc[0]
                    body = "".join(f"<tr><td>{lbl}</td><td>{r[col]:.4f}</td></tr>"
                                   for col, lbl in zip(FEATURE_COLS, FEATURE_LABELS))
                    st.markdown(f"<table class='kv'>{body}</table>", unsafe_allow_html=True)
            with bcol:
                slabel("Raw telemetry for this lap")
                rtag = f"{int(lap['Season'])}_{lap['Race'].lower()}"
                if rtag in raw and not raw[rtag][truth].empty:
                    fig = make_telemetry(raw[rtag], truth, float(lap["LapNumber"]), height=300)
                    if fig:
                        st.plotly_chart(fig, width="stretch",
                                        config={"displayModeBar": False})
                    else:
                        st.markdown("<div class='empty'>No telemetry samples for this "
                                    "lap.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='empty'>Raw telemetry is not bundled in "
                                "this environment.</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# TAB 3 — Telemetry & audio
# ═════════════════════════════════════════════
with tab_telemetry:
    st.markdown(
        "<div class='intro'>The raw signal behind everything else. Pick a lap, then "
        "overlay a second driver to see where two styles diverge &mdash; and use the "
        "same selection to <b>hear</b> it, since a braking rhythm is often easier to "
        "recognise by ear than by eye.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    sel_race = c1.selectbox("Race", options=race_tags, format_func=race_label,
                            index=len(race_tags) - 1)
    sel_driver = c2.selectbox("Driver", options=all_drivers,
                              format_func=lambda d: f"{d} — {DRIVER_NAMES.get(d, d)}")

    available = raw.get(sel_race, {})
    has_data = bool(available) and not available[sel_driver].empty

    if has_data:
        laps = sorted(available[sel_driver]["LapNumber"].dropna().unique().astype(int))
        sel_lap = c3.selectbox("Lap", options=laps, index=min(9, len(laps) - 1))
        others = [d for d in all_drivers if d != sel_driver
                  and not available.get(d, pd.DataFrame()).empty]
        sel_cmp = c4.selectbox("Overlay", options=["None"] + others,
                               format_func=lambda d: d if d == "None"
                               else f"{d} — {DRIVER_NAMES.get(d, d)}")
        cmp_driver = None if sel_cmp == "None" else sel_cmp
    else:
        sel_lap, cmp_driver = None, None
        c3.selectbox("Lap", options=["—"], disabled=True)
        c4.selectbox("Overlay", options=["—"], disabled=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if not has_data:
        st.markdown(
            "<div class='empty'>Raw telemetry is not available in this environment "
            "&mdash; the parquet files are large and may be excluded from the deploy. "
            "Clone the repo and run <code>python src/data/fetch_telemetry.py</code> to "
            "populate it.</div>", unsafe_allow_html=True)
    else:
        fig = make_telemetry(raw[sel_race], sel_driver, float(sel_lap), compare=cmp_driver)
        if fig:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            "<div class='note'><b>Throttle</b> — square drops mean an abrupt lift, "
            "gradual ramps mean a smoother release. &nbsp; <b>Brake</b> — binary, so "
            "each block is one braking zone. &nbsp; <b>Speed</b> — the valleys are "
            "corners, and a higher valley floor means more speed carried through. "
            "&nbsp; <b>Gear</b> — busier steps mean more mechanical input.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        slabel("The same lap, as sound")
        st.markdown(
            "<div class='intro'>Engine pitch follows <b>RPM</b>, loudness follows "
            "<b>throttle</b>, and every <b>braking</b> zone lands as a percussive hit. "
            "Generate both drivers and play them back to back — a smooth trail-braker "
            "and a late stabber sound obviously different.</div>",
            unsafe_allow_html=True,
        )

        def audio_block(container, driver, slot):
            """Render a generate-button plus player, keyed so stale audio is discarded."""
            sig = f"{sel_race}|{driver}|{sel_lap}"
            with container:
                st.markdown(
                    f"<div class='slabel' style='color:{DRIVER_COLORS.get(driver, TEXT)}'>"
                    f"{driver} — {DRIVER_NAMES.get(driver, driver)}</div>",
                    unsafe_allow_html=True)
                if st.button("Generate audio", key=f"gen_{slot}", width="stretch"):
                    st.session_state[f"audio_{slot}"] = (sig, sonify_lap(
                        get_lap_df(raw[sel_race], driver, float(sel_lap))))
                stored = st.session_state.get(f"audio_{slot}")
                if stored and stored[0] == sig and stored[1][0] is not None:
                    samples, rate = stored[1]
                    st.audio(samples, sample_rate=rate)
                elif stored and stored[0] != sig:
                    st.markdown(f"<div class='metric-note'>Selection changed — "
                                f"generate again.</div>", unsafe_allow_html=True)

        acol, bcol = st.columns(2, gap="large")
        audio_block(acol, sel_driver, "a")
        if cmp_driver:
            audio_block(bcol, cmp_driver, "b")
        else:
            bcol.markdown("<div style='height:1.9rem'></div>"
                          "<div class='empty'>Choose an overlay driver above to "
                          "compare two laps by ear.</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# TAB 4 — Model
# ═════════════════════════════════════════════
with tab_model:
    st.markdown(
        "<div class='intro'>What the models actually learned, and where the result "
        "should be treated with suspicion.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        slabel("Which features carried the signal")
        st.plotly_chart(make_feature_importance(fi_df), width="stretch",
                        config={"displayModeBar": False})
        st.markdown(
            "<div class='note'><b>Trail braking</b> and <b>gear changes</b> lead, which "
            "is reassuring — both describe how a driver works the car rather than how "
            "fast the car is. <b>Corner speed</b> ranks high too, but it is the one "
            "metric most contaminated by raw machinery.</div>",
            unsafe_allow_html=True,
        )

    with right:
        slabel("CNN training, validated on unseen circuits")
        st.plotly_chart(make_training_curve(history_df), width="stretch",
                        config={"displayModeBar": False})
        best_val = history_df["val_acc"].max() * 100
        st.markdown(
            f"<div class='note'>Validation laps come from <b>entirely held-out "
            f"circuits</b>, not merely unseen laps, peaking at <b>{best_val:.1f}%</b>. "
            f"That it beats the feature-based model by such a margin is itself a "
            f"warning sign — see below.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    slabel("How the two evaluations differ")
    ev1, ev2 = st.columns(2, gap="large")
    with ev1:
        st.markdown(f"""
        <div class='panel'>
          <div style='font-family:Barlow Condensed;font-size:1.6rem;font-weight:700;
          color:{ACCENT};line-height:1;'>{metrics.get('intrack', 0)*100:.1f}%</div>
          <div class='metric-label' style='margin-top:6px;'>Same circuit · stratified 5-fold</div>
          <div class='metric-note'>Laps from one race can land in both the training and
          validation folds. It answers "is there a detectable style at all", and it is
          the number most projects would quote on its own.</div>
        </div>""", unsafe_allow_html=True)
    with ev2:
        gap = (metrics.get("intrack", 0) - metrics.get("cross", 0)) * 100
        st.markdown(f"""
        <div class='panel'>
          <div style='font-family:Barlow Condensed;font-size:1.6rem;font-weight:700;
          color:#f97316;line-height:1;'>{metrics.get('cross', 0)*100:.1f}%</div>
          <div class='metric-label' style='margin-top:6px;'>Held-out circuit · GroupKFold by race</div>
          <div class='metric-note'>Whole circuits are removed from training, so nothing
          from the test track leaks in. The <b>{gap:.1f} point</b> drop is the honest
          cost of generalising to a track the model has never seen.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    slabel("Known limitations")
    st.markdown(f"""
    <div class='panel' style='font-size:0.84rem;color:{TEXT_DIM};line-height:1.7;'>
      <p style='margin:0 0 0.9rem;'><b style='color:{TEXT};'>The car is a confound, and
      it showed up concretely.</b> An early version fed <code>RPM</code> to the CNN and
      scored a suspicious 99.4% on held-out circuits. RPM range is set almost entirely by
      the power unit, so the network was largely fingerprinting the car. Dropping the
      channel took it to 94.2% — still far above the 61.2% the feature-based model
      manages, which suggests <code>Speed</code> and <code>nGear</code> carry residual
      car and aero signal too.</p>
      <p style='margin:0 0 0.9rem;'><b style='color:{TEXT};'>The clean experiment has not
      been run yet.</b> Comparing team-mates in identical machinery — Verstappen against
      Pérez, Leclerc against Sainz — would separate driver from car properly. Until then
      every number here should be read as "driver and car combined".</p>
      <p style='margin:0;'><b style='color:{TEXT};'>Two seasons is a narrow window.</b>
      Twelve races across 2023–2024 keeps the driver line-ups stable, which is what makes
      the comparison tractable, but it also means these fingerprints are not tested
      against regulation changes or mid-career team moves.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    sil_ce = metrics.get("silhouette")
    sil_sc = compute_silhouette(D["contrast_df"])
    if sil_sc is not None:
        slabel("Encoder comparison")
        st.markdown(f"""
        <div class='panel'>
          <table class='kv'>
            <tr><td>Cross-entropy encoder — silhouette (32-dim)</td>
                <td>{sil_ce:.2f}</td></tr>
            <tr><td>Supervised-contrastive encoder — silhouette (cosine)</td>
                <td>{sil_sc:.2f}</td></tr>
          </table>
          <div class='metric-note' style='margin-top:0.8rem;'>Training the same backbone
          with a SupCon objective, so that same-driver laps are pulled together directly
          rather than as a side effect of classification, produces a visibly tighter
          embedding space. Both were trained and scored on the same held-out-circuit
          split.</div>
        </div>""", unsafe_allow_html=True)
