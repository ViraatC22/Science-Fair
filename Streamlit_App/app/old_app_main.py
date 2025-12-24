import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
import time
from scipy import stats

# Import our new modules
from simulation_logic import run_simulation_logic, LITERATURE_DATA
from ml_modules import ModelManager
from optimization_logic import GeneticOptimizer, train_initial_model

# --- THEME SELECTION MAP (Restored) ---
_theme_map = {
    "Ocean (Blue/Teal)": {"primary": "#1f77b4", "secondary": "#2c3e50", "accent": "#00b894", "bg": "#f7f9fc", "text": "#2c3e50", "surface": "#ffffff", "surface_border": "#e6edf5", "sidebar": "#fbfcfe", "dark": False},
    "Forest (Green)": {"primary": "#2ecc71", "secondary": "#1b5e20", "accent": "#27ae60", "bg": "#f6fbf6", "text": "#1b4332", "surface": "#ffffff", "surface_border": "#e4efe7", "sidebar": "#f7fbf8", "dark": False},
    "Sunset (Orange/Purple)": {"primary": "#ff7f0e", "secondary": "#5b2c6f", "accent": "#c0392b", "bg": "#fff8f2", "text": "#2d2a32", "surface": "#ffffff", "surface_border": "#f3e6dc", "sidebar": "#fffaf5", "dark": False},
    "Monochrome (Gray)": {"primary": "#6c757d", "secondary": "#343a40", "accent": "#868e96", "bg": "#f8f9fa", "text": "#343a40", "surface": "#ffffff", "surface_border": "#e9ecef", "sidebar": "#f8f9fa", "dark": False},
    "Dark Slate": {"primary": "#3b82f6", "secondary": "#e5e7eb", "accent": "#22c55e", "bg": "#0b1220", "text": "#e5e7eb", "surface": "#111827", "surface_border": "#1f2937", "sidebar": "#0f172a", "dark": True}
}

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Neural Slime Simulation", page_icon="🧠")

# --- INITIALIZE SESSION STATE ---
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = GeneticOptimizer(st.session_state.model_manager)
if 'nn_config' not in st.session_state:
    st.session_state.nn_config = {}
if 'last_report' not in st.session_state:
    st.session_state.last_report = ""
if 'last_execution_seed' not in st.session_state:
    st.session_state.last_execution_seed = None
if 'run_history' not in st.session_state:
    st.session_state.run_history = []
if 'last_optimization_history' not in st.session_state:
    st.session_state.last_optimization_history = []
if 'last_report_data' not in st.session_state:
    st.session_state.last_report_data = {}

def _parse_hidden_layers(text):
    raw = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    out = []
    for x in raw:
        try:
            v = int(x)
            if v > 0:
                out.append(v)
        except ValueError:
            continue
    return out if out else [64, 128, 64]

def _metric_help():
    return {
        "avg_growth_rate": "How fast Physarum expands along the scaffold (mm/hr). Higher is better for rapid coverage.",
        "total_network_length": "Total length of all tube segments in the network (mm). Higher usually means more exploration/connectivity.",
        "num_junctions": "Count of junctions where 3+ branches meet (network complexity).",
        "num_edges": "Count of tube segments between junctions (graph edges).",
        "avg_branches_per_node": "Average branching at each junction (connectivity).",
        "graph_density": "Edges present / max possible edges (0–1). Higher means more connected.",
        "dye_diffusion_rate": "Proxy for dye spread rate if you add dye to nutrient pockets.",
        "time_to_connection": "Estimated minutes to connect two random points (lower is faster).",
        "time_to_reconnection": "Estimated minutes to reconnect after a local removal (lower is more resilient).",
        "mean_tortuosity": "How winding paths are (1.0 is straight). Higher means more complex paths.",
        "fractal_dimension": "Complexity of the branching pattern (higher is more complex).",
        "penetration_depth": "Estimated depth into nutrient layers (mm) for 3D models.",
        "mst_ratio": "Literature metric: total length normalized by MST length (efficiency/robustness proxy).",
        "path_efficiency": "Literature-style transport efficiency proxy (higher is better).",
    }

def _format_label(key):
    return str(key).replace("_", " ").strip()

def _try_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None

def _metric_units():
    return {
        "avg_growth_rate": "mm/hr",
        "total_network_length": "mm",
        "num_junctions": "count",
        "num_edges": "count",
        "avg_branches_per_node": "unitless",
        "graph_density": "ratio",
        "dye_diffusion_rate": "proxy",
        "time_to_connection": "min",
        "time_to_reconnection": "min",
        "mean_tortuosity": "unitless",
        "fractal_dimension": "unitless",
        "penetration_depth": "mm",
        "mst_ratio": "ratio",
        "path_efficiency": "ratio",
    }

def _higher_is_better(metric_key):
    lower_better = {
        "time_to_connection",
        "time_to_reconnection",
        "mean_tortuosity",
    }
    return metric_key not in lower_better

def _coerce_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None

def _downsample_df(df, x_col, y_col, max_points=900):
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return df
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / float(max_points)))
    return df.iloc[::max(1, stride), :].copy()

def _range_from_literature(metric_key):
    if metric_key == "avg_growth_rate":
        low = LITERATURE_DATA.get("Kay_2022_GrowthRate_Low")
        high = LITERATURE_DATA.get("Kay_2022_GrowthRate_High")
        if low and high:
            lo = _try_float(low.get("mean"))
            hi = _try_float(high.get("mean"))
            if lo is not None and hi is not None and hi > lo:
                return float(lo), float(hi)
    if metric_key == "total_network_length":
        ref = LITERATURE_DATA.get("AutoAnalysis_TotalLength")
        if ref:
            mean = _try_float(ref.get("mean"))
            std = _try_float(ref.get("std"))
            if mean is not None and std is not None and std > 0:
                return max(0.0, mean - 2 * std), mean + 2 * std
    if metric_key == "mst_ratio":
        ref = LITERATURE_DATA.get("Tero_2010_MST_Ratio")
        if ref:
            mean = _try_float(ref.get("mean"))
            std = _try_float(ref.get("std"))
            if mean is not None and std is not None and std > 0:
                return max(0.0, mean - 2 * std), mean + 2 * std
    if metric_key == "path_efficiency":
        ref = LITERATURE_DATA.get("Tero_2010_Transport_Ratio")
        if ref:
            mean = _try_float(ref.get("mean"))
            std = _try_float(ref.get("std"))
            if mean is not None and std is not None and std > 0:
                return max(0.0, mean - 2 * std), mean + 2 * std
    return None

def _range_from_history(df_hist, metric_key):
    if df_hist is None or df_hist.empty or metric_key not in df_hist.columns:
        return None
    s = pd.to_numeric(df_hist[metric_key], errors="coerce").dropna()
    if s.empty:
        return None
    lo = float(s.min())
    hi = float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi == lo:
        return lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad

def _metric_chart_type(metric_key):
    k = str(metric_key)
    if "density" in k or "fraction" in k or "coverage" in k:
        return "area"
    if k.startswith("num_") or "count" in k or "edges" in k or "junction" in k:
        return "bar"
    if "time_to_" in k:
        return "line"
    if "rate" in k:
        return "line"
    if "ratio" in k:
        return "line"
    return "line"

def _metric_color(metric_key):
    pal = list(getattr(qualitative, "Set2", [])) or list(getattr(qualitative, "Plotly", []))
    if not pal:
        return _t["primary"]
    idx = abs(hash(str(metric_key))) % len(pal)
    return pal[idx]

def _color_with_opacity(color, opacity):
    c = str(color).strip()
    if c.startswith("#"):
        return hex_to_rgba(c, opacity)
    if c.startswith("rgb(") and c.endswith(")"):
        inner = c[len("rgb(") : -1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) == 3:
            try:
                r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
                return f"rgba({r}, {g}, {b}, {opacity})"
            except ValueError:
                return c
    if c.startswith("rgba(") and c.endswith(")"):
        return c
    return c

def _plot_metric_history(df_hist, metric_key, base_key, height=320):
    if df_hist is None or df_hist.empty or metric_key not in df_hist.columns:
        return None
    df = df_hist.copy()
    if "run_index" not in df.columns:
        df["run_index"] = np.arange(1, len(df) + 1)
    df["run_index"] = pd.to_numeric(df["run_index"], errors="coerce")
    df = df.dropna(subset=["run_index"])
    y = pd.to_numeric(df[metric_key], errors="coerce")
    df = df.assign(_y=y).dropna(subset=["_y"]).sort_values("run_index")
    if df.empty:
        return None
    df = _downsample_df(df, "run_index", "_y", max_points=900)
    chart_type = _metric_chart_type(metric_key)
    color = _metric_color(metric_key)
    unit = _metric_units().get(metric_key, "")
    hover_parts = ["Run %{x}", f"{_format_label(metric_key)}: %{{y:.6g}} {unit}".strip()]
    if "param_model_type" in df.columns:
        hover_parts.append("Model: %{customdata[0]}")
    if "run_seed" in df.columns:
        hover_parts.append("Seed: %{customdata[1]}")
    if "run_ts" in df.columns:
        hover_parts.append("Time: %{customdata[2]}")
    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"
    customdata = []
    if "param_model_type" in df.columns or "run_seed" in df.columns or "run_ts" in df.columns:
        customdata = np.stack(
            [
                df["param_model_type"].astype(str) if "param_model_type" in df.columns else np.array(["—"] * len(df)),
                df["run_seed"].astype(str) if "run_seed" in df.columns else np.array(["—"] * len(df)),
                df["run_ts"].astype(str) if "run_ts" in df.columns else np.array(["—"] * len(df)),
            ],
            axis=1,
        )

    fig = go.Figure()
    if chart_type == "bar":
        fig.add_trace(
            go.Bar(
                x=df["run_index"],
                y=df["_y"],
                marker_color=color,
                opacity=0.9,
                hovertemplate=hovertemplate,
                customdata=customdata if len(customdata) else None,
            )
        )
    else:
        fill = "tozeroy" if chart_type == "area" else None
        fig.add_trace(
            go.Scatter(
                x=df["run_index"],
                y=df["_y"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                fill=fill,
                fillcolor=_color_with_opacity(color, 0.18) if fill else None,
                hovertemplate=hovertemplate,
                customdata=customdata if len(customdata) else None,
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=30, r=20, t=60, b=30),
        title=f"{_format_label(metric_key)} ({unit})" if unit else _format_label(metric_key),
        xaxis=dict(
            title="Run #",
            rangeslider=dict(visible=True) if chart_type != "bar" else None,
        ),
        yaxis=dict(title=unit or "Value"),
        transition=dict(duration=350, easing="cubic-in-out"),
        showlegend=False,
    )
    return fig

def _plot_series(y, title, y_label=None):
    if not y:
        return None
    idx = list(range(len(y)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y,
            mode="lines+markers",
            line=dict(color=_t["primary"], width=2),
            marker=dict(size=5, color=_t["accent"]),
            hovertemplate="Step %{x}<br>Value %{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=30, r=20, t=60, b=30),
        xaxis_title="Step",
        yaxis_title=y_label or "Value",
        showlegend=False,
    )
    return fig

def _plot_ga_progress(best_fitness_history, height=320):
    if not best_fitness_history:
        return None
    y = pd.to_numeric(pd.Series(best_fitness_history), errors="coerce").dropna()
    if y.empty:
        return None
    gen = np.arange(1, len(y) + 1)
    dy = np.diff(np.r_[y.iloc[0], y.to_numpy()])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=gen,
            y=dy,
            marker_color=_color_with_opacity(_t["accent"], 0.35),
            hovertemplate="Gen %{x}<br>Δ best %{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=y,
            mode="lines+markers",
            line=dict(color=_t["primary"], width=2),
            marker=dict(
                size=8,
                color=dy,
                colorscale="Viridis",
                cmin=float(np.min(dy)),
                cmax=float(np.max(dy)) if float(np.max(dy)) != float(np.min(dy)) else float(np.min(dy)) + 1e-9,
                showscale=True,
                colorbar=dict(title="Δ"),
            ),
            fill="tozeroy",
            fillcolor=_color_with_opacity(_t["primary"], 0.12),
            hovertemplate="Gen %{x}<br>Best fitness %{y:.4g}<extra></extra>",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="🧬 Genetic Algorithm Progress",
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=30, r=20, t=60, b=30),
        xaxis=dict(title="Generation", dtick=1),
        yaxis=dict(title="Improvement (Δ best)", zeroline=True),
        yaxis2=dict(title="Best Fitness (Z)", overlaying="y", side="right", showgrid=False),
        showlegend=False,
        bargap=0.2,
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    return fig

def _plot_dmem_delta(params):
    dmem_baseline = {"dmem_glucose": 25.0, "dmem_glutamine": 45.0, "dmem_pyruvate": 1.0}
    rows = []
    for k, base in dmem_baseline.items():
        if k not in params:
            continue
        val = _try_float(params.get(k))
        if val is None:
            continue
        rows.append({"Component": _format_label(k.replace("dmem_", "")), "Delta (mM)": val - float(base), "Value (mM)": val})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = go.Figure()
    colors = [_t["accent"] if v >= 0 else _t["primary"] for v in df["Delta (mM)"]]
    fig.add_trace(
        go.Bar(
            x=df["Component"],
            y=df["Delta (mM)"],
            marker_color=colors,
            hovertemplate="%{x}<br>Δ %{y:.2f} mM<br>Value %{customdata:.2f} mM<extra></extra>",
            customdata=df["Value (mM)"],
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=_t["surface_border"])
    fig.update_layout(
        title="DMEM Setup vs Baseline (Δ from 25/45/1 mM)",
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(l=30, r=20, t=60, b=30),
        yaxis_title="Delta (mM)",
    )
    return fig

def _plot_lit_comparison(metrics, selected_pairs, normalize):
    rows = []
    for mk, lk in selected_pairs:
        if mk not in metrics or lk not in LITERATURE_DATA:
            continue
        sim = _try_float(metrics.get(mk))
        if sim is None:
            continue
        lit = LITERATURE_DATA[lk]
        mean = _try_float(lit.get("mean"))
        std = _try_float(lit.get("std"))
        if mean is None:
            continue
        if normalize:
            rows.append(
                {
                    "Metric": _format_label(mk),
                    "Simulation": sim / mean if mean != 0 else np.nan,
                    "Literature mean": 1.0,
                    "Literature std": (std / mean) if (std is not None and mean != 0) else None,
                    "Sim raw": sim,
                    "Lit raw": mean,
                }
            )
        else:
            rows.append(
                {
                    "Metric": _format_label(mk),
                    "Simulation": sim,
                    "Literature mean": mean,
                    "Literature std": std,
                    "Sim raw": sim,
                    "Lit raw": mean,
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Simulation", x=df["Metric"], y=df["Simulation"], marker_color=_t["primary"]))
    fig.add_trace(
        go.Bar(
            name="Literature mean",
            x=df["Metric"],
            y=df["Literature mean"],
            error_y=dict(
                type="data",
                array=[v if v is not None else 0.0 for v in df["Literature std"]],
                visible=True,
                thickness=1.5,
                width=4,
            ),
            marker_color=_t["secondary"],
        )
    )
    fig.update_layout(
        barmode="group",
        title="Simulation vs Literature",
        template=PLOTLY_TEMPLATE,
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=60, b=30),
    )
    y_title = "Ratio to literature mean (1.0 = match)" if normalize else "Value"
    fig.update_yaxes(title=y_title)
    return fig

def _plot_radar(metrics, selected_keys, df_hist=None):
    units = _metric_units()
    values = []
    labels = []
    custom = []
    for k in selected_keys:
        raw = _try_float(metrics.get(k))
        if raw is None:
            continue
        rng = _range_from_literature(k) or _range_from_history(df_hist, k)
        if rng is None:
            lo = 0.0
            hi = max(1.0, abs(raw) * 2.0)
        else:
            lo, hi = rng
        if hi == lo:
            hi = lo + 1.0
        scaled = (raw - lo) / (hi - lo)
        scaled = float(np.clip(scaled, 0.0, 1.0))
        if not _higher_is_better(k):
            scaled = 1.0 - scaled
        unit = units.get(k, "")
        label = f"{_format_label(k)} ({unit})" if unit else _format_label(k)
        values.append(scaled)
        labels.append(label)
        custom.append([raw, lo, hi, unit, int(_higher_is_better(k))])
    if len(values) < 3:
        return None
    values = values + [values[0]]
    labels = labels + [labels[0]]
    custom = custom + [custom[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="This run",
            line=dict(color=_t["primary"], width=2),
            fillcolor=hex_to_rgba(_t["primary"], 0.20),
            customdata=custom,
            hovertemplate=(
                "%{theta}"
                "<br>Raw: %{customdata[0]:.6g} %{customdata[3]}"
                "<br>Range: [%{customdata[1]:.3g}, %{customdata[2]:.3g}]"
                "<br>Score (0–1): %{r:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Output Profile (radar)",
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(l=30, r=20, t=60, b=30),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], tickvals=[0, 0.5, 1.0], ticktext=["Low", "Mid", "High"]),
        ),
        transition=dict(duration=450, easing="cubic-in-out"),
        showlegend=False,
    )
    return fig

def _run_history_df(report_data):
    rh = report_data.get("run_history") or []
    if not rh:
        return None
    df = pd.DataFrame(rh)
    if df.empty:
        return None
    return df

def _plot_run_history(df, metric_key):
    if df is None or df.empty or metric_key not in df.columns:
        return None
    y = pd.to_numeric(df[metric_key], errors="coerce").dropna()
    if y.empty:
        return None
    fig = go.Figure()
    if "param_model_type" in df.columns:
        for name, g in df.groupby("param_model_type"):
            yg = pd.to_numeric(g[metric_key], errors="coerce").dropna()
            if yg.empty:
                continue
            fig.add_trace(
                go.Box(
                    y=yg,
                    name=str(name),
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    marker=dict(size=6, opacity=0.7),
                    line=dict(width=1),
                )
            )
        fig.update_layout(title=f"{_format_label(metric_key)} by Model Type (run history)")
    else:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))),
                y=y,
                mode="lines+markers",
                line=dict(color=_t["primary"], width=2),
                marker=dict(size=6, color=_t["accent"]),
            )
        )
        fig.update_layout(title=f"{_format_label(metric_key)} Across Runs")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        margin=dict(l=30, r=20, t=60, b=30),
        yaxis_title=_format_label(metric_key),
    )
    return fig

def _keyify(text):
    s = str(text) if text is not None else "na"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "na"

def _render_execution_report(report_data):
    if not report_data:
        return
    cfg = report_data.get("config", {})
    summary = report_data.get("summary", {})
    metrics = report_data.get("metrics", {})
    params = report_data.get("params", {})
    literature = report_data.get("literature", {})
    tests = report_data.get("tests", {})
    glossary = _metric_help()
    df_hist = _run_history_df(report_data)
    plotly_cfg = {"displayModeBar": True, "scrollZoom": True, "displaylogo": False}
    base_key = "exec_report__" + "__".join(
        [
            _keyify(summary.get("seed")),
            _keyify(summary.get("mode")),
            _keyify(summary.get("model_type")),
        ]
    )

    st.subheader("🧾 Execution Report (Visual)")

    overview_tab, inputs_tab, outputs_tab, history_tab, validation_tab, technical_tab = st.tabs(
        ["Overview", "Inputs", "Outputs", "History", "Validation", "Technical"]
    )

    with overview_tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("Run Seed", str(summary.get("seed", "—")))
        c2.metric("Mode", str(summary.get("mode", "—")))
        c3.metric("Model", str(summary.get("model_type", "—")))

        k1, k2, k3 = st.columns(3)
        if "training_loss" in summary:
            k1.metric("Training Loss", f"{summary['training_loss']:.4f}")
        if "ga_best_fitness" in summary:
            k2.metric("GA Best Fitness (Z)", f"{summary['ga_best_fitness']:.2f}")
        if "p_value" in summary:
            pv = float(summary["p_value"])
            k3.metric("P-Value (vs growth baseline)", f"{pv:.5f}")

        takeaways = []
        gr = _try_float(metrics.get("avg_growth_rate"))
        if gr is not None:
            takeaways.append(f"Growth rate: {gr:.2f} mm/hr (how fast it spreads across the scaffold).")
        tl = _try_float(metrics.get("total_network_length"))
        if tl is not None:
            takeaways.append(f"Total network length: {tl:.1f} mm (how much tube network forms).")
        bn = _try_float(metrics.get("avg_branches_per_node"))
        if bn is not None:
            takeaways.append(f"Connectivity: {bn:.2f} branches/node (how interconnected it becomes).")
        pdp = _try_float(metrics.get("penetration_depth"))
        if pdp is not None and pdp > 0:
            takeaways.append(f"Penetration depth: {pdp:.2f} mm (3D-only depth into media layers).")
        if takeaways:
            st.markdown("\n".join([f"- {t}" for t in takeaways]))
        st.caption("This report links your scaffold + media setup to biological network outcomes for your science fair project.")

        radar_keys = ["avg_growth_rate", "total_network_length", "avg_branches_per_node", "mst_ratio", "mean_tortuosity", "fractal_dimension"]
        radar_fig = _plot_radar(metrics, radar_keys, df_hist=df_hist)
        if radar_fig is not None:
            st.plotly_chart(radar_fig, use_container_width=True, key=f"{base_key}__overview__radar", config=plotly_cfg)
            st.caption("Radar is a quick “shape” of outcomes; compare runs by pattern, not absolute scale.")

        if df_hist is not None:
            options = [c for c in df_hist.columns if c not in ["param_model_type", "run_seed", "run_index", "run_ts"]]
            options = [c for c in options if pd.to_numeric(df_hist[c], errors="coerce").notna().sum() >= 2]
            if options:
                default_metric = "avg_growth_rate" if "avg_growth_rate" in options else options[0]
                metric_key = st.selectbox("Run-history metric", options=options, index=options.index(default_metric))
            else:
                metric_key = None
            hist_fig = _plot_run_history(df_hist, metric_key)
            if hist_fig is not None:
                st.plotly_chart(hist_fig, use_container_width=True, key=f"{base_key}__overview__history__{_keyify(metric_key)}", config=plotly_cfg)

    with inputs_tab:
        st.markdown(
            "These are the knobs you control in the real build: scaffold geometry (CAD/print), media chemistry (DMEM + ions), and environmental conditions (light, depth, replenishment)."
        )
        quick = ["pillar_count", "pillar_size_mm", "channel_width_mm", "channel_node_size_mm", "scaffold_density_g_cm3", "scaffold_stiffness_kPa", "elasticity", "initial_mass_g", "media_depth_mm", "replenish_freq_hr", "light_type", "light_lumens"]
        shown = [k for k in quick if k in params]
        if shown:
            cols = st.columns(3)
            for i, k in enumerate(shown):
                cols[i % 3].metric(_format_label(k), f"{params.get(k)}")

        dmem_fig = _plot_dmem_delta(params)
        if dmem_fig is not None:
            st.plotly_chart(dmem_fig, use_container_width=True, key=f"{base_key}__inputs__dmem_delta", config=plotly_cfg)
            st.caption("Positive Δ means more concentrated than the DMEM baseline you listed (25/45/1 mM).")

        if any(k in params for k in ["ion_na", "ion_k", "ion_cl", "ion_ca"]):
            rows = []
            for k in ["ion_na", "ion_k", "ion_cl", "ion_ca"]:
                val = _try_float(params.get(k))
                if val is None:
                    continue
                rows.append({"Ion": _format_label(k.replace("ion_", "")).upper(), "mM": val})
            if rows:
                df = pd.DataFrame(rows)
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=df["Ion"],
                        y=df["mM"],
                        marker_color=_t["primary"],
                        hovertemplate="%{x}<br>%{y:.2f} mM<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Ions in Medium (mM)",
                    template=PLOTLY_TEMPLATE,
                    height=320,
                    margin=dict(l=30, r=20, t=60, b=30),
                    yaxis_title="mM",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{base_key}__inputs__ions", config=plotly_cfg)
                st.caption("These ions are treated as allowed chemical variables (no pheromone/other stimuli modeled).")

        st.divider()
        st.markdown("**Printability checks (based on your constraints)**")
        checks = []
        cw = _try_float(params.get("channel_width_mm"))
        if cw is not None:
            checks.append(("Channel width ≥ 2 mm", cw >= 2.0, f"{cw:.2f} mm"))
        ps = _try_float(params.get("pillar_size_mm"))
        if ps is not None:
            checks.append(("Feature size ≥ 2 mm", ps >= 2.0, f"{ps:.2f} mm"))
        md = _try_float(params.get("media_depth_mm"))
        if md is not None:
            checks.append(("Media depth within slider bounds", 0.5 <= md <= 5.0, f"{md:.2f} mm"))
        if checks:
            a, b, c = st.columns(3)
            cols = [a, b, c]
            for i, (name, ok, val) in enumerate(checks):
                cols[i % 3].metric(name, "OK" if ok else "Review", val)

    with outputs_tab:
        st.markdown(
            "These are the dependent variables you care about: how fast the mold grows, how connected its network becomes, and how robust/efficient the resulting transport graph is."
        )

        all_keys = [
            "avg_growth_rate",
            "total_network_length",
            "num_junctions",
            "num_edges",
            "avg_branches_per_node",
            "graph_density",
            "time_to_connection",
            "time_to_reconnection",
            "dye_diffusion_rate",
            "mst_ratio",
            "path_efficiency",
            "mean_tortuosity",
            "fractal_dimension",
            "penetration_depth",
        ]
        available = [k for k in all_keys if k in metrics and metrics[k] is not None]
        default_sel = [k for k in ["avg_growth_rate", "total_network_length", "mst_ratio", "mean_tortuosity"] if k in available]
        selected = st.multiselect("Show metric cards", options=available, default=default_sel)
        if selected:
            cols = st.columns(3)
            for i, k in enumerate(selected):
                val = metrics.get(k)
                cols[i % 3].metric(_format_label(k), f"{val:.4g}" if isinstance(val, (int, float)) else str(val))
                if k in glossary:
                    cols[i % 3].caption(glossary[k])

        if available:
            profile_fig = _plot_radar(metrics, selected if len(selected) >= 3 else available[:6], df_hist=df_hist)
            if profile_fig is not None:
                st.plotly_chart(profile_fig, use_container_width=True, key=f"{base_key}__outputs__radar", config=plotly_cfg)

    with history_tab:
        st.markdown(
            "A per-metric dashboard for all tracked outputs. Each chart updates automatically as you run more simulations."
        )
        if df_hist is None or df_hist.empty:
            st.info("No run history yet. Run 2+ simulations to populate this dashboard.")
        else:
            df = df_hist.copy()
            if "run_index" not in df.columns:
                df["run_index"] = np.arange(1, len(df) + 1)
            df["run_index"] = pd.to_numeric(df["run_index"], errors="coerce")
            df = df.dropna(subset=["run_index"]).sort_values("run_index")
            if df.empty:
                st.info("Run history contains no valid rows to plot.")
            else:
                min_run = int(df["run_index"].min())
                max_run = int(df["run_index"].max())
                if max_run <= min_run:
                    run_range = (min_run, max_run)
                    st.caption("Run range: only one run available.")
                else:
                    default_low = max(min_run, max_run - 30)
                    run_range = st.slider(
                        "Run range",
                        min_value=min_run,
                        max_value=max_run,
                        value=(default_low, max_run),
                        key=f"{base_key}__history__range",
                    )
                df = df[(df["run_index"] >= run_range[0]) & (df["run_index"] <= run_range[1])]
                meta_cols = {"param_model_type", "run_seed", "run_ts", "run_index"}
                metric_cols = [c for c in df.columns if c not in meta_cols]
                numeric_metrics = []
                for c in metric_cols:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() >= 2:
                        numeric_metrics.append(c)
                if not numeric_metrics:
                    st.info("No numeric metrics found in run history.")
                else:
                    for metric_key in sorted(numeric_metrics):
                        with st.expander(_format_label(metric_key), expanded=False):
                            fig = _plot_metric_history(df, metric_key, base_key=base_key, height=320)
                            if fig is None:
                                st.warning("Not enough valid data to render this metric.")
                            else:
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"{base_key}__history__metric__{_keyify(metric_key)}",
                                    config=plotly_cfg,
                                )

    with validation_tab:
        st.markdown(
            "This section compares your run to published reference ranges (literature) and to your own run history so you can claim improvement with evidence."
        )
        pair_map = {
            "avg_growth_rate": ["Kay_2022_GrowthRate_High", "Kay_2022_GrowthRate_Low"],
            "total_network_length": ["AutoAnalysis_TotalLength"],
            "mst_ratio": ["Tero_2010_MST_Ratio"],
            "path_efficiency": ["Tero_2010_Transport_Ratio"],
        }
        pairs = []
        for mk, lks in pair_map.items():
            for lk in lks:
                if lk in LITERATURE_DATA:
                    pairs.append((mk, lk))
        pair_labels = [f"{_format_label(mk)} vs {lk}" for mk, lk in pairs]
        default_picks = []
        for i, (mk, lk) in enumerate(pairs):
            if mk in metrics and lk in LITERATURE_DATA and lk in ["Kay_2022_GrowthRate_High", "AutoAnalysis_TotalLength", "Tero_2010_MST_Ratio"]:
                default_picks.append(pair_labels[i])
        picked = st.multiselect("Literature comparisons", options=pair_labels, default=default_picks)
        normalize = st.toggle("Normalize to literature mean", value=True)
        selected_pairs = []
        for lab in picked:
            idx = pair_labels.index(lab)
            selected_pairs.append(pairs[idx])
        lit_fig = _plot_lit_comparison(metrics, selected_pairs, normalize=normalize)
        if lit_fig is not None:
            st.plotly_chart(lit_fig, use_container_width=True, key=f"{base_key}__validation__lit_compare__norm_{int(bool(normalize))}", config=plotly_cfg)
            st.caption("Error bars show literature variability (±1σ when available).")
        else:
            st.info("No matching literature comparisons are available for the metrics in this run.")

        if df_hist is not None and "avg_growth_rate" in df_hist.columns:
            hist_fig = _plot_run_history(df_hist, "avg_growth_rate")
            if hist_fig is not None:
                st.plotly_chart(hist_fig, use_container_width=True, key=f"{base_key}__validation__runhist__avg_growth_rate", config=plotly_cfg)

        st.divider()
        if tests.get("note"):
            st.info(tests["note"])
        if tests.get("one_sample_t"):
            st.markdown("**One-sample t-test (your runs vs literature mean)**")
            st.dataframe(pd.DataFrame(tests["one_sample_t"]), use_container_width=True)
        if tests.get("anova"):
            st.markdown("**ANOVA (does model choice affect growth rate?)**")
            st.dataframe(pd.DataFrame([tests["anova"]]), use_container_width=True)
        if tests.get("chi_square"):
            st.markdown("**Chi-square (binned growth-rate distribution vs literature normal)**")
            st.dataframe(pd.DataFrame([tests["chi_square"]]), use_container_width=True)

    with technical_tab:
        st.markdown(
            "Technical diagnostics for reproducibility: neural-network configuration, training curves, and genetic-algorithm progress."
        )
        st.write(
            {
                "hidden_layers": cfg.get("hidden_layers"),
                "activation": cfg.get("activation"),
                "dropout": cfg.get("dropout"),
                "optimizer": cfg.get("optimizer_name"),
                "lr": cfg.get("learning_rate"),
                "scheduler": cfg.get("scheduler_name"),
                "seed": summary.get("seed"),
                "deterministic": cfg.get("deterministic"),
            }
        )
        loss_hist = report_data.get("loss_history") or []
        lr_hist = report_data.get("lr_history") or []
        lh = _plot_series(loss_hist, "Training Loss", y_label="Loss")
        if lh is not None:
            st.plotly_chart(lh, use_container_width=True, key=f"{base_key}__technical__loss", config=plotly_cfg)
        lrh = _plot_series([x for x in lr_hist if x is not None], "Learning Rate", y_label="LR")
        if lrh is not None:
            st.plotly_chart(lrh, use_container_width=True, key=f"{base_key}__technical__lr", config=plotly_cfg)
        opt_hist = report_data.get("optimization_history") or []
        oh = _plot_ga_progress(opt_hist, height=320)
        if oh is not None:
            st.plotly_chart(oh, use_container_width=True, key=f"{base_key}__technical__ga_fitness", config=plotly_cfg)

        with st.expander("Full Report Text (downloadable)", expanded=False):
            if report_data.get("raw_markdown"):
                st.code(report_data["raw_markdown"])
            if report_data.get("download_text"):
                st.download_button(
                    "Download report (markdown)",
                    data=report_data["download_text"],
                    file_name="execution_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔬 Simulation Configuration")
    
    # 1. Theme Selection (Restored)
    color_scheme = st.selectbox("🎨 Color Scheme", _theme_map.keys())
    
    # 2. Mode Selection
    mode = st.radio("Operation Mode", ["Manual Simulation", "Neural Optimization"])
    
    st.divider()
    
    # 3. Model Selection (Shared)
    with st.expander("📐 Model Selection", expanded=True):
        model_type = st.radio("Choose Simulation Model", 
                              ["2.5D Surface (Pillar Tops)", 
                               "3D Porous (Channel Diffusion)", 
                               "3D Structured (Channel Flow)"])
    
    st.divider()
    
    with st.expander("🔁 Reproducibility & Logging", expanded=False):
        use_fixed_seed = st.checkbox("Use Fixed Seed", value=False)
        seed_value = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=12345, step=1, disabled=not use_fixed_seed)
        deterministic_torch = st.checkbox("Deterministic Torch Ops", value=True, disabled=not use_fixed_seed)
        
        report_include_weights = st.checkbox("Include Weight Matrices", value=False)
        report_max_elements = st.number_input("Max Tensor Elements", min_value=100, max_value=200000, value=5000, step=100)

    # --- PARAMETER INITIALIZATION ---
    # We define these variables here so they exist for both modes, 
    # but we only show the widgets in "Manual" mode.
    # Default values are set to match the original app.
    
    if mode == "Manual Simulation":
        st.subheader("⚙️ Scaffold Parameters (from CAD)")
        
        # --- NEW CAD-BASED PARAMS ---
        pillar_count = st.slider("Pillar Count (N x N)", 4, 20, 4, 1)
        pillar_size_mm = st.slider("Pillar Size (mm)", 10.0, 50.0, 30.0, 1.0)
        channel_width_mm = st.slider("Channel Width (mm)", 5.0, 30.0, 15.0, 0.5)
        
        # Conditional parameter for channel nodes
        if "3D" in model_type:
            channel_node_size_mm = st.slider("Channel Node Size (mm)", 0.0, 15.0, 5.0, 0.5)
        else:
            channel_node_size_mm = 0.0
        
        # --- SHARED PARAMS ---
        scaffold_stiffness_kPa = st.slider("Stiffness (Young's Modulus, kPa)", 1.0, 50.0, 25.0, 0.5)
        elasticity = st.slider("Elasticity (Unitless)", 0.0, 1.0, 0.5, 0.05) 
        scaffold_density_g_cm3 = st.slider("Scaffold Density (g/cm³)", 1.0, 1.5, 1.2, 0.01)
        
        st.divider()
        st.subheader("🧬 Biological Parameters")
        
        initial_mass_g = st.slider("Initial Slime Mold Mass (g)", 0.1, 2.0, 0.5, 0.1)
        dev_stage = st.selectbox("Developmental Stage", ["Active Plasmodium", "Sclerotium (Reactivated)", "Spore (Germinated)"])
        
        st.divider()
        st.subheader("🧪 Nutrient & Medium")
        
        nutrient_config = st.selectbox("Nutrient Configuration", ["Even Distribution", "Concentrated Pockets"])
        media_depth_mm = st.slider("Medium Depth (mm)", 0.5, 5.0, 2.0, 0.1)
        replenish_freq_hr = st.slider("Replenishment Frequency (hours)", 1, 48, 24, 1) 
    
        with st.expander("🧪 DMEM Components (mM)"): 
            dmem_glucose = st.number_input("Glucose", 0.0, 50.0, 25.0, 1.0)
            dmem_glutamine = st.number_input("Glutamine", 0.0, 50.0, 45.0, 1.0) 
            dmem_pyruvate = st.number_input("Sodium Pyruvate", 0.0, 5.0, 1.0, 0.1) 
    
        with st.expander("🧪 Ion Concentrations (mM)"): 
            ion_na = st.number_input("Na+", 100.0, 200.0, 154.0, 1.0)
            ion_k = st.number_input("K+", 1.0, 10.0, 5.4, 0.1)
            ion_cl = st.number_input("Cl-", 100.0, 200.0, 140.0, 1.0)
            ion_ca = st.number_input("Ca2+", 0.1, 5.0, 1.8, 0.1)
    
        st.divider()
        st.subheader("💡 Optional Parameters")
        
        with st.expander("💡 Light Exposure"): 
            light_type = st.selectbox("Light Type", ["None (Dark)", "Visible (White)", "Red (660nm)", "Blue (470nm)", "UV", "IR"])
            light_lumens = st.slider("Light Intensity (Lumens)", 0, 1000, 0, 50)
    
        with st.expander("🔬 Hydrogel Ratio (Optional)"): 
            gel_g = st.number_input("Gelatin (g)", 0.0, 10.0, 3.0, 0.1)
            dmem_ml = st.number_input("DMEM (ml)", 50.0, 200.0, 100.0, 1.0)
            alg_g = st.number_input("Sodium Alginate (g)", 0.0, 5.0, 1.0, 0.1)
            cacl_g = st.number_input("Calcium Chloride (g)", 0.0, 2.0, 0.3, 0.05)
            
    else: # Neural Optimization Mode
        st.subheader("Optimization Targets")
        st.info("The system will use the Neural Network + Genetic Algorithm to find the best parameters.")

        with st.expander("What the neural network is doing here", expanded=False):
            st.markdown(
                "- The neural network learns a shortcut: it predicts key biology/transport metrics from your scaffold and media settings.\n"
                "- The genetic algorithm generates many candidate scaffolds and uses the neural network to score them fast.\n"
                "- After optimization, the app runs the physics simulation once to verify the best candidate.\n"
                "- The execution report shows the exact network architecture + training/optimization settings used for that run."
            )
        
        opt_generations = st.slider("Generations", 5, 50, 10)
        opt_pop_size = st.slider("Population Size", 10, 100, 20)

        with st.expander("🧠 Neural Network Settings", expanded=False):
            hidden_layers_text = st.text_input("Hidden Layers (comma-separated)", value="64,128,64")
            activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "leaky_relu"], index=0)
            optimizer_name = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=0.001, step=0.0001, format="%.6f")
            weight_decay = st.number_input("L2 Weight Decay", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.6f")
            l1_lambda = st.number_input("L1 Lambda", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.6f")
            dropout = st.slider("Dropout", 0.0, 0.9, 0.2, 0.05)
            use_attention = st.checkbox("Use Attention Block", value=True)
            shuffle_data = st.checkbox("Shuffle Training Data", value=True)

            scheduler_name = st.selectbox("Learning Rate Scheduler", ["none", "steplr", "exponentiallr"], index=0)
            scheduler_params = {}
            if scheduler_name == "steplr":
                scheduler_params["step_size"] = st.number_input("Step Size", min_value=1, max_value=200, value=10, step=1)
                scheduler_params["gamma"] = st.number_input("Gamma", min_value=0.01, max_value=0.999, value=0.5, step=0.01, format="%.3f")
            elif scheduler_name == "exponentiallr":
                scheduler_params["gamma"] = st.number_input("Gamma", min_value=0.01, max_value=0.999, value=0.9, step=0.01, format="%.3f")
        
        nn_config = {
            "input_dim": 17,
            "output_dim": 5,
            "hidden_layers": _parse_hidden_layers(hidden_layers_text),
            "activation": activation,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "dropout": dropout,
            "scheduler_name": scheduler_name,
            "scheduler_params": scheduler_params,
            "use_attention": use_attention,
            "shuffle": shuffle_data,
        }
        
        if not st.session_state.model_manager.is_trained:
            st.warning("⚠️ Neural Network not trained yet.")
            if st.button("Train Network (Synthetic Data)"):
                exec_seed = int(seed_value) if use_fixed_seed else ModelManager.new_seed()
                st.session_state.last_execution_seed = exec_seed
                st.session_state.model_manager = ModelManager(
                    **nn_config,
                    seed=exec_seed,
                    deterministic=bool(use_fixed_seed and deterministic_torch),
                )
                st.session_state.optimizer = GeneticOptimizer(st.session_state.model_manager)
                with st.spinner("Generating synthetic data and training..."):
                    loss = train_initial_model(st.session_state.model_manager)
                st.success(f"Training Complete! Loss: {loss:.4f}")
                raw_report = st.session_state.model_manager.build_report(
                    include_weights=bool(report_include_weights),
                    max_tensor_elements=int(report_max_elements),
                    extra_metrics={"training_loss": float(loss)},
                    feature_names=list(st.session_state.optimizer.param_names) if hasattr(st.session_state, "optimizer") else None,
                    output_names=["avg_growth_rate", "mean_tortuosity", "permeability_kappa_iso", "mst_ratio", "fractal_dimension"],
                    output_units={
                        "avg_growth_rate": "mm/hr",
                        "mean_tortuosity": "unitless",
                        "permeability_kappa_iso": "proxy",
                        "mst_ratio": "unitless",
                        "fractal_dimension": "unitless",
                    },
                )
                st.session_state.last_report = raw_report
                st.session_state.last_report_data = {
                    "summary": {"seed": exec_seed, "mode": "training_only", "model_type": "3D Porous (Channel Diffusion)", "training_loss": float(loss)},
                    "config": dict(getattr(st.session_state.model_manager, "config", {})),
                    "params": {},
                    "metrics": {},
                    "literature": dict(LITERATURE_DATA),
                    "tests": {"note": "Run at least 2 simulations to enable t-test/ANOVA/chi-square comparisons."},
                    "loss_history": list(getattr(st.session_state.model_manager, "loss_history", [])),
                    "lr_history": list(getattr(st.session_state.model_manager, "lr_history", [])),
                    "run_history": list(st.session_state.run_history),
                    "optimization_history": list(st.session_state.last_optimization_history),
                    "raw_markdown": raw_report if report_include_weights else "",
                    "download_text": raw_report,
                }
                st.rerun()

    st.divider()
    run_button = st.button("▶️ Execute", type="primary", use_container_width=True)

if mode == "Neural Optimization":
    if st.session_state.nn_config != nn_config:
        st.session_state.nn_config = nn_config
        init_seed = int(seed_value) if use_fixed_seed else ModelManager.new_seed()
        st.session_state.model_manager = ModelManager(
            **nn_config,
            seed=init_seed,
            deterministic=bool(use_fixed_seed and deterministic_torch),
        )
        st.session_state.optimizer = GeneticOptimizer(st.session_state.model_manager)

# --- THEME INJECTION (Restored) ---
def hex_to_rgba(hex, opacity):
    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

_t = _theme_map.get(color_scheme, _theme_map["Ocean (Blue/Teal)"])
PLOTLY_TEMPLATE = "plotly_dark" if _t["dark"] else "plotly_white"
st.markdown(f"""
<style>
:root {{
    --primary: {_t['primary']}; --secondary: {_t['secondary']}; --accent: {_t['accent']};
    --bg: {_t['bg']}; --text: {_t['text']}; --surface: {_t['surface']};
    --surface-border: {_t['surface_border']}; --sidebar-bg: {_t['sidebar']};
}}
body {{ background-color: var(--bg); color: var(--text); }}
.stApp {{ background-color: var(--bg); }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }}
h1, h2, h3 {{ color: var(--text); }}
h1 {{ font-weight: 700; letter-spacing: 0.2px; }}
h2 {{ font-weight: 600; border-bottom: 2px solid var(--primary); padding-bottom: .4rem; margin-top: 1.25rem; }}
h3 {{ font-weight: 600; margin-top: .75rem; }}
section[data-testid="stSidebar"] {{ background-color: var(--sidebar-bg); border-right: 1px solid {_t['surface_border']}; }}
.streamlit-expanderHeader {{ background-color: {_t['surface']}; border: 1px solid {_t['surface_border']}; border-radius: 6px; font-weight: 600; color: var(--text); }}
div[data-testid="metric-container"] {{ background: var(--surface); border: 1px solid var(--surface-border); padding: 14px 16px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04); }}
.info-box {{ background: linear-gradient(135deg, {_t['surface']} 0%, {_t['bg']} 100%); border-left: 4px solid var(--primary); padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 10px; border: 1px solid var(--surface-border); }}
.success-box {{ background: linear-gradient(135deg, {hex_to_rgba(_t['accent'], 0.05)} 0%, {_t['bg']} 100%); border-left: 4px solid var(--accent); padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 10px; border: 1px solid var(--surface-border); }}
.hero {{ background: linear-gradient(100deg, {hex_to_rgba(_t['primary'], 0.12)}, {hex_to_rgba(_t['accent'], 0.12)}); border: 1px solid var(--surface-border); border-radius: 16px; padding: 22px 24px; margin-bottom: 16px; }}
.hero h1 {{ margin: 0 0 6px 0; color: var(--secondary); }}
.hero p {{ margin: 0; color: var(--text); opacity: 0.8; }}
.stButton>button {{ background: linear-gradient(180deg, {hex_to_rgba(_t['primary'], 0.9)}, {_t['primary']}); color: #fff; border: 1px solid {hex_to_rgba(_t['primary'], 0.9)}; border-radius: 10px; padding: 10px 16px; box-shadow: 0 6px 12px {hex_to_rgba(_t['primary'], 0.18)}; transition: transform .08s ease-in-out, box-shadow .08s ease-in-out; }}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px {hex_to_rgba(_t['primary'], 0.22)}; }}
.stButton>button:focus {{ outline: none; }}
[data-testid="stTabs"] button[role="tab"] {{ background: {_t['surface']}; border: 1px solid var(--surface-border) !important; border-bottom-color: transparent !important; border-radius: 10px 10px 0 0; margin-right: 6px; color: var(--text); opacity: 0.7; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ background: var(--surface); color: var(--primary); opacity: 1; border-bottom: 2px solid var(--primary) !important; margin-bottom: -2px; }}
[data-testid="stTabs"] div[role="tablist"] {{ padding-bottom: 0; border-bottom: 2px solid var(--surface-border); }}
.plotly .legend {{ border-radius: 8px; border: 1px solid var(--surface-border); background-color: var(--surface) !important; }}
</style>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown(f"""
<div class="hero">
  <h1>🦠 3D Physarum Simulation Platform</h1>
  <p>An advanced digital twin for modeling slime mold growth on engineered scaffolds.</p>
  <p style="margin-top:8px;color:{_t['secondary']}; opacity: 0.7;">Configure your full experimental query in the sidebar and run the simulation.</p>
</div>
""", unsafe_allow_html=True)

st.title("🧠 Neural-Guided Slime Mold Simulation")
st.markdown("""
This system uses a **Deep Neural Network with Attention Mechanisms** coupled with a **Genetic Algorithm** 
to generate uniquely optimized scaffold structures that maximize biological growth and connectivity.
""")

if run_button:
    exec_seed = int(seed_value) if use_fixed_seed else ModelManager.new_seed()
    st.session_state.last_execution_seed = exec_seed
    ModelManager.set_seed(exec_seed, deterministic=bool(use_fixed_seed and deterministic_torch))
    if hasattr(st.session_state, "model_manager") and hasattr(st.session_state.model_manager, "config"):
        st.session_state.model_manager.config["seed"] = exec_seed
        st.session_state.model_manager.config["deterministic"] = bool(use_fixed_seed and deterministic_torch)
    if mode == "Manual Simulation":
        # Build params dict manually from the restored widgets
        params = {
            "model_type": model_type,
            "pillar_count": pillar_count,
            "pillar_size_mm": pillar_size_mm,
            "channel_width_mm": channel_width_mm,
            "channel_node_size_mm": channel_node_size_mm,
            "scaffold_stiffness_kPa": scaffold_stiffness_kPa,
            "elasticity": elasticity,
            "scaffold_density_g_cm3": scaffold_density_g_cm3,
            "initial_mass_g": initial_mass_g,
            "dev_stage": dev_stage,
            "nutrient_config": nutrient_config,
            "media_depth_mm": media_depth_mm,
            "replenish_freq_hr": replenish_freq_hr,
            "dmem_glucose": dmem_glucose,
            "dmem_glutamine": dmem_glutamine,
            "dmem_pyruvate": dmem_pyruvate,
            "ion_na": ion_na, "ion_k": ion_k, "ion_cl": ion_cl, "ion_ca": ion_ca,
            "light_type": light_type,
            "light_lumens": light_lumens,
            "gel_ratio": f"{gel_g}g Gel : {dmem_ml}ml DMEM : {alg_g}g Alg : {cacl_g}g CaCl2"
        }
        
        with st.spinner("Running Manual Simulation..."):
            result = run_simulation_logic(params, model_type)
            run_idx = len(st.session_state.run_history) + 1
            run_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.run_history.append(
                {
                    **dict(result.get("metrics", {})),
                    "param_model_type": model_type,
                    "run_seed": exec_seed,
                    "run_index": run_idx,
                    "run_ts": run_ts,
                }
            )
            st.session_state.latest_result = result
            raw_report = st.session_state.model_manager.build_report(
                include_weights=bool(report_include_weights),
                max_tensor_elements=int(report_max_elements),
                extra_metrics={
                    "mode": "manual",
                    "model_type": model_type,
                    "avg_growth_rate": float(result["metrics"].get("avg_growth_rate", 0.0)),
                    "total_network_length": float(result["metrics"].get("total_network_length", 0.0)),
                },
                feature_names=list(st.session_state.optimizer.param_names) if hasattr(st.session_state, "optimizer") else None,
                output_names=["avg_growth_rate", "mean_tortuosity", "permeability_kappa_iso", "mst_ratio", "fractal_dimension"],
                output_units={
                    "avg_growth_rate": "mm/hr",
                    "mean_tortuosity": "unitless",
                    "permeability_kappa_iso": "proxy",
                    "mst_ratio": "unitless",
                    "fractal_dimension": "unitless",
                },
            )
            st.session_state.last_report = raw_report
            st.session_state.last_report_data = {
                "summary": {"seed": exec_seed, "mode": "manual", "model_type": model_type},
                "config": dict(getattr(st.session_state.model_manager, "config", {})),
                "params": dict(result.get("params", {})),
                "metrics": dict(result.get("metrics", {})),
                "literature": dict(LITERATURE_DATA),
                "tests": {"note": "Run at least 2 simulations to enable t-test/ANOVA/chi-square comparisons."},
                "loss_history": list(getattr(st.session_state.model_manager, "loss_history", [])),
                "lr_history": list(getattr(st.session_state.model_manager, "lr_history", [])),
                "run_history": list(st.session_state.run_history),
                "optimization_history": list(st.session_state.last_optimization_history),
                "raw_markdown": raw_report if report_include_weights else "",
                "download_text": raw_report,
            }
            
    else: # Neural Optimization
        if not st.session_state.model_manager.is_trained:
            st.error("Please train the model first!")
        else:
            with st.status("🧬 Evolving optimal scaffold...") as status:
                st.write("Initializing Genetic Algorithm...")
                best_params, best_fitness, history = st.session_state.optimizer.run_optimization(
                    model_type, pop_size=opt_pop_size, generations=opt_generations
                )
                st.session_state.last_optimization_history = history
                st.write(f"Optimization Complete! Best Fitness (Z-Score): {best_fitness:.2f}")
                status.update(label="Optimization Complete!", state="complete")
            
            # Run final verification simulation
            with st.spinner("Verifying with Physics Engine..."):
                best_params = {**best_params, "model_type": model_type}
                result = run_simulation_logic(best_params, model_type)
                run_idx = len(st.session_state.run_history) + 1
                run_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.run_history.append(
                    {
                        **dict(result.get("metrics", {})),
                        "param_model_type": model_type,
                        "run_seed": exec_seed,
                        "run_index": run_idx,
                        "run_ts": run_ts,
                    }
                )
                st.session_state.latest_result = result
                st.success("Verification Simulation Complete")
            
            raw_report = st.session_state.model_manager.build_report(
                include_weights=bool(report_include_weights),
                max_tensor_elements=int(report_max_elements),
                extra_metrics={
                    "mode": "neural_optimization",
                    "model_type": model_type,
                    "ga_generations": int(opt_generations),
                    "ga_population": int(opt_pop_size),
                    "ga_best_fitness": float(best_fitness),
                    "avg_growth_rate": float(result["metrics"].get("avg_growth_rate", 0.0)),
                    "total_network_length": float(result["metrics"].get("total_network_length", 0.0)),
                },
                feature_names=list(st.session_state.optimizer.param_names) if hasattr(st.session_state, "optimizer") else None,
                output_names=["avg_growth_rate", "mean_tortuosity", "permeability_kappa_iso", "mst_ratio", "fractal_dimension"],
                output_units={
                    "avg_growth_rate": "mm/hr",
                    "mean_tortuosity": "unitless",
                    "permeability_kappa_iso": "proxy",
                    "mst_ratio": "unitless",
                    "fractal_dimension": "unitless",
                },
            )
            st.session_state.last_report = raw_report
            st.session_state.last_report_data = {
                "summary": {
                    "seed": exec_seed,
                    "mode": "neural_optimization",
                    "model_type": model_type,
                    "ga_best_fitness": float(best_fitness),
                },
                "config": dict(getattr(st.session_state.model_manager, "config", {})),
                "params": dict(result.get("params", {})),
                "metrics": dict(result.get("metrics", {})),
                "literature": dict(LITERATURE_DATA),
                "tests": {},
                "loss_history": list(getattr(st.session_state.model_manager, "loss_history", [])),
                "lr_history": list(getattr(st.session_state.model_manager, "lr_history", [])),
                "run_history": list(st.session_state.run_history),
                "optimization_history": list(st.session_state.last_optimization_history),
                "raw_markdown": raw_report if report_include_weights else "",
                "download_text": raw_report,
            }

# --- RESULTS DISPLAY ---
if 'latest_result' in st.session_state:
    res = st.session_state.latest_result
    metrics = res['metrics']
    params = res['params']
    
    st.subheader("📊 Optimization Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Growth Rate", f"{metrics['avg_growth_rate']:.2f} mm/hr")
    c2.metric("Network Length", f"{metrics['total_network_length']:.1f} mm")
    c3.metric("Connectivity", f"{metrics['avg_branches_per_node']:.2f}")
    
    if mode == "Neural Optimization" and st.session_state.last_optimization_history:
        st.subheader("🧬 Genetic Algorithm Progress")
        ga_fig = _plot_ga_progress(st.session_state.last_optimization_history, height=360)
        _seed_for_key = st.session_state.last_report_data.get("summary", {}).get("seed") if isinstance(st.session_state.get("last_report_data"), dict) else None
        if ga_fig is not None:
            st.plotly_chart(
                ga_fig,
                use_container_width=True,
                key=f"main__ga_progress__{_keyify(_seed_for_key)}",
                config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
            )
        st.caption("Bars show per-generation improvement; the line shows the running best.")
    else:
        st.info("Manual Simulation Mode: Real-time metrics calculated above.")
    
    st.divider()
    
    st.subheader("🧪 3D Visualization")
    st.subheader("Generated Scaffold Structure")
    growth_data = res['growth_data']
    scaffold_data = res['scaffold_data']
    
    fig = go.Figure()
    
    if params['model_type'] == "2.5D Surface (Pillar Tops)":
        fig.add_trace(go.Surface(
            z=growth_data, colorscale='Greens',
            colorbar=dict(title='Density'), name="Organism/Cells"
        ))
        fig.update_layout(title="2.5D Surface Growth", height=600, template=PLOTLY_TEMPLATE, 
                          scene_camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)))
    else:
        x, y, z = np.mgrid[:growth_data.shape[0], :growth_data.shape[1], :growth_data.shape[2]]
        
        if scaffold_data is not None:
            fig.add_trace(go.Isosurface(
                x=x.flatten(), y=y.flatten(), z=z.flatten(),
                value=scaffold_data.flatten(),
                isomin=0.5, isomax=1.0, opacity=0.2, colorscale='Blues', showscale=False, name="Scaffold"
            ))
        fig.add_trace(go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=growth_data.flatten(),
            isomin=0.3, isomax=0.8, opacity=0.8, colorscale='Greens', colorbar=dict(title='Density'), name="Slime Mold"
        ))
        fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), height=600, template=PLOTLY_TEMPLATE)
        
    _seed_for_key = st.session_state.last_report_data.get("summary", {}).get("seed") if isinstance(st.session_state.get("last_report_data"), dict) else None
    st.plotly_chart(fig, use_container_width=True, key=f"main__3d_viz__{_keyify(_seed_for_key)}")
    
    st.divider()
    
    st.subheader("📈 Statistical Validation (vs. Literature)")
    
    lit_mean = LITERATURE_DATA["Kay_2022_GrowthRate_High"]["mean"]
    sim_val = metrics['avg_growth_rate']
    
    delta = sim_val - lit_mean
    pct_change = (delta / lit_mean) * 100
    
    col_a, col_b = st.columns(2)
    col_a.metric("Literature Mean (Growth)", f"{lit_mean:.2f}")
    col_b.metric("Simulated Growth", f"{sim_val:.2f}", f"{pct_change:.1f}%")
    
    z_score = (sim_val - lit_mean) / LITERATURE_DATA["Kay_2022_GrowthRate_High"]["std"]
    p_val = 1 - stats.norm.cdf(z_score)
    
    st.markdown(f"**Calculated P-Value:** `{p_val:.5f}`")
    
    if p_val < 0.05:
        st.success("✅ **Statistically Significant Improvement:** The generated scaffold promotes growth significantly better than the baseline literature.")
    else:
        st.warning("⚠️ **Not Significant:** The improvement is within the margin of error of the literature baseline.")

    if st.session_state.last_report:
        st.divider()
        _render_execution_report(st.session_state.last_report_data)
