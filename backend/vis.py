import numpy as np
import plotly.graph_objects as go

def draw_nn(
    weights,
    activations=None,
    layer_gap=2.0,
    node_gap=1.5,
    show_key=True,
    max_edges=200,
    layout="hierarchical",
    label_mode="auto",
    edge_threshold=0.15,
    show_arrows=True,
    top_paths_k=8,
    show_activation_sign=False,
    view_mode="analysis",
    palette="colorblind",
):
    w1, w2, w3 = weights
    n_in = w1.shape[1]
    n_h1 = w1.shape[0]
    n_h2 = w2.shape[0]
    n_out = 1
    n_max = max(n_in, n_h1, n_h2, n_out)
    def layer_coords_hier(n, idx):
        ys = (np.arange(n) - (n - 1) / 2.0) * node_gap
        xs = np.full(n, idx * layer_gap)
        return xs, ys
    def layer_coords_radial(n, idx):
        r = (idx + 1) * layer_gap
        ang = np.linspace(-np.pi / 2, np.pi / 2, n) if n > 1 else np.array([0.0])
        xs = r * np.cos(ang)
        ys = r * np.sin(ang)
        return xs, ys
    coord_fn = layer_coords_radial if layout == "radial" else layer_coords_hier
    x_in, y_in = coord_fn(n_in, 0)
    x_h1, y_h1 = coord_fn(n_h1, 1)
    x_h2, y_h2 = coord_fn(n_h2, 2)
    x_out, y_out = coord_fn(n_out, 3)
    node_x = np.concatenate([x_in, x_h1, x_h2, x_out])
    node_y = np.concatenate([y_in, y_h1, y_h2, y_out])
    labels = [f"x{i}" for i in range(n_in)] + [f"h1_{j}" for j in range(n_h1)] + [f"h2_{k}" for k in range(n_h2)] + ["y"]
    if activations is not None:
        a_in = np.array(activations.get("in", np.zeros(n_in)))
        a_h1 = np.array(activations.get("h1", np.zeros(n_h1))).flatten()
        a_h2 = np.array(activations.get("h2", np.zeros(n_h2))).flatten()
        a_out = np.array([activations.get("out", 0.0)])
        acts_raw = np.concatenate([a_in, a_h1, a_h2, a_out])
        acts_abs = np.abs(acts_raw)
    else:
        acts_raw = np.zeros_like(node_x)
        acts_abs = np.zeros_like(node_x)
    imp_in = np.sum(np.abs(w1), axis=0) if w1.size else np.zeros(n_in)
    imp_h1 = (np.sum(np.abs(w1), axis=1) if w1.size else np.zeros(n_h1)) + (np.sum(np.abs(w2), axis=0) if w2.size else np.zeros(n_h1))
    imp_h2 = (np.sum(np.abs(w2), axis=1) if w2.size else np.zeros(n_h2)) + (np.abs(w3[0, :]) if w3.size else np.zeros(n_h2))
    imp_out = np.array([np.sum(np.abs(w3)) if w3.size else 0.0])
    imps = np.concatenate([imp_in, imp_h1, imp_h2, imp_out])
    imps = imps / (imps.max() if np.max(imps) > 0 else 1.0)
    if np.max(acts_abs) > 0:
        mix = 0.6 * imps + 0.4 * (acts_abs / np.max(acts_abs))
    else:
        mix = imps
    sizes = 12.0 + 22.0 * np.power(mix, 0.6)
    showscale_flag = bool(np.max(acts_abs) > 0)
    if label_mode == "none":
        text_labels = [""] * len(labels)
    elif label_mode == "all":
        text_labels = labels
    elif label_mode == "important":
        k = int(max(5, min(20, len(labels) // 3)))
        idx_top = np.argsort(mix)[-k:]
        mask = np.zeros(len(labels), dtype=bool)
        mask[idx_top] = True
        text_labels = [lab if mask[i] else "" for i, lab in enumerate(labels)]
    else:
        if n_max > 25:
            k = int(max(6, min(18, len(labels) // 4)))
            idx_top = np.argsort(mix)[-k:]
            mask = np.zeros(len(labels), dtype=bool)
            mask[idx_top] = True
            text_labels = [lab if mask[i] else "" for i, lab in enumerate(labels)]
        else:
            text_labels = labels
    if palette == "colorblind":
        col_i_h1 = (0, 114, 178)
        col_h1_h2 = (213, 94, 0)
        col_h2_out = (0, 158, 115)
        col_top = (255, 150, 0)
    else:
        col_i_h1 = (50, 50, 200)
        col_h1_h2 = (200, 50, 50)
        col_h2_out = (50, 200, 50)
        col_top = (255, 150, 0)
    if show_activation_sign and showscale_flag:
        cscale = "RdBu"
        cmin = -float(np.max(np.abs(acts_raw))) if np.max(np.abs(acts_raw)) > 0 else 0.0
        cmax = float(np.max(np.abs(acts_raw)))
        cvals = acts_raw
    else:
        cscale = "Viridis"
        cmin = None
        cmax = None
        cvals = acts_abs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text_labels, textposition="bottom center",
                             marker=dict(size=sizes, color=cvals, colorscale=cscale, showscale=showscale_flag,
                                         cmin=cmin, cmax=cmax,
                                         colorbar=dict(title="Activation", x=1.02, y=0.5),
                                         line=dict(width=np.where(mix > 0.7, 1.5, 0.8), color="rgba(80,80,80,0.35)")),
                             showlegend=False))
    def add_edges_topk(w, x0s, y0s, x1s, y1s, color, quota):
        wabs = np.abs(w)
        if wabs.size == 0 or quota <= 0:
            return
        idxs = np.unravel_index(np.argsort(wabs, axis=None)[-quota:], w.shape)
        for j, i in zip(idxs[0], idxs[1]):
            a = float(min(1.0, wabs[j, i] / (wabs.max() if wabs.max() > 0 else 1.0)))
            if a < float(edge_threshold):
                continue
            wid = 1.0 + 3.0 * a
            fig.add_shape(type="line", x0=float(x0s[i]), y0=float(y0s[i]), x1=float(x1s[j]), y1=float(y1s[j]),
                          line=dict(color=f"rgba({color[0]},{color[1]},{color[2]},{a})", width=wid))
            if show_arrows and a >= max(float(edge_threshold), 0.6):
                dx = float(x1s[j] - x0s[i])
                dy = float(y1s[j] - y0s[i])
                norm = np.hypot(dx, dy) if np.hypot(dx, dy) > 0 else 1.0
                ux = dx / norm
                uy = dy / norm
                ax = float(x1s[j] - ux * 0.15 * layer_gap)
                ay = float(y1s[j] - uy * 0.15 * layer_gap)
                fig.add_trace(go.Scatter(x=[ax], y=[ay], mode="markers",
                                         marker=dict(symbol="triangle-right", size=8, color=f"rgba({color[0]},{color[1]},{color[2]},1)"),
                                         showlegend=False))
    q1 = int(max_edges // 3)
    q2 = int(max_edges // 3)
    q3 = int(max_edges - q1 - q2)
    if view_mode == "presentation":
        q1 = int(q1 * 0.6)
        q2 = int(q2 * 0.6)
        q3 = int(q3 * 0.6)
    add_edges_topk(w1, x_in, y_in, x_h1, y_h1, col_i_h1, max(1, min(q1, w1.size)))
    add_edges_topk(w2, x_h1, y_h1, x_h2, y_h2, col_h1_h2, max(1, min(q2, w2.size)))
    add_edges_topk(w3, x_h2, y_h2, x_out, y_out, col_h2_out, max(1, min(q3, w3.size)))
    def top_paths(w1, w2, w3, top_k=5):
        paths = []
        if w1.size == 0 or w2.size == 0 or w3.size == 0:
            return paths
        w1a = np.abs(w1)
        w2a = np.abs(w2)
        w3a = np.abs(w3[0, :])
        for i in range(w1.shape[1]):
            best_s = 0.0
            best = None
            for j in range(w1.shape[0]):
                for k in range(w2.shape[0]):
                    s = float(w1a[j, i] * w2a[k, j] * w3a[k])
                    if s > best_s:
                        best_s = s
                        best = (i, j, k, s)
            if best is not None:
                paths.append(best)
        paths.sort(key=lambda t: t[3], reverse=True)
        return paths[:min(top_k, len(paths))]
    paths = top_paths(w1, w2, w3, top_k=int(top_paths_k))
    for i, j, k, s in paths:
        fig.add_shape(type="line", x0=float(x_in[i]), y0=float(y_in[i]), x1=float(x_h1[j]), y1=float(y_h1[j]),
                      line=dict(color=f"rgba({col_top[0]},{col_top[1]},{col_top[2]},0.95)", width=(3.8 if view_mode == "presentation" else 4.5)))
        fig.add_shape(type="line", x0=float(x_h1[j]), y0=float(y_h1[j]), x1=float(x_h2[k]), y1=float(y_h2[k]),
                      line=dict(color=f"rgba({col_top[0]},{col_top[1]},{col_top[2]},0.95)", width=(3.8 if view_mode == "presentation" else 4.5)))
        fig.add_shape(type="line", x0=float(x_h2[k]), y0=float(y_h2[k]), x1=float(x_out[0]), y1=float(y_out[0]),
                      line=dict(color=f"rgba({col_top[0]},{col_top[1]},{col_top[2]},0.95)", width=(3.8 if view_mode == "presentation" else 4.5)))
        fig.add_trace(go.Scatter(x=[x_h2[k]], y=[y_h2[k]], mode="markers", marker=dict(symbol="triangle-right", size=10, color=f"rgba({col_top[0]},{col_top[1]},{col_top[2]},0.95)"), name=None, showlegend=False))
    if show_key:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=f"rgba({col_i_h1[0]},{col_i_h1[1]},{col_i_h1[2]},1)", width=3), name="Input→H1"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=f"rgba({col_h1_h2[0]},{col_h1_h2[1]},{col_h1_h2[2]},1)", width=3), name="H1→H2"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=f"rgba({col_h2_out[0]},{col_h2_out[1]},{col_h2_out[2]},1)", width=3), name="H2→Out"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=f"rgba({col_top[0]},{col_top[1]},{col_top[2]},1)", width=4.0), name="Top path"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=16, color="rgba(100,100,100,1)"), name="Node size = importance"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=16, color="rgba(0,140,255,1)"), name="Node color = activation"))
    fig.update_layout(title=dict(text="Neural Network Pathways", y=0.98, x=0, xanchor='left'),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=True,
                      legend=dict(
                          orientation="v",
                          yanchor="bottom",
                          y=0.02,
                          xanchor="left",
                          x=0.02,
                          bgcolor="#0f172a",
                          bordercolor="rgba(255, 255, 255, 0.1)",
                          borderwidth=1,
                          font=dict(size=11, color="rgba(255, 255, 255, 0.9)")
                      ),
                      width=900, height=int(340 + n_max * 32),
                      margin=dict(t=60, b=20, l=20, r=20))
    fig.add_annotation(x=0, y=1.08, xref="paper", yref="paper",
                       text="Node size = importance; color = activation; edges width ∝ |weight|; orange = strongest input→output paths",
                       showarrow=False, font=dict(size=12), xanchor="left", yanchor="bottom")
    return fig


def _classic_layout(title, height=None, width=None):
    """Returns a layout dict with a timeless, scientific publication style."""
    return dict(
        title=dict(text=title, font=dict(family="Times New Roman", size=20, color="black"), x=0.5, xanchor='center'),
        font=dict(family="Times New Roman", size=12, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="#E0E0E0", zeroline=False, showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="#E0E0E0", zeroline=False, showline=True, linewidth=1, linecolor="black", mirror=True),
        height=height,
        width=width,
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=0.5)
    )

def _prepare_data(run_df, metric_col):
    """Helper to prepare x, y data and axis config for evolution plots."""
    if run_df.empty or metric_col not in run_df.columns:
        return None, None, None
        
    y_data = run_df[metric_col].values.astype(float)
    x_data = run_df.index.values + 1
    
    # Smart Axis Config
    xaxis_config = dict(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)',
        zeroline=False, showline=True, linecolor='rgba(128,128,128,0.15)',
        tickmode='linear', dtick=1,
        title=None
    )
    
    if len(y_data) == 1:
        xaxis_config['range'] = [0.5, 1.5]
        xaxis_config['tickvals'] = [1]
        xaxis_config['ticktext'] = ["Run 1"]
        
    return x_data, y_data, xaxis_config

def draw_metric_area(run_df, metric_col, title, color="#1f77b4", unit=""):
    """Unique Visual 1: Area Chart for Magnitude (e.g., Length)"""
    x, y, xaxis = _prepare_data(run_df, metric_col)
    if x is None: return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', fill='tozeroy',
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color, line=dict(width=2, color='white')),
        name=title
    ))
    
    # Add value label on last point
    fig.add_annotation(
        x=x[-1], y=y[-1], text=f"<b>{y[-1]:.1f}</b>",
        showarrow=True, arrowhead=0, ax=0, ay=-20,
        font=dict(color=color)
    )

    layout = _classic_layout(title)
    layout.update(xaxis=xaxis, height=300, margin=dict(l=20, r=20, t=40, b=20))
    layout['yaxis']['title'] = unit
    fig.update_layout(layout)
    return fig

def draw_metric_bar(run_df, metric_col, title, color="#ff7f0e", unit=""):
    """Unique Visual 2: Bar Chart for Counts (e.g., Junctions)"""
    x, y, xaxis = _prepare_data(run_df, metric_col)
    if x is None: return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=y, 
        marker=dict(color=color, opacity=0.8, line=dict(width=0)),
        name=title, width=0.4 if len(x) < 5 else None # Thinner bars if few points
    ))
    
    # Add value label on top of bars
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='text', text=[f"{v:.0f}" for v in y],
        textposition="top center", textfont=dict(color=color, weight="bold"),
        showlegend=False
    ))

    layout = _classic_layout(title)
    layout.update(xaxis=xaxis, height=300, margin=dict(l=20, r=20, t=40, b=20))
    layout['yaxis']['title'] = unit
    fig.update_layout(layout)
    return fig

def draw_metric_line(run_df, metric_col, title, color="#2ca02c", unit=""):
    """Unique Visual 3: Thick Step/Spline Line for Complexity (e.g., Fractal Dim)"""
    x, y, xaxis = _prepare_data(run_df, metric_col)
    if x is None: return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', 
        line=dict(color=color, width=4, shape='spline', smoothing=0.5), # Smooth curve
        marker=dict(size=10, symbol='diamond', color=color, line=dict(width=2, color='white')),
        name=title
    ))
    
    # Label last point
    fig.add_annotation(
        x=x[-1], y=y[-1], text=f"<b>{y[-1]:.3f}</b>",
        showarrow=True, arrowhead=0, ax=0, ay=-25,
        font=dict(color=color)
    )

    layout = _classic_layout(title)
    layout.update(xaxis=xaxis, height=300, margin=dict(l=20, r=20, t=40, b=20))
    layout['yaxis']['title'] = unit
    fig.update_layout(layout)
    return fig

def draw_metric_lollipop(run_df, metric_col, title, color="#d62728", unit="", baseline=1.0):
    """Unique Visual 4: Lollipop Chart for Deviation (e.g., Tortuosity)"""
    x, y, xaxis = _prepare_data(run_df, metric_col)
    if x is None: return go.Figure()
    
    fig = go.Figure()
    
    # Draw stems
    for xi, yi in zip(x, y):
        fig.add_shape(type="line",
            x0=xi, y0=baseline, x1=xi, y1=yi,
            line=dict(color=color, width=2, dash="dot")
        )
    
    # Draw heads
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        marker=dict(size=12, color=color, symbol='circle'),
        text=[f"{v:.2f}" for v in y], textposition="top center",
        textfont=dict(color=color),
        name=title
    ))
    
    # Baseline line
    fig.add_shape(type="line",
        x0=min(x)-0.5, y0=baseline, x1=max(x)+0.5, y1=baseline,
        line=dict(color="gray", width=1, dash="dash")
    )

    layout = _classic_layout(title)
    layout.update(xaxis=xaxis, height=300, margin=dict(l=20, r=20, t=40, b=20))
    layout['yaxis']['title'] = unit
    # Ensure y-axis starts near 1.0 if it's tortuosity
    min_y = min(min(y), baseline)
    max_y = max(y)
    padding = (max_y - min_y) * 0.2
    layout['yaxis']['range'] = [min_y - padding, max_y + padding]
    
    fig.update_layout(layout)
    return fig

def draw_evolution_plot(
    run_df,
    metric_col,
    title,
    subtitle=None,
    unit="",
    color="#1f77b4",
    ref_band=None,
    ref_name="Reference",
    normalize=False,
    template="plotly_white"
):
    """
    Draws an evolution-aware plot with reference bands, annotations, and smart styling.
    """
    if run_df.empty or metric_col not in run_df.columns:
        return go.Figure()

    y_data = run_df[metric_col].values.astype(float)
    x_data = run_df.index.values + 1 # 1-based index for "Runs"

    # Normalization
    is_normalized = False
    if normalize:
        if np.max(y_data) > np.min(y_data):
            y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
            is_normalized = True
            unit = "Norm (0-1)"
        elif np.max(y_data) != 0:
             y_data = y_data / np.max(y_data)
             is_normalized = True
             unit = "Norm (Max=1)"

    fig = go.Figure()

    # Reference Band (only if not normalized, to keep context clear)
    if ref_band and not is_normalized:
        y0, y1 = ref_band
        # Ensure y0, y1 are within reasonable bounds of the data for visibility
        # But actually, bands are absolute references.
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color, opacity=0.1,
            layer="below", line_width=0,
            annotation_text=ref_name, annotation_position="top left",
            annotation_font_size=10, annotation_font_color=color
        )

    # Main Evolution Trace
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='lines+markers',
        line=dict(color=color, width=3, dash='solid'),
        marker=dict(size=8, color=color, line=dict(width=1.5, color='white'), opacity=1.0),
        name=title,
        showlegend=False
    ))

    # Last Point Annotation
    if len(y_data) > 0:
        last_x = x_data[-1]
        last_y = y_data[-1]
        last_val_fmt = f"{last_y:.2f}" if is_normalized else f"{last_y:.1f}"
        
        # Delta calculation
        delta_str = ""
        if len(y_data) > 1:
            prev_y = y_data[-2]
            if abs(prev_y) > 1e-9:
                pct_change = ((last_y - prev_y) / prev_y) * 100
                symbol = "▲" if pct_change > 0 else "▼"
                delta_str = f" {symbol} {abs(pct_change):.1f}%"
        
        # Determine text color (black or white depending on theme, but here we can force one or use auto)
        # Using the trace color for the text is safe.
        fig.add_annotation(
            x=last_x, y=last_y,
            text=f"Run {last_x}: <b>{last_val_fmt}</b><span style='font-size:10px'>{delta_str}</span>",
            showarrow=True, arrowhead=0, ax=0, ay=-30,
            font=dict(color=color, size=12),
            bgcolor="rgba(255,255,255,0.8)" if "white" in template else "rgba(0,0,0,0.6)",
            bordercolor=color, borderwidth=1, borderpad=4,
            opacity=0.9
        )

    # Layout styling
    full_title = f"{title} <span style='font-size:14px; opacity:0.6'>({unit})</span>" if unit else title
    if subtitle:
        full_title += f"<br><span style='font-size:12px; font-weight:normal; opacity:0.7'><i>{subtitle}</i></span>"
    
    # Grid styling
    grid_color = 'rgba(128,128,128,0.15)'
    if "dark" in template:
        grid_color = 'rgba(255,255,255,0.1)'

    # Handle single point case specially to make it "Self-Explanatory"
    xaxis_config = dict(
        showgrid=True, gridwidth=1, gridcolor=grid_color,
        zeroline=False, showline=True, linecolor=grid_color,
        tickmode='linear', dtick=1, # Force integer ticks
        title=None
    )
    
    if len(y_data) == 1:
        # If only one point, center it and provide more context
        xaxis_config['range'] = [0.5, 1.5]
        xaxis_config['tickvals'] = [1]
        xaxis_config['ticktext'] = ["Run 1"]

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=16)),
        template=template,
        margin=dict(l=20, r=20, t=50, b=30), # Increased bottom margin for subtitles
        height=300,
        xaxis=xaxis_config,
        yaxis=dict(
            showgrid=True, gridwidth=1.5, gridcolor=grid_color, # Stronger horizontal
            zeroline=False, showline=False,
            tickfont=dict(color='rgba(128,128,128,0.8)'),
            title=None
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    
    return fig

def draw_radar_chart(current_metrics, history_df=None, literature_data=None):
    """
    Draws a radar chart comparing current metrics to history max or literature values.
    Classic 'Spider Plot' style.
    """
    keys = ["total_network_length", "avg_growth_rate", "num_junctions", 
            "coverage_fraction", "fractal_dimension", "pillar_adhesion_index"]
    keys = [k for k in keys if k in current_metrics]
    
    if not keys: return go.Figure()

    vals = [current_metrics[k] for k in keys]
    
    # Ranges
    ranges = {}
    for k in keys:
        max_val = current_metrics[k]
        if history_df is not None and not history_df.empty and k in history_df.columns:
            max_val = max(max_val, history_df[k].max())
        if literature_data and k in literature_data:
            max_val = max(max_val, literature_data[k] * 1.2)
        if max_val == 0: max_val = 1.0
        ranges[k] = max_val

    # Normalize
    r_current = [vals[i] / ranges[keys[i]] for i in range(len(keys))]
    
    fig = go.Figure()

    # Current Run - Dark Blue, filled
    fig.add_trace(go.Scatterpolar(
        r=r_current + [r_current[0]], # Close loop
        theta=[k.replace('_', ' ').title() for k in keys] + [keys[0].replace('_', ' ').title()],
        fill='toself',
        name='Current Run',
        line_color='#003366', # Classic Navy
        marker=dict(size=4),
        fillcolor='rgba(0, 51, 102, 0.1)'
    ))
    
    # Literature - Dark Red, dashed
    if literature_data:
        r_lit = []
        for k in keys:
            r_lit.append(literature_data.get(k, 0) / ranges[k])
        
        if any(r_lit):
            fig.add_trace(go.Scatterpolar(
                r=r_lit + [r_lit[0]],
                theta=[k.replace('_', ' ').title() for k in keys] + [keys[0].replace('_', ' ').title()],
                fill='none',
                name='Literature Mean',
                line=dict(color='#8B0000', dash='dash', width=2), # Dark Red
                marker=dict(size=0)
            ))

    layout = _classic_layout("Metric Fingerprint (Normalized)")
    layout.update(polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], showline=False, gridcolor="#E0E0E0"),
        angularaxis=dict(gridcolor="#E0E0E0", linecolor="black")
    ))
    fig.update_layout(layout)
    return fig


def draw_phase_space(history_df, x_col, y_col, current_idx=None):
    """
    Draws a 2D scatter plot (phase space) with trajectory.
    """
    if history_df is None or history_df.empty: return go.Figure()
        
    fig = go.Figure()
    
    # History trajectory - Grey line
    fig.add_trace(go.Scatter(
        x=history_df[x_col],
        y=history_df[y_col],
        mode='lines',
        line=dict(color='#A9A9A9', width=1), # DarkGray
        name='Evolution Trajectory',
        showlegend=False
    ))
    
    # History points - Small circles
    fig.add_trace(go.Scatter(
        x=history_df[x_col],
        y=history_df[y_col],
        mode='markers',
        marker=dict(size=6, color='#696969', opacity=0.6), # DimGray
        name='Historical Runs'
    ))
    
    # Highlight current - Gold Star
    if current_idx is not None and current_idx < len(history_df):
        curr = history_df.iloc[current_idx]
        fig.add_trace(go.Scatter(
            x=[curr[x_col]],
            y=[curr[y_col]],
            mode='markers',
            marker=dict(size=14, color='#DAA520', symbol='star', line=dict(width=1, color='black')), # GoldenRod
            name='Current Run'
        ))

    layout = _classic_layout(f"Phase Portrait: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")
    layout['xaxis']['title'] = x_col.replace('_', ' ').title()
    layout['yaxis']['title'] = y_col.replace('_', ' ').title()
    fig.update_layout(layout)
    return fig


def draw_3d_metric_space(history_df, x_col, y_col, z_col, current_idx=None):
    """
    Draws a 3D scatter plot.
    """
    if history_df is None or history_df.empty: return go.Figure()

    fig = go.Figure()

    # All runs - Color by index (Time) - viridis is fine, but let's do a classic heatmap style
    fig.add_trace(go.Scatter3d(
        x=history_df[x_col],
        y=history_df[y_col],
        z=history_df[z_col],
        mode='markers',
        marker=dict(
            size=4,
            color=history_df.index,
            colorscale='Bluered', # Classic Blue to Red
            opacity=0.7,
            line=dict(width=0)
        ),
        name='History'
    ))

    # Highlight current
    if current_idx is not None and current_idx < len(history_df):
        curr = history_df.iloc[current_idx]
        fig.add_trace(go.Scatter3d(
            x=[curr[x_col]],
            y=[curr[y_col]],
            z=[curr[z_col]],
            mode='markers',
            marker=dict(size=8, color='#DAA520', symbol='diamond', line=dict(width=1, color='black')),
            name='Current Run'
        ))

    layout = _classic_layout(f"3D Parameter Space Analysis")
    layout['scene'] = dict(
        xaxis=dict(title=x_col.replace('_', ' ').title(), backgroundcolor="white", gridcolor="#E0E0E0", showbackground=True),
        yaxis=dict(title=y_col.replace('_', ' ').title(), backgroundcolor="white", gridcolor="#E0E0E0", showbackground=True),
        zaxis=dict(title=z_col.replace('_', ' ').title(), backgroundcolor="white", gridcolor="#E0E0E0", showbackground=True),
    )
    fig.update_layout(layout)
    return fig


def draw_correlation_matrix(history_df):
    """
    Draws a correlation heatmap.
    """
    if history_df is None or len(history_df) < 2: return go.Figure()
    
    df_numeric = history_df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.loc[:, (df_numeric != df_numeric.iloc[0]).any()] 
    if df_numeric.empty: return go.Figure()

    corr = df_numeric.corr()
    
    # Classic Red-Blue Diverging
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[c.replace('_', ' ').title() for c in corr.columns],
        y=[c.replace('_', ' ').title() for c in corr.index],
        colorscale='RdBu',
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(title="Corr", thickness=15)
    ))
    
    layout = _classic_layout("Pearson Correlation Matrix", height=600, width=600)
    fig.update_layout(layout)
    return fig

def draw_parallel_coordinates(history_df, params_keys, metrics_keys):
    """
    Draws a parallel coordinates plot to show flow from Params -> Metrics.
    """
    if history_df is None or history_df.empty: return go.Figure()
    
    # Select a few key columns to avoid clutter
    dims = []
    
    # Add params
    for p in params_keys:
        if p in history_df.columns:
            dims.append(dict(range=[history_df[p].min(), history_df[p].max()],
                             label=p.replace('_', ' ').title(), values=history_df[p]))
            
    # Add metrics
    for m in metrics_keys:
        if m in history_df.columns:
            dims.append(dict(range=[history_df[m].min(), history_df[m].max()],
                             label=m.replace('_', ' ').title(), values=history_df[m]))
            
    if not dims: return go.Figure()
    
    # Color lines by the last metric (usually the 'target' like growth rate)
    last_metric = dims[-1]['values']
    
    fig = go.Figure(data=go.Parcoords(
        line = dict(color = last_metric, colorscale = 'Bluered', showscale = True, colorbar=dict(title="Performance")),
        dimensions = dims
    ))
    
    # Layout adjustments for Parcoords are tricky, keep it simple
    layout = _classic_layout("Parameter Sensitivity & Flow")
    layout['font']['size'] = 10 # Smaller font for dense labels
    fig.update_layout(layout)
    return fig
