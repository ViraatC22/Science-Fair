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
                                         line=dict(width=np.where(mix > 0.7, 1.5, 0.8), color="rgba(80,80,80,0.35)"))))
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
    fig.update_layout(title="Neural Network Pathways", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      width=900, height=int(340 + n_max * 32))
    fig.add_annotation(x=0, y=float(max(node_y) + 1.5), xref="x", yref="y",
                       text="Node size = importance; color = activation; edges width ∝ |weight|; orange = strongest input→output paths",
                       showarrow=False, font=dict(size=12))
    return fig
