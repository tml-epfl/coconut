import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_slide_ready_layered_dag(
    layers,
    edges,
    highlight_path=None,
    layer_titles=None,
    title="Synthetic Reasoning Task: Layered DAG Traversal",
    subtitle="Edges are logical implications; highlighted chain is one valid reasoning path",
    outfile="slide_6_graph.pdf",  # PDF/SVG best for slides
):
    # ---------------- Theme (modern + slide-friendly) ----------------
    # Transparent canvas so the figure can be placed on colored slides.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )

    COLORS = {
        "ink": "#0B1220",
        "muted": "#475569",  # slate-600
        "edge": "#64748B",  # slate-500
        "edge_hi": "#F43F5E",  # rose-500
        "lane_fill": "#F8FAFC",  # slate-50
        "lane_border": "#E2E8F0",  # slate-200
        "node_fill": "#EEF2FF",  # indigo-50
        "node_border": "#1D4ED8",  # blue-700
        "node_hi_fill": "#FFF1F2",  # rose-50
        "node_hi_border": "#F43F5E",
        "title": "#0F172A",
        "accent": "#1E3A8A",
    }

    # ---------------- Layout knobs ----------------
    layer_ids = sorted(layers.keys())
    n_layers = len(layer_ids)
    max_nodes = max(len(layers[i]) for i in layer_ids)

    x_gap = 2.9
    y_gap = 1.08
    lane_pad_x = 1.15
    lane_pad_y = 0.95

    # Node box geometry
    node_w = 2.15
    node_h = 0.70
    rounding = 0.18

    # ---------------- Compute grid positions ----------------
    pos = {}
    order_in_layer = {}
    for li, layer_id in enumerate(layer_ids):
        col_x = li * x_gap
        nodes = layers[layer_id]
        total_h = (len(nodes) - 1) * y_gap
        y_top = total_h / 2.0
        for r, n in enumerate(nodes):
            pos[n] = (col_x, y_top - r * y_gap)
            order_in_layer[n] = r

    # Highlight sets
    hi_nodes = set(highlight_path or [])
    hi_edges = set()
    if highlight_path and len(highlight_path) >= 2:
        hi_edges = set(zip(highlight_path[:-1], highlight_path[1:]))

    # ---------------- Figure sizing ----------------
    fig_w = max(11.5, 2.6 + n_layers * 2.55)
    fig_h = max(6.2, 2.4 + max_nodes * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    # Ensure the canvas itself is transparent (not just saved output).
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # ---------------- Helper: glossy rounded node ----------------
    def add_node(center, text, is_hi=False):
        x, y = center
        fc = COLORS["node_hi_fill"] if is_hi else COLORS["node_fill"]
        ec = COLORS["node_hi_border"] if is_hi else COLORS["node_border"]
        lw = 2.6 if is_hi else 2.0

        # Shadow
        shadow = patches.FancyBboxPatch(
            (x - node_w / 2 + 0.045, y - node_h / 2 - 0.045),
            node_w,
            node_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=0,
            facecolor="black",
            alpha=0.10 if is_hi else 0.08,
            zorder=2,
        )
        ax.add_patch(shadow)

        # Base
        box = patches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w,
            node_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(box)

        # Gloss highlight (top half sheen)
        gloss = patches.FancyBboxPatch(
            (x - node_w / 2 + 0.03, y + 0.02),
            node_w - 0.06,
            node_h / 2 - 0.05,
            boxstyle=f"round,pad=0.02,rounding_size={rounding * 0.95}",
            linewidth=0,
            facecolor="white",
            alpha=0.35 if not is_hi else 0.28,
            zorder=4,
        )
        ax.add_patch(gloss)

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["ink"],
            zorder=5,
            fontweight="semibold" if is_hi else "medium",
        )

    # ---------------- Helper: curved edges with consistent routing ----------------
    def add_edge(u, v, is_hi=False):
        (x0, y0) = pos[u]
        (x1, y1) = pos[v]

        # Make curvature depend on vertical displacement + deterministic sign
        dy = y1 - y0
        sign = 1 if (order_in_layer[u] % 2 == 0) else -1
        rad = sign * (0.10 + 0.06 * min(3.0, abs(dy)))

        color = COLORS["edge_hi"] if is_hi else COLORS["edge"]
        lw = 3.4 if is_hi else 2.0
        alpha = 0.95 if is_hi else 0.42

        arrow = patches.FancyArrowPatch(
            (x0 + node_w / 2 - 0.06, y0),
            (x1 - node_w / 2 + 0.06, y1),
            arrowstyle="-|>",
            mutation_scale=15 if is_hi else 13,
            lw=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1 if not is_hi else 6,
            shrinkA=6,
            shrinkB=6,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(arrow)

    # ---------------- Draw layer “swimlanes” ----------------
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    y_min, y_max = min(ys), max(ys)

    lane_h = (y_max - y_min) + 2 * lane_pad_y
    for li, layer_id in enumerate(layer_ids):
        x_center = li * x_gap
        lane = patches.FancyBboxPatch(
            (x_center - lane_pad_x, y_min - lane_pad_y),
            2 * lane_pad_x,
            lane_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.2,
            edgecolor=COLORS["lane_border"],
            facecolor=COLORS["lane_fill"],
            zorder=0,
        )
        ax.add_patch(lane)

        if layer_titles is None:
            layer_titles = {i: f"Layer {i}" for i in layer_ids}
        ax.text(
            x_center,
            y_max + lane_pad_y - 0.10,
            layer_titles.get(layer_id, f"Layer {layer_id}"),
            ha="center",
            va="top",
            fontsize=13,
            color=COLORS["accent"],
            fontweight="bold",
        )

    # ---------------- Draw edges then nodes ----------------
    for (u, v) in edges:
        add_edge(u, v, is_hi=((u, v) in hi_edges))

    for n, p in pos.items():
        add_node(p, n, is_hi=(n in hi_nodes))

    # ---------------- Title + subtitle ----------------
    ax.text(
        0.5,
        1.03,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color=COLORS["title"],
    )
    ax.text(
        0.5,
        1.005,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=COLORS["muted"],
    )

    # ---------------- Legend (placed outside to avoid overlap) ----------------
    if highlight_path:
        ax.plot(
            [],
            [],
            color=COLORS["edge"],
            lw=2.0,
            alpha=0.42,
            label="Implication edge",
        )
        ax.plot(
            [],
            [],
            color=COLORS["edge_hi"],
            lw=3.4,
            alpha=0.95,
            label="Highlighted reasoning chain",
        )
        leg = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.085),
            ncol=2,
            frameon=True,
            fontsize=11.0,
            handlelength=2.2,
            columnspacing=1.8,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor(COLORS["lane_border"])
        leg.get_frame().set_alpha(0.92)

    # ---------------- Bounds / cleanup ----------------
    margin_x = 1.55
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(y_min - lane_pad_y - 0.25, y_max + lane_pad_y + 0.35)
    ax.axis("off")

    # Reserve bottom space for the legend.
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    layers = {
        0: ["Fact A", "Fact B", "Fact C", "Fact D"],
        1: ["Deduction 1", "Deduction 2", "Deduction 3"],
        2: ["Deduction 4", "Deduction 5"],
        3: ["Conclusion"],
    }

    edges = [
        ("Fact A", "Deduction 1"),
        ("Fact A", "Deduction 2"),
        ("Fact B", "Deduction 2"),
        ("Fact C", "Deduction 1"),
        ("Fact C", "Deduction 3"),
        ("Fact D", "Deduction 2"),
        ("Deduction 1", "Deduction 4"),
        ("Deduction 2", "Deduction 4"),
        ("Deduction 2", "Deduction 5"),
        ("Deduction 3", "Deduction 5"),
        ("Deduction 4", "Conclusion"),
        ("Deduction 5", "Conclusion"),
    ]

    highlight_path = ["Fact A", "Deduction 2", "Deduction 4", "Conclusion"]
    layer_titles = {
        0: "Layer 0 (Facts)",
        1: "Layer 1 (Intermediate)",
        2: "Layer 2 (Intermediate)",
        3: "Layer 3 (Target)",
    }

    for ext in ("pdf", "svg", "png"):
        draw_slide_ready_layered_dag(
            layers,
            edges,
            highlight_path=highlight_path,
            layer_titles=layer_titles,
            outfile=f"notebooks/slide_6_graph.{ext}",
        )


