"""
plot_results.py
===============

  Q1  Does window length matter? What coordination types are detectable?
      -> plot_window_length()        precision+stability vs tx, by campaign type

  Q2  Does window overlap matter?
      -> plot_overlap_effect()       delta-precision/stability boxplots per tx

  Q3  Are coordinated communities detectable across window settings?
      -> plot_campaign_heatmap()     campaign x tx heatmap of IO precision
      -> plot_precision_recall()     precision vs recall scatter

Usage
-----
  python plot_results.py
  python plot_results.py --summary logs/profile_summary.csv --outdir plots/
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

TX_LABELS = {10: "10 s", 60: "1 min", 900: "15 min", 3600: "1 h",
             21600: "6 h", 43200: "12 h", 86400: "1 day"}

CAMPAIGN_LABELS = {
    "armenia":       "armenia",
    "bangladesh":    "bangladesh",
    "catalonia":     "catalonia",
    "china-1":       "china-1",
    "china-2":       "china-2",
    "ecuador":       "ecuador",
    "egypt_uae":     "egypt_uae",
    "ghana_nigeria": "ghana_nigeria",
    "qatar":         "qatar",
    "spain":         "spain",
    "thailand":      "thailand",
    "uae":           "uae",
    "venezuela-1":   "venezuela-1",
    "venezuela-2":   "venezuela-2",
}

def camp_label(c):
    """Return display label for a campaign name."""
    return CAMPAIGN_LABELS.get(c, c)

import matplotlib as mpl
mpl.rcParams.update({
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize":  13,
})

TYPE_COLORS = {
    "concentrated": "#e45756",
    "diffuse":      "#4c78a8",
    "bursty":       "#f58518",
    "weak":         "#bab0ac",
}
TYPE_LABELS = {
    "concentrated": "Concentrated  (tight botnet / astroturfing)",
    "diffuse":      "Diffuse  (large-scale state-sponsored IO)",
    "bursty":       "Bursty  (event-driven trolling)",
    "weak":         "Weak signal",
}

SAVE_KW   = dict(dpi=300, bbox_inches="tight")
LEGEND_KW = dict(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                 borderaxespad=0, framealpha=0.9, fontsize=8)

STRIDE_TX_FOCUS  = [60, 3600, 21600]
STRIDE_TX_LABELS = {60: "1 min", 3600: "1 h", 21600: "6 h"}
HEATMAP_CAMPAIGNS = ["egypt_uae", "ghana_nigeria", "china-1", "venezuela-1"]


def load(path):
    df = pd.read_csv(path)
    df["tx_label"] = df["tx"].map(TX_LABELS).fillna(df["tx"].astype(str))
    df["io_lineage_stability"] = df["io_lineage_stability"].fillna(0.0)
    return df


def classify_campaign(c_df):
    s = c_df[c_df["ts"] == 0]
    if s.empty:
        return "weak"
    max_prec  = s["mean_io_precision"].max()
    mean_stab = s["io_lineage_stability"].mean()
    max_lin   = s["io_lineage_count"].max()
    fine_prec = s[s["tx"] <= 3600]["mean_io_precision"].mean()
    if max_prec >= 0.70 and mean_stab >= 0.65:
        return "concentrated"
    if max_lin >= 15 and fine_prec >= 0.40:
        return "bursty"
    if 0.30 <= max_prec < 0.70 and mean_stab >= 0.60:
        return "diffuse"
    return "weak"


def add_types(df):
    campaigns = sorted(df["campaign"].unique())
    type_map = {c: classify_campaign(df[df["campaign"] == c]) for c in campaigns}
    df = df.copy()
    df["coord_type"] = df["campaign"].map(type_map)
    return df, type_map

def _campaign_grid(n):
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _finish_grid(fig, axes, n, nrows, ncols, suptitle, path):
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def _load_per_window(logs_dir, campaign, tx, ts=0):
    fname = f"{campaign}_tx{tx}_ts{ts}_per_window.csv"
    path  = os.path.join(logs_dir, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_topk_per_window(logs_dir, campaign, tx, ts=0):
    fname = f"{campaign}_tx{tx}_ts{ts}_topk_per_window.csv"
    path  = os.path.join(logs_dir, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)




# ─────────────────────────────────────────────────────────────────────────────
# Q1: Window length
# ─────────────────────────────────────────────────────────────────────────────

def plot_overlap_effect(df, outdir):
    """
    14-panel grid, one per campaign (alphabetical).
    For each campaign, show precision (solid) and recall (dashed) vs stride
    (0%, 20%, 50%) for three window sizes (1min, 1h, 6h).

    This shows directly:
    - whether overlap changes what you detect (precision lines moving)
    - whether overlap changes how much of the operation you capture (recall)
    - whether the effect differs across window sizes

    Both metrics on the same y-axis [0,1] so they are directly comparable.
    """
    campaigns = sorted(df["campaign"].unique())
    nrows, ncols = _campaign_grid(len(campaigns))

    tx_colors = {60: "#e45756", 3600: "#4c78a8", 21600: "#54a24b"}

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.6, nrows * 2.0),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.30)

    for idx, camp in enumerate(campaigns):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        c_df = df[df["campaign"] == camp]

        for tx in STRIDE_TX_FOCUS:
            tx_df = c_df[c_df["tx"] == tx].sort_values("ts")
            if tx_df.empty:
                continue
            color = tx_colors[tx]
            fracs = [f"{int(round(ts/tx*100))}%" if tx > 0 else "0%"
                     for ts in tx_df["ts"]]
            xs = list(range(len(fracs)))

            # Precision — solid
            ax.plot(xs, tx_df["mean_io_precision"].values,
                    color=color, marker="o", markersize=4,
                    lw=1.8, ls="-", label=f"{STRIDE_TX_LABELS[tx]} prec")
            # Recall — dashed, same colour
            ax.plot(xs, tx_df["mean_io_recall"].values,
                    color=color, marker="s", markersize=3,
                    lw=1.2, ls="--", alpha=0.7,
                    label=f"{STRIDE_TX_LABELS[tx]} rec")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0%", "20%", "50%"], fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(camp_label(camp), fontsize=11, pad=3)
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.4)
        ax.set_axisbelow(True)
        ax.margins(y=0.08)
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Legend: tx colours + solid/dashed for precision/recall
    legend_handles = []
    for tx in STRIDE_TX_FOCUS:
        legend_handles.append(
            plt.Line2D([0], [0], color=tx_colors[tx], lw=2,
                       marker="o", markersize=4,
                       label=f"tx = {STRIDE_TX_LABELS[tx]}"))
    legend_handles.append(
        plt.Line2D([0], [0], color="#555", lw=2, ls="-",
                   label="Precision (solid)"))
    legend_handles.append(
        plt.Line2D([0], [0], color="#555", lw=1.5, ls="--",
                   alpha=0.7, label="Recall (dashed)"))

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=11, borderaxespad=0, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    _finish_grid(fig, axes, len(campaigns), nrows, ncols, "",
                 os.path.join(outdir, "figure2.png"))


def plot_A_conditional_precision(logs_dir, summary_df, outdir):
    """
    14-panel grid, one per campaign (alphabetical).

    For each window size, collect per-window IO precision values ONLY from
    windows where at least one IO-containing community was detected
    (io_comm_count > 0). Show as a boxplot per tx.

    Above each box: activity rate = fraction of windows with any IO community.
    This separates two questions:
      (a) When IO communities DO appear, how pure are they?  → box position
      (b) How often do they appear at all?                   → activity rate label

    A fine window with high box + low activity rate = tight automated clusters
    that fire occasionally.  A coarse window with lower box + high activity
    rate = diffuse mixing of IO with organic users most of the time.
    """
    campaigns = sorted(summary_df["campaign"].unique())
    tx_vals   = sorted(summary_df["tx"].unique())
    nrows, ncols = _campaign_grid(len(campaigns))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 2.2),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.30)

    for idx, camp in enumerate(campaigns):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        boxes, positions, activity_rates = [], [], []
        for xi, tx in enumerate(tx_vals):
            df = _load_per_window(logs_dir, camp, tx, ts=0)
            if df is None or df.empty:
                continue
            active = df[df["io_comm_count"] > 0]["mean_io_precision"].dropna()
            total  = len(df)
            if active.empty:
                continue
            boxes.append(active.values)
            positions.append(xi)
            activity_rates.append(len(active) / total)

        if boxes:
            bp = ax.boxplot(boxes, positions=positions, widths=0.55,
                            patch_artist=True,
                            boxprops=dict(facecolor="#4c78a8", alpha=0.6),
                            medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(lw=1),
                            capprops=dict(lw=1),
                            flierprops=dict(marker="o", markersize=2,
                                            alpha=0.4, markerfacecolor="#4c78a8"))
            # Activity rate annotation above each box
            for xi, rate in zip(positions, activity_rates):
                ax.text(xi, 1.04, f"{rate:.0%}",
                        ha="center", va="bottom", fontsize=8, color="#555")

        ax.set_xticks(range(len(tx_vals)))
        ax.set_xticklabels([TX_LABELS.get(tx, str(tx)) for tx in tx_vals],
                           fontsize=10, rotation=45, ha="right")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(camp_label(camp), fontsize=11, pad=3)
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.4)
        ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel("IO precision", fontsize=12)
        else:
            ax.set_ylabel("")

    _finish_grid(fig, axes, len(campaigns), nrows, ncols,
                 "",
                 os.path.join(outdir, "figure1.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot B — stride effect (fixed y-axis, annotated if flat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_D_timeseries_heatmap(logs_dir, outdir,
                              campaigns=None,
                              tx_vals=None,
                              top_k=5):
    """
    For each campaign × window size, two stacked heatmaps:
      Top row:    IO precision  (YlOrRd, 0–1)
      Bottom row: Cohesion      (Blues,  0–max_cohesion)

    x-axis = window index (time), y-axis = community rank (1 = largest).
    Grey cells = community rank not present in that window.

    Showing both together answers: when IO-heavy communities appear, are
    they also behaviourally intense (high co-action frequency), or just
    compositionally pure (IO accounts present but not co-reposting heavily)?
    """
    if campaigns is None:
        campaigns = HEATMAP_CAMPAIGNS
    if tx_vals is None:
        tx_vals = [60, 3600, 21600]

    ncols = len(tx_vals)
    # Two sub-rows per campaign (precision + cohesion), with a small gap
    # between campaigns.  We use gridspec for fine control.
    n_camps  = len(campaigns)
    n_metric = 2   # precision, cohesion

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(ncols * 4.8, n_camps * n_metric * 1.4 + 0.6))

    # Outer gridspec: one row per campaign, small gap between campaigns
    outer = gridspec.GridSpec(n_camps, 1, figure=fig,
                              hspace=0.35)

    cmap_prec = plt.cm.YlOrRd
    cmap_coh  = plt.cm.Blues
    cmap_prec.set_bad(color="#eeeeee")
    cmap_coh.set_bad(color="#eeeeee")

    # Collect per-campaign cohesion max for normalisation (0–1 per campaign)
    camp_coh_max = {}
    for camp in campaigns:
        cmax = 1.0
        for tx in tx_vals:
            df = _load_topk_per_window(logs_dir, camp, tx, ts=0)
            if df is not None and not df.empty and "cohesion" in df.columns:
                v = df["cohesion"].max()
                if not np.isnan(v) and v > cmax:
                    cmax = v
        camp_coh_max[camp] = cmax

    for ri, camp in enumerate(campaigns):
        # Inner gridspec: n_metric rows × ncols cols for this campaign
        inner = gridspec.GridSpecFromSubplotSpec(
            n_metric, ncols,
            subplot_spec=outer[ri],
            hspace=0.05, wspace=0.18
        )

        for ci, tx in enumerate(tx_vals):
            df = _load_topk_per_window(logs_dir, camp, tx, ts=0)

            ax_prec = fig.add_subplot(inner[0, ci])
            ax_coh  = fig.add_subplot(inner[1, ci])

            if df is None or df.empty:
                for ax in (ax_prec, ax_coh):
                    ax.set_xticks([])
                    ax.set_yticks([])
                continue

            df = df[df["rank"] <= top_k]
            t_vals_sorted = sorted(df["t"].unique())
            n_windows = len(t_vals_sorted)
            t_idx = {t: i for i, t in enumerate(t_vals_sorted)}

            # Build matrices
            mat_prec = np.full((top_k, n_windows), np.nan)
            mat_coh  = np.full((top_k, n_windows), np.nan)
            for _, row in df.iterrows():
                wi = t_idx[row["t"]]
                rk = int(row["rank"]) - 1
                if 0 <= rk < top_k:
                    mat_prec[rk, wi] = row.get("io_precision", np.nan)
                    raw_coh = row.get("cohesion", np.nan)
                    if not np.isnan(raw_coh):
                        raw_coh = raw_coh / camp_coh_max[camp]
                    mat_coh[rk, wi] = raw_coh

            def _draw(ax, mat, cmap, vmin, vmax):
                masked = np.ma.masked_invalid(mat)
                ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax,
                          cmap=cmap, interpolation="nearest")
                ax.set_yticks(range(top_k))
                ax.set_yticklabels([f"#{r+1}" for r in range(top_k)],
                                   fontsize=5.5)
                step = max(1, n_windows // 5)
                ax.set_xticks(range(0, n_windows, step))

            _draw(ax_prec, mat_prec, cmap_prec, 0, 1)
            _draw(ax_coh,  mat_coh,  cmap_coh,  0, 1)

            # x-tick labels only on bottom sub-row
            step = max(1, n_windows // 5)
            ax_prec.set_xticklabels([])
            ax_coh.set_xticklabels(
                [str(t_vals_sorted[i]) for i in range(0, n_windows, step)],
                fontsize=7, rotation=45, ha="right"
            )

            # Column headers on top row of first campaign only
            if ri == 0:
                ax_prec.set_title(TX_LABELS.get(tx, str(tx)),
                                  fontsize=18, pad=8, fontweight="bold")

            # Row label: campaign + metric labels on leftmost column
            if ci == 0:
                ax_prec.set_ylabel(f"{camp_label(camp)}\nPrecision",
                                   fontsize=14, labelpad=4)
                ax_coh.set_ylabel("Cohesion", fontsize=14, labelpad=4)
            else:
                ax_prec.set_ylabel("")
                ax_coh.set_ylabel("")

    # Two colourbars: precision (right) and cohesion (far right)
    fig.subplots_adjust(right=0.85, left=0.08, top=0.96, bottom=0.06)

    cb_prec_ax = fig.add_axes([0.86, 0.15, 0.012, 0.70])
    sm_prec = plt.cm.ScalarMappable(cmap=cmap_prec,
                                     norm=plt.Normalize(0, 1))
    sm_prec.set_array([])
    fig.colorbar(sm_prec, cax=cb_prec_ax, label="IO precision").ax.tick_params(labelsize=9)
    cb_prec_ax.set_ylabel("IO precision", fontsize=11)

    cb_coh_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    sm_coh = plt.cm.ScalarMappable(cmap=cmap_coh,
                                    norm=plt.Normalize(0, 1))
    sm_coh.set_array([])
    fig.colorbar(sm_coh, cax=cb_coh_ax, label="Cohesion (normalised)").ax.tick_params(labelsize=9)
    cb_coh_ax.set_ylabel("Cohesion (normalised)", fontsize=11)

    fig.text(0.47, 0.01, r"Window index $t$",
             ha="center", va="bottom", fontsize=13)
    path = os.path.join(outdir, "figure3.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def make_structural_table(df, outdir):
    """
    LaTeX table: rows = campaigns (alphabetical), column groups = window sizes.
    Within each group: mean community count, mean relative size (%), mean cohesion.
    ts=0 only. Saved as .tex, .csv, .txt.
    """
    sub       = df[df["ts"] == 0].copy()
    tx_vals   = sorted(sub["tx"].unique())
    campaigns = sorted(sub["campaign"].unique())

    struct_cols = [
        ("comm_count_mean", r"$\bar{N}_c$"),
        ("size_rel_mean",   r"$\bar{s}$ (\%)"),
        ("mean_cohesion",   r"$\bar{w}$"),
    ]

    # Build data matrix
    rows = []
    for camp in campaigns:
        c_df = sub[sub["campaign"] == camp]
        row  = {"campaign": camp_label(camp).replace("_", r"\_")}
        for tx in tx_vals:
            lbl  = TX_LABELS.get(tx, str(tx))
            t_df = c_df[c_df["tx"] == tx]
            for col, _ in struct_cols:
                val = t_df[col].iloc[0] if len(t_df) else None
                if val is None:
                    row[f"{lbl}__{col}"] = "---"
                elif col == "size_rel_mean":
                    row[f"{lbl}__{col}"] = f"{float(val)*100:.2f}"
                elif col == "comm_count_mean":
                    row[f"{lbl}__{col}"] = f"{float(val):.1f}"
                else:
                    row[f"{lbl}__{col}"] = f"{float(val):.2f}"
        rows.append(row)

    # CSV
    csv_path = os.path.join(outdir, "structural_table.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    # LaTeX — single rotated table with all three metrics.
    # Rows = campaigns (alphabetical), column groups = window sizes,
    # within each group: N_c, s(%), w.
    # Wrapped in \rotatebox{90}{\begin{minipage}{\textheight}...}
    # so it fits portrait page without pdflscape.
    # Requires: \usepackage{booktabs} in preamble.

    nm = len(struct_cols)
    nt = len(tx_vals)
    col_spec = "l" + (" " + "r" * nm) * nt

    L = []
    A = L.append
    A(r"\begin{table}[htbp]")
    A(r"  \centering")
    A(r"  \caption{Structural community statistics per campaign and window size"
      r" (non-overlapping, $t_s=0$, averaged across all windows)."
      r" $\bar{N}_c$: mean community count;"
      r" $\bar{s}$: mean community size as \% of campaign nodes;"
      r" $\bar{w}$: mean intra-community edge weight.}")
    A(r"  \label{tab:structural_stats}")
    A(r"  \rotatebox{90}{%")
    A(r"  \begin{minipage}{\textheight}")
    A(r"  \small")
    A(f"  \\begin{{tabular}}{{{col_spec}}}")
    A(r"    \toprule")

    # Top header: window size labels spanning nm cols each
    top = "    Campaign"
    for tx in tx_vals:
        lbl = TX_LABELS.get(tx, str(tx))
        top += f" & \\multicolumn{{{nm}}}{{c}}{{{lbl}}}"
    top += r" \\"
    A(top)

    # Cmidrules under each tx group
    cr = "    "
    for gi in range(nt):
        sc = 2 + gi * nm
        ec = sc + nm - 1
        cr += f"\\cmidrule(lr){{{sc}-{ec}}} "
    A(cr.rstrip())

    # Sub-header: metric names
    sub_h = "    Campaign"
    for _ in range(nt):
        for _, mname in struct_cols:
            sub_h += f" & {mname}"
    sub_h += r" \\"
    A(sub_h)
    A(r"    \midrule")

    # Data rows
    for row in rows:
        cells = [row["campaign"]]
        for tx in tx_vals:
            lbl = TX_LABELS.get(tx, str(tx))
            for col, _ in struct_cols:
                cells.append(row.get(f"{lbl}__{col}", "---"))
        A("    " + " & ".join(cells) + r" \\")

    A(r"    \bottomrule")
    A(r"  \end{tabular}")
    A(r"  \end{minipage}")
    A(r"  }% end rotatebox")
    A(r"\end{table}")

    tex_path = os.path.join(outdir, "structural_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  -> {tex_path}")

    txt_path = os.path.join(outdir, "structural_table.txt")
    with open(txt_path, "w") as f:
        f.write(pd.DataFrame(rows).to_string(index=False))
    print(f"  -> {txt_path}")

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary",    default="logs/profile_summary.csv")
    p.add_argument("--outdir",     default="plots/results/")
    p.add_argument("--logs-dir",   default="logs/")
    p.add_argument("--heatmap-campaigns", nargs="+",
                   default=HEATMAP_CAMPAIGNS,
                   help="Campaigns to show in Figure 3 (time-series heatmap)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.summary)
    df, type_map = add_types(df)

    np.random.seed(42)

    print("Generating structural table ...")
    make_structural_table(df, args.outdir)

    print("Generating Figure 1 (conditional precision) ...")
    plot_A_conditional_precision(args.logs_dir, df, args.outdir)

    print("Generating Figure 2 (overlap effect) ...")
    plot_overlap_effect(df, args.outdir)

    print("Generating Figure 3 (time-series heatmap) ...")
    plot_D_timeseries_heatmap(
        args.logs_dir, args.outdir,
        campaigns=args.heatmap_campaigns,
        tx_vals=[60, 3600, 21600],
        top_k=5,
    )

    print(f"\nDone.")
    print("  structural_table.tex/.csv/.txt")
    print("  figure1.png  — conditional IO precision by window size")
    print("  figure2.png  — precision and recall vs window stride")
    print("  figure3.png  — IO precision of top-5 communities over time")


if __name__ == "__main__":
    main()
