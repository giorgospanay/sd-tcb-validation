import random
import pickle
import matplotlib
matplotlib.use('Agg') 
import glob
import os
import re
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

### removed synthetic data generation for tracking


# campaign_id: corresponds to data file folder. tx: length, ts: stride
def load_real_network(graphs_dir, campaign_id, tx, ts):
    label = f"tx{tx}_ts{ts}"
    folder = os.path.join(graphs_dir, campaign_id, label)
    pkl_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {folder}")

    net = {}
    for path in pkl_files:
        m = re.search(r"_w(\d+)\.pkl$", path)
        idx = int(m.group(1)) if m else len(net)
        with open(path, "rb") as f:
            net[idx] = pickle.load(f)

    print(f"  Loaded {len(net)} windows from {folder}")
    return net


def list_campaigns(graphs_dir):
    return [
        d for d in sorted(os.listdir(graphs_dir))
        if os.path.isdir(os.path.join(graphs_dir, d))
    ]


MAX_WINDOW_SECONDS = 86400 
def list_window_configs(graphs_dir, campaign_id, max_tx=MAX_WINDOW_SECONDS):
    campaign_dir = os.path.join(graphs_dir, campaign_id)
    configs = []
    for name in sorted(os.listdir(campaign_dir)):
        if name == "tx_full":
            continue   # full-campaign baseline handled separately
        m = re.match(r"tx(\d+)_ts(\d+)$", name)
        if m:
            tx = int(m.group(1))
            if tx <= max_tx:
                configs.append((tx, int(m.group(2))))
    return configs


def load_full_campaign_graph(graphs_dir, campaign_id):
    import glob as _glob
    folder = os.path.join(graphs_dir, campaign_id, "tx_full")
    pkls = _glob.glob(os.path.join(folder, "*.pkl"))
    if not pkls:
        return None
    with open(pkls[0], "rb") as f:
        return pickle.load(f)


## ----------------------
## Community detection: flatten graph then run louvain
## ----------------------

# For flattening: union two networkx graphs, sum shared edges
def flatten_layers(g1,g2):
    """Union two nx graphs, summing weights on shared edges."""
    flat = nx.compose(g1, g2)
    shared = {e: g1.edges[e]["weight"] + g2.edges[e]["weight"]
              for e in g1.edges & g2.edges}
    nx.set_edge_attributes(flat, shared, "weight")
    return flat

# Run Louvain CD for every consecutive pair of layers. 
# Skip windows where Louvain fails (eg temporal windows with no actions
def run_louvain(net):
    comms = {}
    l_list = sorted(net.keys())
    for i in range(len(l_list) - 1):
        flat = flatten_layers(net[l_list[i]], net[l_list[i + 1]])

        if flat.number_of_nodes() == 0 or flat.number_of_edges() == 0:
            comms[i] = []
            continue

        # Remove isolated nodes before running Louvain — they contribute
        # nothing to community structure and can cause singular matrix errors
        isolates = list(nx.isolates(flat))
        if isolates:
            flat.remove_nodes_from(isolates)

        if flat.number_of_nodes() < 2:
            comms[i] = []
            continue

        try:
            result = nx.community.louvain_communities(flat, weight="weight", seed=42)
            comms[i] = [frozenset(c) for c in result]
            print(f"  t={i}: {len(comms[i])} communities detected")
        except Exception as e:
            print(f"  t={i}: Louvain failed ({type(e).__name__}: {e}) — skipping window")
            comms[i] = []

    return comms


## ----------------------
## Community lineage tracking (not used in the workshop edition)
## ----------------------

def jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)

# Track community lineages using Jaccard sim.
def build_lineages(comms, jaccard_threshold=0.1):
    timesteps = sorted(comms.keys())
    # Each community starts as its own lineage
    # lineage_id[t][c_index] = lineage index
    lineage_id = {}
    lineages = []

    # Initialise lineages at t=0
    t0 = timesteps[0]
    lineage_id[t0] = {}
    for ci in range(len(comms[t0])):
        lid = len(lineages)
        lineages.append([(t0, ci)])
        lineage_id[t0][ci] = lid

    jaccard_scores = {}

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        curr_comms = comms[t_curr]
        next_comms = comms[t_next]

        lineage_id[t_next] = {}

        # Build Jaccard matrix: rows=curr, cols=next
        J = np.zeros((len(curr_comms), len(next_comms)))
        for ci, ca in enumerate(curr_comms):
            for cj, cb in enumerate(next_comms):
                J[ci, cj] = jaccard(ca, cb)

        matched_next = set()
        # Greedy matching: best successor for each current community
        for ci in range(len(curr_comms)):
            best_j = J[ci].max() if len(next_comms) > 0 else 0.0
            best_cj = int(J[ci].argmax()) if best_j > 0 else -1

            if best_j >= jaccard_threshold and best_cj not in matched_next:
                # Continue this lineage
                lid = lineage_id[t_curr][ci]
                lineages[lid].append((t_next, best_cj))
                lineage_id[t_next][best_cj] = lid
                matched_next.add(best_cj)
                jaccard_scores[(t_curr, ci)] = best_j
            else:
                # Community dies here
                jaccard_scores[(t_curr, ci)] = float("nan")

        # New communities at t_next that were not matched → new lineages
        for cj in range(len(next_comms)):
            if cj not in lineage_id[t_next]:
                lid = len(lineages)
                lineages.append([(t_next, cj)])
                lineage_id[t_next][cj] = lid

    return lineages, jaccard_scores


def lineage_stability(lineages, jaccard_scores, comms):
    rows = []
    for lid, lin in enumerate(lineages):
        if len(lin) < 2:
            # Born and immediately died — skip or flag
            t0, c0 = lin[0]
            rows.append(dict(
                lineage_id=lid,
                lifespan=1,
                mean_jaccard=float("nan"),
                mean_size=len(comms[t0][c0]),
                size_at_birth=len(comms[t0][c0]),
                size_at_death=len(comms[t0][c0]),
            ))
            continue

        j_vals = [jaccard_scores.get((t, ci), float("nan"))
                  for t, ci in lin[:-1]]  # last step has no outgoing score
        j_vals = [v for v in j_vals if not np.isnan(v)]

        sizes = [len(comms[t][ci]) for t, ci in lin]
        rows.append(dict(
            lineage_id=lid,
            lifespan=len(lin),
            mean_jaccard=np.mean(j_vals) if j_vals else float("nan"),
            mean_size=np.mean(sizes),
            size_at_birth=sizes[0],
            size_at_death=sizes[-1],
        ))
    return pd.DataFrame(rows)



## ----------------------
## Actor matching: IO / precision
## ----------------------

def get_ground_truth_label(node, n_nodes, n_comms):
    comm_size = n_nodes // n_comms
    return int(node) // comm_size


def community_purity_synth(comm_set, n_nodes, n_comms):
    if len(comm_set) == 0:
        return 0.0
    label_counts = defaultdict(int)
    for node in comm_set:
        label_counts[get_ground_truth_label(node, n_nodes, n_comms)] += 1
    return max(label_counts.values()) / len(comm_set)

# get fraction of community members that IO is false
def community_purity_real(comm_set, net, t):
    if len(comm_set) == 0:
        return 0.0
    g = net.get(t)
    io_count = 0
    known = 0
    for node in comm_set:
        val = g.nodes[node].get("is_control") if (g and node in g) else None
        if val is None:
            continue
        known += 1
        if val is False or val == 0:   # is_control=False → IO account
            io_count += 1
    return io_count / known if known > 0 else 0.0

# compute 
def compute_purity_per_step(comms, mode="synth", n_nodes=None, n_comms=None, net=None):
    """
    Compute purity for every (t, community_index).

    Parameters
    ----------
    mode    : "synth" or "real"
    n_nodes, n_comms : required for mode="synth"
    net     : the raw network dict {t: nx.Graph}, required for mode="real"

    Returns
    -------
    dict {(t, c_index): purity_float}
    """
    purity = {}
    for t, comm_list in comms.items():
        for ci, c in enumerate(comm_list):
            if mode == "real":
                purity[(t, ci)] = community_purity_real(c, net, t)
            else:
                purity[(t, ci)] = community_purity_synth(c, n_nodes, n_comms)
    return purity


# Cohesion: _mean_ weight of edges within the community on a given graph
def intra_cohesion(comm_set, graph):
    """
    Mean weight of edges *within* the community on the given graph.
    Uses graph.subgraph() to avoid double-counting edges (NetworkX yields
    each undirected edge twice when iterating via graph.edges(nodelist)).
    Returns 0.0 if there are no intra-community edges.
    """
    sub = graph.subgraph(comm_set)
    weights = [d.get("weight", 1.0) for _, _, d in sub.edges(data=True)]
    return float(np.mean(weights)) if weights else 0.0


def compute_cohesion_per_step(comms, net):
    """
    Compute intra-cohesion for every (t, c_index) on the flattened graph.
    Because comms[t] comes from flat(net[t], net[t+1]) we rebuild it here.
    """
    cohesion = {}
    l_list = sorted(net.keys())
    flat_cache = {}
    for i in range(len(l_list) - 1):
        flat_cache[i] = flatten_layers(net[l_list[i]], net[l_list[i + 1]])

    for t, comm_list in comms.items():
        graph = flat_cache.get(t)
        for ci, c in enumerate(comm_list):
            cohesion[(t, ci)] = intra_cohesion(c, graph) if graph else 0.0
    return cohesion




def save_run_stats(path, label, lineages, jaccard_scores, purity_scores,
                   cohesion_scores, comms, ci_df):
    """
    Serialise all per-run statistics to a JSON file so plots can be
    regenerated or new plots created without re-running the pipeline.

    JSON structure
    --------------
    {
      "label": "...",
      "ci_df": [...],                        # rows of the CI dataframe
      "lineages": [[[t, ci], ...], ...],     # list of lineage paths
      "jaccard_scores": {"t,ci": value},     # transition Jaccard values
      "purity_scores":  {"t,ci": value},
      "cohesion_scores":{"t,ci": value},
      "comm_sizes": {"t": [size, ...]},      # community sizes per timestep
    }
    """
    import json

    def key(t, ci):
        return f"{t},{ci}"

    data = {
        "label": label,
        "ci_df": ci_df.to_dict(orient="records"),
        "lineages": [[[t, ci] for t, ci in lin] for lin in lineages],
        "jaccard_scores":  {key(t, ci): v for (t, ci), v in jaccard_scores.items()},
        "purity_scores":   {key(t, ci): v for (t, ci), v in purity_scores.items()},
        "cohesion_scores": {key(t, ci): v for (t, ci), v in cohesion_scores.items()},
        "comm_sizes": {
            str(t): [len(c) for c in comm_list]
            for t, comm_list in comms.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=True)
    print(f"  Stats saved -> {path}")


def load_run_stats(path):
    import json
    with open(path) as f:
        data = json.load(f)

    def parse_key(k):
        t, ci = k.split(",")
        return int(t), int(ci)

    lineages = [[(t, ci) for t, ci in lin] for lin in data["lineages"]]
    jaccard_scores  = {parse_key(k): v for k, v in data["jaccard_scores"].items()}
    purity_scores   = {parse_key(k): v for k, v in data["purity_scores"].items()}
    cohesion_scores = {parse_key(k): v for k, v in data["cohesion_scores"].items()}
    comm_sizes      = {int(t): sizes for t, sizes in data["comm_sizes"].items()}
    ci_df           = pd.DataFrame(data["ci_df"])
    return (data["label"], lineages, jaccard_scores,
            purity_scores, cohesion_scores, comm_sizes, ci_df)

def compute_io_composition(comms, net, mode="real",
                           n_nodes=None, n_comms=None):
    """
    For every (t, community_index), compute:
      - io_count    : number of confirmed IO nodes in the community
      - total       : total community size
      - precision   : io_count / total  (same as purity for real data)
      - recall      : io_count / total_io_in_dataset
      - f1          : harmonic mean of precision and recall

    Returns
    -------
    composition : dict {(t, ci): dict}
    total_io    : int  — total IO accounts seen across the whole dataset
    """
    # Build the global IO node set once
    if mode == "real":
        all_nodes = set()
        for g in net.values():
            all_nodes.update(g.nodes())
        io_nodes = set()
        for g in net.values():
            for n, d in g.nodes(data=True):
                v = d.get("is_control")
                if v is False or v == 0:
                    io_nodes.add(n)
    else:
        comm_size = n_nodes // n_comms
        # All nodes that belong to any ground-truth IO community
        # (for synthetic we treat all nodes as potentially IO — use purity instead)
        all_nodes = set(range(n_nodes))
        io_nodes  = all_nodes  # not meaningful for synth; recall will be low

    total_io = len(io_nodes)

    composition = {}
    for t, comm_list in comms.items():
        for ci, c in enumerate(comm_list):
            io_in_comm = len(c & io_nodes)
            total      = len(c)
            precision  = io_in_comm / total if total > 0 else 0.0
            recall     = io_in_comm / total_io if total_io > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            composition[(t, ci)] = dict(
                io_count=io_in_comm,
                total=total,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
            )
    return composition, total_io


def lineage_io_summary(lineages, composition, ci_df):
    """
    Aggregate IO composition metrics across each lineage.

    Returns DataFrame with columns:
      lineage_id, mean_precision, mean_recall, mean_f1,
      peak_io_count, peak_total, CI  (joined from ci_df)
    """
    rows = []
    for lid, lin in enumerate(lineages):
        prec_vals = [composition[(t, ci)]["precision"] for t, ci in lin
                     if (t, ci) in composition]
        rec_vals  = [composition[(t, ci)]["recall"]    for t, ci in lin
                     if (t, ci) in composition]
        f1_vals   = [composition[(t, ci)]["f1"]        for t, ci in lin
                     if (t, ci) in composition]
        io_counts = [composition[(t, ci)]["io_count"]  for t, ci in lin
                     if (t, ci) in composition]
        totals    = [composition[(t, ci)]["total"]     for t, ci in lin
                     if (t, ci) in composition]

        if not prec_vals:
            continue

        row = dict(
            lineage_id     = lid,
            mean_precision = round(float(np.mean(prec_vals)), 4),
            mean_recall    = round(float(np.mean(rec_vals)),  4),
            mean_f1        = round(float(np.mean(f1_vals)),   4),
            peak_io_count  = int(max(io_counts)),
            peak_total     = int(max(totals)),
        )
        # join CI score if available (ci_df may be empty / missing columns)
        if ci_df is not None and "lineage_id" in ci_df.columns:
            ci_row = ci_df[ci_df["lineage_id"] == lid]
            row["CI"] = float(ci_row["CI"].values[0]) if len(ci_row) else float("nan")
        else:
            row["CI"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("mean_f1", ascending=False).reset_index(drop=True)
    return df


def log_io_composition(io_df, label, logs_dir, total_io):
    """
    Write a human-readable log of IO composition per lineage.
    Saved as  logs/<label>_io_composition.txt
    """
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"{label}_io_composition.txt")
    lines = [
        f"IO Composition Report — {label}",
        f"Total confirmed IO accounts in dataset: {total_io}",
        f"Lineages reported: {len(io_df)}",
        "",
        f"{'Rank':<5} {'LineageID':<10} {'MeanPrec':>9} {'MeanRec':>8} "
        f"{'MeanF1':>7} {'PeakIO':>7} {'PeakSize':>9} {'CI':>6}",
        "-" * 62,
    ]
    for rank, row in io_df.iterrows():
        lines.append(
            f"{rank+1:<5} {int(row['lineage_id']):<10} "
            f"{row['mean_precision']:>9.3f} {row['mean_recall']:>8.3f} "
            f"{row['mean_f1']:>7.3f} {int(row['peak_io_count']):>7} "
            f"{int(row['peak_total']):>9} {row['CI']:>6.3f}"
        )
    lines += [
        "",
        "Metric definitions:",
        "  Precision = IO accounts in community / community size",
        "  Recall    = IO accounts in community / total IO accounts in dataset",
        "  F1        = harmonic mean of precision and recall",
        "  CI        = Coordination Index (stability*alpha + purity*beta + cohesion*gamma)",
        "",
        "Interpretation:",
        "  High F1 + high CI  -> strong, coherent IO cluster",
        "  High precision, low recall -> small tight cluster, may be one cell of larger op",
        "  Low precision, high recall -> large community capturing most IO but also organic users",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  IO composition log -> {path}")
    return path


def plot_io_composition(io_df, label, top_n=15, plot_path=None):
    """
    Horizontal bar chart: top-N lineages by F1, showing precision/recall/F1
    side by side.  Makes it easy to see which communities best capture IO actors.
    """
    if io_df is None or len(io_df) == 0:
        print("  IO composition plot skipped: no data")
        return

    top = io_df.head(top_n).iloc[::-1]  # reverse so highest is at top
    y   = [f"L{int(r['lineage_id'])} (CI={r['CI']:.2f})"
           for _, r in top.iterrows()]
    x_prec = top["mean_precision"].tolist()
    x_rec  = top["mean_recall"].tolist()
    x_f1   = top["mean_f1"].tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.45 + 1.5)))
    bar_h = 0.25
    positions = np.arange(len(top))

    ax.barh(positions - bar_h, x_prec, height=bar_h, label="Precision", color="#4c78a8")
    ax.barh(positions,          x_rec,  height=bar_h, label="Recall",    color="#f58518")
    ax.barh(positions + bar_h, x_f1,   height=bar_h, label="F1",        color="#54a24b")

    ax.set_yticks(positions)
    ax.set_yticklabels(y, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(f"IO composition — top {top_n} lineages by F1\n{label}", pad=8)
    ax.margins(y=0.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, fontsize=8, borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  COMMUNITY ASSIGNMENT CACHE
# ─────────────────────────────────────────────────────────────────────────────
#
# Community detection is expensive. Once run, assignments are cached as a
# single pickle file under:
#
#   comms/<campaign_id>/<label>/comms.pkl
#
# where label = e.g. "ira_2019_tx3600_ts0".
# The file contains the raw comms dict: {t: [frozenset, ...]}.
#
# If the cache file exists, run_pipeline loads it instead of re-running
# Louvain. Delete the file (or the folder) to force a fresh run.

def comms_cache_path(comms_dir, label):
    return os.path.join(comms_dir, label, "comms.pkl")


def save_comms(comms, comms_dir, label):
    path = comms_cache_path(comms_dir, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(comms, f)
    print(f"  Community assignments cached -> {path}")


def load_comms(comms_dir, label):
    path = comms_cache_path(comms_dir, label)
    with open(path, "rb") as f:
        comms = pickle.load(f)
    n_windows  = len(comms)
    n_total    = sum(len(v) for v in comms.values())
    print(f"  Loaded cached community assignments: "
          f"{n_windows} windows, {n_total} communities total  ({path})")
    return comms


def comms_cached(comms_dir, label):
    return os.path.exists(comms_cache_path(comms_dir, label))

## 
## Community profile summary.
##
## Per window (timestep t):
##   community_count          number of detected communities
##   size_mean/median/std     community size distribution
##   size_rel_mean/median/std same, divided by total campaign node count             
##   io_comm_count            communities containing >=1 confirmed IO nodes
##   io_comm_frac             io_comm_count / community_count
##   mean_io_precision        mean(IO nodes / community size) over IO-containing comms
##   io_recall                (unique IO nodes appearing in any community) / total IO
##
## Per (campaign, window config) — aggregated across timesteps:
##   All of the above, plus:
##   io_lineage_stability     mean Jaccard of IO-heavy communities across consecutive
##                            windows (community is "IO-heavy" if io_precision > 0.5)
##   io_lineage_count         number of distinct IO-heavy lineages
#
# Label coverage warning
# ──────────────────────
# If is_control is missing from most nodes, IO metrics will be unreliable.
# The profiler reports label_coverage = fraction of nodes with known is_control
# so you can see this immediately.
#
# =============================================================================

def get_campaign_io_nodes(net):
    """
    Return (io_nodes, control_nodes, unlabelled_nodes) sets across all windows.
    is_control=False → IO account
    is_control=True  → control/organic
    is_control=None/missing → unlabelled
    """
    io_nodes, ctrl_nodes, unlabelled = set(), set(), set()
    for g in net.values():
        for n, d in g.nodes(data=True):
            v = d.get("is_control")
            if v is False or v == 0:
                io_nodes.add(n)
            elif v is True or v == 1:
                ctrl_nodes.add(n)
            else:
                unlabelled.add(n)
    # nodes seen as both labelled and unlabelled → labelled wins
    unlabelled -= (io_nodes | ctrl_nodes)
    return io_nodes, ctrl_nodes, unlabelled


def profile_window(comms_t, net_t, io_nodes, campaign_node_count):
    """
    Compute per-timestep community metrics for one window graph.

    Parameters
    ----------
    comms_t              : list[frozenset]  communities at this timestep
    net_t                : nx.Graph         the window graph
    io_nodes             : set              all confirmed IO nodes in campaign
    campaign_node_count  : int              total unique nodes across campaign

    Returns
    -------
    dict of scalar metrics
    """
    if not comms_t:
        return None

    sizes = [len(c) for c in comms_t]
    total_nodes = sum(sizes)

    # IO composition per community
    io_prec_vals = []        # precision for IO-containing communities
    io_nodes_seen = set()    # union of IO nodes appearing in any community
    io_comm_count = 0

    for c in comms_t:
        io_in_c = c & io_nodes
        if io_in_c:
            io_comm_count += 1
            io_prec_vals.append(len(io_in_c) / len(c))
            io_nodes_seen |= io_in_c

    io_recall = len(io_nodes_seen) / len(io_nodes) if io_nodes else float("nan")

    # F1 per community, averaged over IO-containing communities
    total_io = len(io_nodes)
    f1_vals = []
    for c in comms_t:
        io_in_c = c & io_nodes
        if not io_in_c:
            continue
        prec = len(io_in_c) / len(c)
        rec  = len(io_in_c) / total_io if total_io else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_vals.append(f1)

    # Cohesion: mean intra-community edge weight across all communities
    # (measures average co-action frequency within detected groups)
    coh_vals = []
    if net_t is not None:
        for c in comms_t:
            sub = net_t.subgraph(c)
            w = [d.get("weight", 1.0) for _, _, d in sub.edges(data=True)]
            if w:
                coh_vals.append(float(np.mean(w)))

    return dict(
        community_count   = len(comms_t),
        size_mean         = float(np.mean(sizes)),
        size_median       = float(np.median(sizes)),
        size_std          = float(np.std(sizes)),
        size_rel_mean     = float(np.mean(sizes))   / campaign_node_count,
        size_rel_median   = float(np.median(sizes)) / campaign_node_count,
        size_rel_std      = float(np.std(sizes))    / campaign_node_count,
        io_comm_count     = io_comm_count,
        io_comm_frac      = io_comm_count / len(comms_t),
        mean_io_precision = float(np.mean(io_prec_vals)) if io_prec_vals else 0.0,
        io_recall         = io_recall,
        mean_f1           = float(np.mean(f1_vals))  if f1_vals  else 0.0,
        mean_cohesion     = float(np.mean(coh_vals)) if coh_vals else 0.0,
    )


def profile_run(comms, net, label, tx, ts,
                io_nodes, ctrl_nodes, unlabelled_nodes,
                lineages, jaccard_scores,
                io_precision_threshold=0.5):
    """
    Aggregate per-timestep profiles into a single run-level summary,
    and compute IO lineage stability.

    Returns
    -------
    summary : dict   — one row for the cross-campaign summary table
    per_window_df : DataFrame — one row per timestep, for time-series plots
    """
    all_nodes = set()
    for g in net.values():
        all_nodes.update(g.nodes())
    campaign_node_count = len(all_nodes)

    total_io = len(io_nodes)
    total_labelled = len(io_nodes) + len(ctrl_nodes)
    total_nodes_seen = total_labelled + len(unlabelled_nodes)
    label_coverage = total_labelled / total_nodes_seen if total_nodes_seen else 0.0

    # ── Per-timestep profiles ────────────────────────────────────────────────
    rows = []
    for t in sorted(comms.keys()):
        p = profile_window(comms[t], net.get(t), io_nodes, campaign_node_count)
        if p is None:
            continue
        p["t"] = t
        p["label"] = label
        p["tx"] = tx
        p["ts"] = ts
        rows.append(p)

    per_window_df = pd.DataFrame(rows)

    if per_window_df.empty:
        return None, per_window_df

    # ── IO lineage stability ─────────────────────────────────────────────────
    # Identify IO-heavy communities at each timestep (precision > threshold)
    io_heavy = {}   # {(t, ci): True/False}
    for t, comm_list in comms.items():
        for ci, c in enumerate(comm_list):
            io_in_c = c & io_nodes
            prec = len(io_in_c) / len(c) if c else 0.0
            io_heavy[(t, ci)] = prec >= io_precision_threshold

    # For each lineage, compute mean Jaccard only over steps where the
    # community was IO-heavy at both ends of the transition
    io_lineage_jaccards = []
    io_lineage_count = 0
    for lin in lineages:
        # Is this lineage IO-heavy at any point?
        if not any(io_heavy.get((t, ci), False) for t, ci in lin):
            continue
        io_lineage_count += 1
        j_vals = [
            jaccard_scores.get((t, ci), float("nan"))
            for t, ci in lin[:-1]
            if io_heavy.get((t, ci), False)
        ]
        j_vals = [v for v in j_vals if not np.isnan(v)]
        if j_vals:
            io_lineage_jaccards.append(float(np.mean(j_vals)))

    io_lineage_stability = (float(np.mean(io_lineage_jaccards))
                            if io_lineage_jaccards else float("nan"))

    # ── Run-level summary ────────────────────────────────────────────────────
    def agg(col):
        if col not in per_window_df.columns:
            return (float("nan"), float("nan"), float("nan"))
        return (float(per_window_df[col].mean()),
                float(per_window_df[col].median()),
                float(per_window_df[col].std()))

    (cc_mean, cc_med, cc_std)         = agg("community_count")
    (sz_mean, sz_med, sz_std)         = agg("size_rel_mean")
    (io_prec_mean, _, __)             = agg("mean_io_precision")
    (io_rec_mean, _, __)              = agg("io_recall")
    (io_cc_mean, io_cc_med, io_cc_std)= agg("io_comm_count")
    (io_cf_mean, _, __)               = agg("io_comm_frac")
    (f1_mean, _, __)                  = agg("mean_f1")
    (coh_mean, _, __)                 = agg("mean_cohesion")

    summary = dict(
        label                 = label,
        tx                    = tx,
        ts                    = ts,
        label_coverage        = round(label_coverage, 3),
        total_io_nodes        = total_io,
        campaign_nodes        = campaign_node_count,
        comm_count_mean       = round(cc_mean, 2),
        comm_count_median     = round(cc_med,  2),
        comm_count_std        = round(cc_std,  2),
        size_rel_mean         = round(sz_mean, 4),
        size_rel_median       = round(sz_med,  4),
        size_rel_std          = round(sz_std,  4),
        mean_io_precision     = round(io_prec_mean, 4),
        mean_io_recall        = round(io_rec_mean,  4),
        mean_f1               = round(f1_mean,      4),
        mean_cohesion         = round(coh_mean,     4),
        io_comm_count_mean    = round(io_cc_mean,   2),
        io_comm_count_median  = round(io_cc_med,    2),
        io_comm_count_std     = round(io_cc_std,    2),
        io_comm_frac_mean     = round(io_cf_mean,   4),
        io_lineage_stability  = round(io_lineage_stability, 4)
                                if not np.isnan(io_lineage_stability) else None,
        io_lineage_count      = io_lineage_count,
    )
    return summary, per_window_df


def save_profile(summary, per_window_df, logs_dir, label):
    """Save run profile: summary as one-line JSON, per-window as CSV."""
    import json
    os.makedirs(logs_dir, exist_ok=True)

    # Per-window CSV (for time-series plots / further analysis)
    csv_path = os.path.join(logs_dir, f"{label}_per_window.csv")
    per_window_df.to_csv(csv_path, index=False)

    # Summary JSON
    json_path = os.path.join(logs_dir, f"{label}_profile.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Profile saved -> {csv_path}, {json_path}")
    return csv_path, json_path



# for each timestep: compute stats for top-k communities in size
def topk_community_stats(comms, net, io_nodes, label, k=10):
    all_nodes = set()
    for g in net.values():
        all_nodes.update(g.nodes())
    campaign_node_count = len(all_nodes)
    total_io = len(io_nodes)

    rows = []
    for t, comm_list in sorted(comms.items()):
        if not comm_list:
            continue
        # sort communities by size descending, take top k
        sorted_comms = sorted(comm_list, key=len, reverse=True)[:k]
        g = net.get(t)
        for rank, c in enumerate(sorted_comms, 1):
            io_in_c  = c & io_nodes
            size     = len(c)
            io_count = len(io_in_c)
            prec     = io_count / size if size > 0 else 0.0
            rec      = io_count / total_io if total_io > 0 else 0.0
            f1       = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            sub      = g.subgraph(c) if g is not None else None
            weights  = ([d.get("weight", 1.0)
                         for _, _, d in sub.edges(data=True)]
                        if sub is not None else [])
            cohesion = float(np.mean(weights)) if weights else 0.0
            rows.append(dict(
                t            = t,
                rank         = rank,
                size         = size,
                size_rel     = size / campaign_node_count,
                io_count     = io_count,
                io_precision = round(prec, 4),
                io_recall    = round(rec,  4),
                f1           = round(f1,   4),
                cohesion     = round(cohesion, 4),
            ))

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    per_window_topk = pd.DataFrame(rows)

    # Aggregate across timesteps: mean ± std per rank
    summary = (per_window_topk
               .groupby("rank")
               .agg(
                   size_mean         = ("size",         "mean"),
                   size_std          = ("size",         "std"),
                   size_rel_mean     = ("size_rel",     "mean"),
                   io_count_mean     = ("io_count",     "mean"),
                   io_precision_mean = ("io_precision", "mean"),
                   io_precision_std  = ("io_precision", "std"),
                   io_recall_mean    = ("io_recall",    "mean"),
                   f1_mean           = ("f1",           "mean"),
                   f1_std            = ("f1",           "std"),
                   cohesion_mean     = ("cohesion",     "mean"),
                   cohesion_std      = ("cohesion",     "std"),
               )
               .reset_index()
               .round(4))

    return per_window_topk, summary


def save_topk_stats(per_window_topk, summary, logs_dir, label, k=10):
    """Save top-k community stats to CSV files."""
    os.makedirs(logs_dir, exist_ok=True)
    pw_path  = os.path.join(logs_dir, f"{label}_topk_per_window.csv")
    sum_path = os.path.join(logs_dir, f"{label}_topk_summary.csv")
    per_window_topk.to_csv(pw_path,  index=False)
    summary.to_csv(sum_path, index=False)
    print(f"  Top-{k} stats -> {sum_path}")

    # Quick readable print of summary
    print(f"  Top-{k} communities (averaged across timesteps):")
    print(f"  {'Rank':>4}  {'Size':>8}  {'IO prec':>8}  {'IO rec':>8}  "
          f"{'F1':>6}  {'Cohesion':>9}")
    print("  " + "-" * 52)
    for _, row in summary.iterrows():
        print(f"  {int(row['rank']):>4}  {row['size_mean']:>8.1f}  "
              f"{row['io_precision_mean']:>8.3f}  {row['io_recall_mean']:>8.3f}  "
              f"{row['f1_mean']:>6.3f}  {row['cohesion_mean']:>9.3f}")
    return pw_path, sum_path



"""
Main call for entire pipeline. Works for both synthetic and real data.

Parameters
----------
mode      : "synth" or "real". Synth deprecated, removed for paper version

Real-data mode
--------------
net       : pre-loaded dict {window_index: nx.Graph} from load_real_network()
            Nodes should carry  is_control (bool)  —  False = IO account

Shared
------
jaccard_threshold : min Jaccard to continue a lineage
min_lifespan      : min steps for a lineage to enter CI ranking
alpha, beta, gamma: CI weights (must sum to 1) -- not used

"""
def run_pipeline(net=None,
                 # synthetic-data args (ignored for real data)
                 edge_path=None, t_split=1, n_nodes=None, n_comms=None, t_max=72,
                 # real-data args
                 mode="synth",
                 # shared args
                 jaccard_threshold=0.1, min_lifespan=3,
                 alpha=0.5, beta=0.3, gamma=0.2,
                 plot_path_prefix=None, logs_dir="logs/",
                 comms_dir="comms/", label=""):
    
    print(f"\n{'='*60}")
    print(f" Pipeline: {label}  (mode={mode})")
    print(f"{'='*60}")

    # 1. Load network (synth only; real data is passed in directly)
    if mode == "synth":
        print("Loading synthetic network …")
        net = load_synth_network(edge_path, t_split=t_split, t_max=t_max)
    else:
        assert net is not None, "Pass net=load_real_network(...) for mode='real'"
        print(f"Using pre-loaded real network ({len(net)} windows)")

    # 2. Community detection — load from cache if available
    if comms_dir and comms_cached(comms_dir, label):
        print("Loading community assignments from cache …")
        comms = load_comms(comms_dir, label)
    else:
        print("Running Louvain …")
        comms = run_louvain(net)
        if comms_dir:
            save_comms(comms, comms_dir, label)

    # 3. Lineage tracking
    if not any(len(v) > 0 for v in comms.values()):
        print("  No communities detected in any window — skipping pipeline.")
        return comms, [], pd.DataFrame()

    print("Building lineages …")
    lineages, jaccard_scores = build_lineages(comms, jaccard_threshold=jaccard_threshold)
    print(f"  -> {len(lineages)} lineages found")

    lin_stats = lineage_stability(lineages, jaccard_scores, comms)
    if len(lin_stats) > 0:
        print(f"  -> median lifespan: {lin_stats['lifespan'].median():.1f} steps")
    else:
        print("  -> no lineages to report")

    # 4. IO matching (purity)
    print("Computing purity against ground truth …")
    if mode == "real":
        purity_scores = compute_purity_per_step(comms, mode="real", net=net)
    else:
        purity_scores = compute_purity_per_step(comms, mode="synth",
                                                n_nodes=n_nodes, n_comms=n_comms)

    # 5. Cohesion
    print("Computing intra-community cohesion …")
    cohesion_scores = compute_cohesion_per_step(comms, net)

    # 6. Coordination Index
    print("Computing Coordination Index …")
    ci_df = coordination_index(
        lineages, jaccard_scores, purity_scores, cohesion_scores, comms,
        alpha=alpha, beta=beta, gamma=gamma, min_lifespan=min_lifespan
    )
    print(f"\nTop 5 lineages by CI:\n{ci_df.head()}\n")

    # 7. IO composition (precision / recall / F1 per lineage)
    print("Computing IO composition …")
    composition, total_io = compute_io_composition(
        comms, net, mode=mode, n_nodes=n_nodes, n_comms=n_comms
    )
    io_df = lineage_io_summary(lineages, composition, ci_df)
    print(f"  -> total IO accounts in dataset: {total_io}")
    if len(io_df):
        print(f"  -> top lineage: precision={io_df.iloc[0]['mean_precision']:.3f}  "
              f"recall={io_df.iloc[0]['mean_recall']:.3f}  "
              f"F1={io_df.iloc[0]['mean_f1']:.3f}")

    # 7b. Community profile (descriptive stats per window + aggregated)
    print("Computing community profile …")
    if mode == "real":
        io_nodes, ctrl_nodes, unlabelled = get_campaign_io_nodes(net)
        cov = (len(io_nodes)+len(ctrl_nodes)) / max(1, len(io_nodes)+len(ctrl_nodes)+len(unlabelled))
        print(f"  -> label coverage: {cov:.1%}  "
              f"({len(io_nodes)} IO, {len(ctrl_nodes)} control, "
              f"{len(unlabelled)} unlabelled)")
    else:
        # synthetic: treat ground-truth community membership as IO label
        comm_size_ = n_nodes // n_comms
        io_nodes   = frozenset(range(n_nodes))   # all nodes are "known"
        ctrl_nodes, unlabelled = set(), set()

    # parse tx/ts from label if not passed directly
    import re as _re
    _m = _re.search(r"_tx(\d+)_ts(\d+)$", label)
    _tx_val = int(_m.group(1)) if _m else 0
    _ts_val = int(_m.group(2)) if _m else 0
    run_summary, per_window_df = profile_run(
        comms, net, label=label, tx=_tx_val, ts=_ts_val,
        io_nodes=io_nodes, ctrl_nodes=ctrl_nodes, unlabelled_nodes=unlabelled,
        lineages=lineages, jaccard_scores=jaccard_scores,
    )

    # 8. Plots
    prefix = plot_path_prefix or ""
    plot_lineage_sizes(
        lineages, comms, ci_df, top_n=10,
        title=f"Top-10 lineage sizes — {label}",
        plot_path=f"{prefix}{label}_lineage_sizes.png" if prefix else None
    )
    plot_ci_distribution(
        ci_df,
        title=f"CI distribution — {label}",
        plot_path=f"{prefix}{label}_ci_dist.png" if prefix else None
    )
    # Find first two timesteps that actually have communities
    valid_ts = [t for t in sorted(comms.keys()) if len(comms[t]) > 0]
    if len(valid_ts) >= 2:
        plot_jaccard_heatmap(
            comms, t_a=valid_ts[0], t_b=valid_ts[1],
            title=f"Jaccard t={valid_ts[0]}->t={valid_ts[1]} — {label}",
            plot_path=f"{prefix}{label}_jaccard_t0_t1.png" if prefix else None
        )
    else:
        print("  Jaccard heatmap skipped: fewer than 2 timesteps with communities")

    if logs_dir and len(io_df):
        log_io_composition(io_df, label, logs_dir, total_io)
    if prefix and len(io_df):
        plot_io_composition(
            io_df, label, top_n=15,
            plot_path=f"{prefix}{label}_io_composition.png"
        )

    # 8b. Save profile + timeseries plot
    if run_summary is not None:
        if logs_dir:
            save_profile(run_summary, per_window_df, logs_dir, label)
        if prefix:
            plot_profile_timeseries(
                per_window_df, label,
                plot_path=f"{prefix}{label}_profile_timeseries.png"
            )

    # 8c. Top-k community stats
    print("Computing top-k community stats …")
    _io_nodes_topk = io_nodes if mode == "real" else set()
    pw_topk, topk_summary = topk_community_stats(
        comms, net, _io_nodes_topk, label=label, k=10
    )
    if logs_dir and not pw_topk.empty:
        save_topk_stats(pw_topk, topk_summary, logs_dir, label, k=10)

    # 9. Save stats for later plot regeneration
    if plot_path_prefix:
        stats_path = f"{plot_path_prefix}{label}_stats.json"
        save_run_stats(stats_path, label, lineages, jaccard_scores,
                       purity_scores, cohesion_scores, comms, ci_df)

    return comms, lineages, ci_df, io_df, run_summary, per_window_df




### Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IO community detection pipeline")
    parser.add_argument("--mode", choices=["synth", "real"], default="synth",
                        help="Data mode: 'synth' (default) or 'real'")
    # real-data args
    parser.add_argument("--graphs-dir", default="graphs",
                        help="Root graphs/ directory from build_repost_graphs.py")
    parser.add_argument("--campaign", default=None,
                        help="Campaign ID to process (default: all campaigns found)")
    parser.add_argument("--tx", type=int, default=None,
                        help="Window size in seconds (default: all available)")
    parser.add_argument("--ts", type=int, default=None,
                        help="Shift in seconds (default: all available)")
    # synth-data args
    parser.add_argument("--synth-path", default="synth/",
                        help="Path to synthetic .edges files")
    parser.add_argument("--n-nodes", type=int, default=3000)
    parser.add_argument("--n-comms", type=int, default=3)
    parser.add_argument("--t-max", type=int, default=72)
    # shared args
    parser.add_argument("--plots", default="plots/",
                        help="Output directory for plots")
    parser.add_argument("--logs",  default="logs/",
                        help="Output directory for logs")
    parser.add_argument("--comms-dir", default="comms/",
                        help="Directory for cached community assignments (default: comms/)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta",  type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--min-lifespan", type=int, default=3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.plots,    exist_ok=True)
    os.makedirs(args.logs,     exist_ok=True)
    os.makedirs(args.comms_dir, exist_ok=True)

    shared = dict(
        jaccard_threshold=args.jaccard_threshold,
        min_lifespan=args.min_lifespan,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        plot_path_prefix=args.plots,
        logs_dir=args.logs,
        comms_dir=args.comms_dir,
    )

    # real datasets:
    if args.mode == "real":
        campaigns = [args.campaign] if args.campaign else list_campaigns(args.graphs_dir)
        print(f"Campaigns to process: {campaigns}")

        all_rows = []
        for campaign_id in campaigns:
            configs = list_window_configs(args.graphs_dir, campaign_id)
            if args.tx is not None:
                configs = [(tx, ts) for tx, ts in configs if tx == args.tx]
            if args.ts is not None:
                configs = [(tx, ts) for tx, ts in configs if ts == args.ts]

            print(f"\nCampaign '{campaign_id}': {len(configs)} window config(s)")
            for tx, ts in configs:
                lbl = f"{campaign_id}_tx{tx}_ts{ts}"
                net = load_real_network(args.graphs_dir, campaign_id, tx, ts)
                comms, lineages, ci_df, io_df, run_summary, per_window_df = run_pipeline(
                    net=net, mode="real", label=lbl, **shared
                )
                if run_summary is not None:
                    # tag with campaign for cross-campaign table
                    run_summary["campaign"] = campaign_id
                    all_rows.append(run_summary)

            # ── Full-campaign baseline (tx_full) ──────────────────────────
            # Run community detection on the single aggregate graph and cache
            # the result. Not included in the cross-window summary table but
            # saved to comms/ and logs/ for later comparison if needed.
            full_g = load_full_campaign_graph(args.graphs_dir, campaign_id)
            if full_g is not None and full_g.number_of_edges() > 0:
                print(f"\n  Running tx_full baseline for {campaign_id} …")
                full_net = {0: full_g}   # single-window net
                lbl_full = f"{campaign_id}_tx_full"
                run_pipeline(
                    net=full_net, mode="real", label=lbl_full,
                    # tx_full has only one window so lineage/stability metrics
                    # are meaningless — use min_lifespan=1 to still get
                    # community stats and IO composition
                    **{**shared, "min_lifespan": 1}
                )
            elif full_g is None:
                print(f"  tx_full not found for {campaign_id} — skipping baseline")

        if all_rows:
            summary_df = pd.DataFrame(all_rows)
            summary_path = os.path.join(args.logs, "profile_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\n\n===== PROFILE SUMMARY =====")
            display_cols = ["campaign","tx","ts","label_coverage",
                            "comm_count_mean","size_rel_mean",
                            "mean_io_precision","mean_io_recall",
                            "io_lineage_stability","io_lineage_count"]
            display_cols = [c for c in display_cols if c in summary_df.columns]
            print(summary_df[display_cols].to_string(index=False))
            print(f"\nFull profile summary -> {summary_path}")

            # Per-campaign window comparison plots
            for cid in summary_df["campaign"].unique():
                plot_window_profile_comparison(
                    summary_df, cid,
                    plot_path=os.path.join(args.plots,
                                           f"{cid}_window_comparison.png")
                )

    else:
        # removed old synthetic data experiments
        pass
