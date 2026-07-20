"""
Usage
-----
python build_repost_graphs.py \\
    --datadir  ../data-infoops \\
    --outdir   graphs/ \\
    --fmt      gpickle
"""

import argparse
import os
import glob
import pickle
from datetime import datetime, timezone
from itertools import combinations

import pandas as pd
import networkx as nx
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Window configuration
# Each entry is (window_size_seconds, shift_seconds).
# shift = 0  → non-overlapping (advances by tx each step)
# shift = tx//4 → 25% shift (75% overlap)
# shift = tx//2 → 50% shift (50% overlap)
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_SIZES = [
    10,      # 10 sec — automatic/bot behaviour threshold
    60,      # 1 min  — automatic/bot behaviour threshold
    900,     # 15 min
    3600,    # 1 h
    21600,   # 6 h
    43200,   # 12 h
    86400,   # 1 day
]

# For each window size, generate shifts at 0%, 25%, 50% of tx.
# shift=0 is treated as non-overlapping (step = tx).
WINDOW_CONFIGS = []
for tx in WINDOW_SIZES:
    for frac in [0, 0.20, 0.50]:
        ts = int(tx * frac)
        WINDOW_CONFIGS.append((tx, ts))


# ──────────────────────────────────────────────────────────────────────────────
# Parquet schema described in Seckin et al zenodo:
#   accountid            : unique anonymised account ID
#   reposted_postid      : ID of the original post that was reposted
#   post_time            : timestamp of the repost
#   is_repost            : boolean — True for reposts
#   is_control           : boolean — True = control, False = IO (ground truth)
#   follower_count       : follower count at collection time
#   following_count      : following count at collection time
#   account_creation_date: account creation date
#   post_language        : language of the post
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"accountid", "reposted_postid", "post_time", "is_repost"}
NODE_ATTR_COLS = [
    "is_control",
    "follower_count",
    "following_count",
    "account_creation_date",
    "post_language",
]


# ──────────────────────────────────────────────────────────────────────────────
# Campaign discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_campaigns(datadir):
    campaigns = {}
    for entry in sorted(os.listdir(datadir)):
        folder = os.path.join(datadir, entry)
        if not os.path.isdir(folder):
            continue
        paths = glob.glob(os.path.join(folder, "*.parquet"))
        if paths:
            campaigns[entry] = sorted(paths)
    if not campaigns:
        raise FileNotFoundError("No campaign folders with parquet files found in " + datadir)
    return campaigns


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_parquets(paths):
    print("  Loading {} file(s)...".format(len(paths)))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print("  {} rows loaded".format(len(df)))
    return df


def save_graph(G, path, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "gpickle":
        with open(path, "wb") as f:
            pickle.dump(G, f)
    elif fmt == "graphml":
        nx.write_graphml(G, path)
    elif fmt == "gexf":
        nx.write_gexf(G, path)
    else:
        raise ValueError("Unknown format: " + fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare(df):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + str(missing))

    df = df[df["is_repost"] == True].copy()
    df = df.dropna(subset=["accountid", "reposted_postid", "post_time"])

    if not pd.api.types.is_datetime64_any_dtype(df["post_time"]):
        df["post_time"] = pd.to_datetime(df["post_time"], utc=True, errors="coerce")
    else:
        if df["post_time"].dt.tz is None:
            df["post_time"] = df["post_time"].dt.tz_localize("UTC")
        else:
            df["post_time"] = df["post_time"].dt.tz_convert("UTC")

    df = df.dropna(subset=["post_time"])
    df["ts"] = df["post_time"].astype("int64") // 10 ** 9

    present_attr = [c for c in NODE_ATTR_COLS if c in df.columns]
    agg = {c: "last" for c in present_attr if c != "post_language"}
    if "post_language" in df.columns:
        agg["post_language"] = lambda x: x.mode().iloc[0] if len(x.mode()) else None

    node_attrs = (
        df.groupby("accountid").agg(agg).reset_index()
        .set_index("accountid").to_dict(orient="index")
    )

    print("  {} repost rows | {} unique accounts".format(len(df), df["accountid"].nunique()))
    return df, node_attrs


# ──────────────────────────────────────────────────────────────────────────────
# Window generator
# shift=0 is treated as non-overlapping: step = tx
# shift>0 is treated as sliding: step = ts
# ──────────────────────────────────────────────────────────────────────────────

def generate_windows(t_min, t_max, tx, ts):
    step = tx if ts == 0 else ts
    start = t_min
    while start < t_max:
        yield start, start + tx
        start += step


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction for a single window
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_for_window(df, node_attrs, t_start, t_end, window_id):
    mask = (df["ts"] >= t_start) & (df["ts"] < t_end)
    window_df = df[mask]

    G = nx.Graph()
    G.graph["window_id"]    = window_id
    G.graph["t_start"]      = t_start
    G.graph["t_end"]        = t_end
    G.graph["t_start_iso"]  = datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat()
    G.graph["t_end_iso"]    = datetime.fromtimestamp(t_end,   tz=timezone.utc).isoformat()
    G.graph["n_rows"]       = int(mask.sum())

    if window_df.empty:
        return G

    groups = (
        window_df.groupby("reposted_postid")["accountid"]
        .apply(lambda x: list(x.unique()))
    )

    edge_weights = {}
    for accounts in groups:
        if len(accounts) < 2:
            continue
        for a, b in combinations(sorted(accounts), 2):
            key = (a, b)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    if not edge_weights:
        return G

    for acc in window_df["accountid"].unique():
        attrs = node_attrs.get(acc, {})
        clean = {}
        for k, v in attrs.items():
            if isinstance(v, (list, dict)):
                clean[k] = v
            elif pd.isna(v):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat()
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        G.add_node(str(acc), **clean)

    for (a, b), w in edge_weights.items():
        G.add_edge(str(a), str(b), weight=w)

    return G


# ──────────────────────────────────────────────────────────────────────────────
# Per-campaign pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_campaign(campaign_id, paths, outdir, fmt):
    ext = {"gpickle": ".pkl", "graphml": ".graphml", "gexf": ".gexf"}[fmt]

    print("\n" + "=" * 60)
    print("Campaign: {}  ({} file(s))".format(campaign_id, len(paths)))
    print("=" * 60)

    df, node_attrs = prepare(load_parquets(paths))

    if df.empty:
        print("  No repost rows — skipping campaign {}".format(campaign_id))
        return []

    t_min = int(df["ts"].min())
    t_max = int(df["ts"].max())
    span_h = (t_max - t_min) / 3600
    print("  Time range : {}  ->  {}".format(
        datetime.fromtimestamp(t_min, tz=timezone.utc).isoformat(),
        datetime.fromtimestamp(t_max, tz=timezone.utc).isoformat(),
    ))
    print("  Span       : {:.1f} hours ({:.1f} days)".format(span_h, span_h / 24))

    summary_rows = []
    campaign_outdir = os.path.join(outdir, campaign_id)

    # Full-campaign single window
    sub = os.path.join(campaign_outdir, "tx_full")
    wid = "tx_full_w0000"
    full_path = os.path.join(sub, wid + ext)
    if os.path.exists(full_path):
        print("\n  [SKIP] full campaign graph already exists")
    else:
        print("\n  [full campaign]")
        G = build_graph_for_window(df, node_attrs, t_min, t_max, wid)
        save_graph(G, full_path, fmt)
        summary_rows.append({
            "campaign":    campaign_id,
            "tx":          t_max - t_min,
            "ts":          0,
            "overlap_pct": 0,
            "window_id":   wid,
            "t_start":     t_min,
            "t_end":       t_max,
            "n_nodes":     G.number_of_nodes(),
            "n_edges":     G.number_of_edges(),
            "n_rows":      G.graph["n_rows"],
        })

    for tx, ts in WINDOW_CONFIGS:

        # Skip window sizes larger than the campaign span — would produce
        # a single window covering everything, which is not informative.
        if tx >= (t_max - t_min):
            print("  [SKIP] tx={}s >= campaign span, skipping".format(tx))
            continue

        label = "tx{}_ts{}".format(tx, ts)
        overlap_pct = 0 if ts == 0 else int((1 - ts / tx) * 100)
        sub = os.path.join(campaign_outdir, label)

        # Skip this config entirely if its output folder already exists
        if os.path.exists(sub):
            print("\n  [SKIP] {} already exists".format(label))
            continue

        windows = list(generate_windows(t_min, t_max, tx, ts))

        print("\n  [{}]  {} windows  ({}% overlap)".format(
            label, len(windows),
            0 if ts == 0 else overlap_pct,
        ))

        for i, (ws, we) in enumerate(tqdm(windows, desc="    " + label)):
            wid = "{}_w{:04d}".format(label, i)
            G   = build_graph_for_window(df, node_attrs, ws, we, wid)
            save_graph(G, os.path.join(sub, wid + ext), fmt)
            summary_rows.append({
                "campaign":    campaign_id,
                "tx":          tx,
                "ts":          ts,
                "overlap_pct": 0 if ts == 0 else overlap_pct,
                "window_id":   wid,
                "t_start":     ws,
                "t_end":       we,
                "n_nodes":     G.number_of_nodes(),
                "n_edges":     G.number_of_edges(),
                "n_rows":      G.graph["n_rows"],
            })

    return summary_rows


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(datadir, outdir, fmt):
    campaigns = discover_campaigns(datadir)
    print("Found {} campaign(s): {}".format(len(campaigns), ", ".join(campaigns.keys())))
    print("\nWindow configs to run per campaign: {}".format(len(WINDOW_CONFIGS)))
    for tx, ts in WINDOW_CONFIGS:
        label = "non-overlapping" if ts == 0 else "{}% overlap".format(int((1 - ts / tx) * 100))
        print("  tx={:>7}s  ts={:>7}s  ({})".format(tx, ts, label))

    all_summary = []
    for campaign_id, paths in campaigns.items():
        rows = run_campaign(campaign_id, paths, outdir, fmt)
        all_summary.extend(rows)
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "windows_summary.csv")

    if not all_summary:
        print("\nNo new campaigns processed.")
    else:
        new_df = pd.DataFrame(all_summary)
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            pd.concat([existing_df, new_df], ignore_index=True).to_csv(summary_path, index=False)
            print("\nAll done. Appended {} rows to summary -> {}".format(len(new_df), summary_path))
        else:
            new_df.to_csv(summary_path, index=False)
            print("\nAll done. Summary -> " + summary_path)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datadir", default="../data-infoops",
                   help="Root folder containing campaign subdirectories (default: ../data-infoops)")
    p.add_argument("--outdir",  default="graphs",
                   help="Root output directory (default: graphs/)")
    p.add_argument("--fmt",     choices=["gpickle", "graphml", "gexf"], default="gpickle",
                   help="Output graph format (default: gpickle)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        datadir=args.datadir,
        outdir=args.outdir,
        fmt=args.fmt,
    )
