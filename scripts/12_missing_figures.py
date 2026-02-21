"""
12_missing_figures.py
=====================
Generates four figures that were absent from the initial pipeline:

  results/figures/snf/fig_snf_heatmap.pdf
      — Fused SNF similarity matrix (644×644), reordered by k=2 cluster,
        with cluster annotation bars. The canonical SNF output figure.

  results/figures/snf/fig_modality_comparison.pdf
      — 4-panel: RNA | CNV | Methylation | Fused affinity matrices,
        each reordered by cluster. Shows what SNF does to each modality.

  results/figures/validation/fig_de_cnv.pdf
      — Top differential CNV loci lollipop (top 10 UP_Basal + top 10 UP_Luminal).

  results/figures/validation/fig_de_meth.pdf
      — Top differential methylation probes lollipop
        (top 10 HYPERmeth_Basal + top 10 HYPOMeth_Basal).
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import mygene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES    = ROOT / "results" / "tables"
OUT_SNF   = ROOT / "results" / "figures" / "snf"
OUT_VAL   = ROOT / "results" / "figures" / "validation"

OUT_SNF.mkdir(parents=True, exist_ok=True)
OUT_VAL.mkdir(parents=True, exist_ok=True)

CLUSTER_COLORS = {0: "#4878D0", 1: "#D65F5F"}
CLUSTER_LABELS = {0: "Luminal (C0, n=503)", 1: "Basal (C1, n=141)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── helpers ────────────────────────────────────────────────────────────────────

def cluster_order(cluster_k2: pd.Series, affinity: np.ndarray) -> np.ndarray:
    """
    Return patient indices sorted by:
      1. cluster label (0 first, then 1)
      2. within-cluster: descending row-sum of affinity (most central patients first)
    This produces the cleanest block structure in the heatmap.
    """
    idx = np.arange(len(cluster_k2))
    row_sums = affinity.sum(axis=1)
    order = np.concatenate([
        idx[(cluster_k2 == 0).values][np.argsort(-row_sums[(cluster_k2 == 0).values])],
        idx[(cluster_k2 == 1).values][np.argsort(-row_sums[(cluster_k2 == 1).values])],
    ])
    return order


def plot_affinity_heatmap(ax, matrix: np.ndarray, order: np.ndarray,
                          title: str, vmax: float = None):
    """
    Plot an ordered affinity heatmap on ax.
    Uses imshow with 'hot_r' colormap (white→red, matches SNF convention).
    """
    M = matrix[np.ix_(order, order)]
    if vmax is None:
        vmax = np.percentile(M, 99)
    im = ax.imshow(M, cmap="hot_r", aspect="auto",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def add_cluster_annotation_bar(fig, ax, order: np.ndarray,
                                cluster_k2: pd.Series,
                                side: str = "top", thickness: float = 0.012):
    """
    Add a thin cluster-coloured annotation bar above or to the left of ax.
    side: 'top' or 'left'
    """
    pos = ax.get_position()
    n = len(order)
    colors_array = np.array([CLUSTER_COLORS[cluster_k2.iloc[i]] for i in order])

    if side == "top":
        bar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, thickness])
        bar_ax.imshow(
            colors_array.reshape(1, n, -1).astype(float) /
            np.array([255, 255, 255]),
            aspect="auto"
        )
        # build colour array properly
        rgb = np.array([
            matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster_k2.iloc[i]])
            for i in order
        ]).reshape(1, n, 3)
        bar_ax.imshow(rgb, aspect="auto")
        bar_ax.set_xticks([]); bar_ax.set_yticks([])
        for sp in bar_ax.spines.values():
            sp.set_visible(False)

    elif side == "left":
        bar_ax = fig.add_axes([pos.x0 - thickness, pos.y0, thickness, pos.height])
        rgb = np.array([
            matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster_k2.iloc[i]])
            for i in order
        ]).reshape(n, 1, 3)
        bar_ax.imshow(rgb, aspect="auto")
        bar_ax.set_xticks([]); bar_ax.set_yticks([])
        for sp in bar_ax.spines.values():
            sp.set_visible(False)


# ── load data ─────────────────────────────────────────────────────────────────

log.info("Loading affinity matrices and cluster assignments …")
aff_rna   = np.load(PROCESSED / "affinity_rna.npy")
aff_cnv   = np.load(PROCESSED / "affinity_cnv.npy")
aff_meth  = np.load(PROCESSED / "affinity_meth.npy")
aff_fused = np.load(PROCESSED / "affinity_fused.npy")

clusters  = pd.read_parquet(PROCESSED / "cluster_assignments.parquet")
cluster_k2 = clusters["cluster_k2"]

log.info("  Affinity matrices shape: %s", aff_fused.shape)
log.info("  Cluster k=2 distribution: %s", dict(cluster_k2.value_counts()))

order = cluster_order(cluster_k2, aff_fused)
n_c0  = (cluster_k2 == 0).sum()
n_c1  = (cluster_k2 == 1).sum()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: SNF Fused similarity matrix heatmap
# ══════════════════════════════════════════════════════════════════════════════
log.info("Plotting: fig_snf_heatmap …")

fig, ax = plt.subplots(figsize=(7, 6.5))
plt.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.06)

vmax = np.percentile(aff_fused, 99.5)
M = aff_fused[np.ix_(order, order)]
im = ax.imshow(M, cmap="hot_r", aspect="auto",
               vmin=0, vmax=vmax, interpolation="nearest")

# cluster boundary line
boundary = n_c0
ax.axvline(boundary - 0.5, color="#3A3A3A", linewidth=1.2, alpha=0.9)
ax.axhline(boundary - 0.5, color="#3A3A3A", linewidth=1.2, alpha=0.9)

# cluster labels on axes
ax.set_xticks([n_c0 / 2, n_c0 + n_c1 / 2])
ax.set_xticklabels(["Luminal\n(C0, n=503)", "Basal\n(C1, n=141)"], fontsize=9)
ax.set_yticks([n_c0 / 2, n_c0 + n_c1 / 2])
ax.set_yticklabels(["Luminal\n(C0, n=503)", "Basal\n(C1, n=141)"], fontsize=9)
ax.tick_params(length=0)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Affinity (fused similarity)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# cluster annotation bars
pos = ax.get_position()
n = len(order)

# top bar
top_ax = fig.add_axes([pos.x0, pos.y1 + 0.005, pos.width, 0.014])
rgb_h = np.array([
    matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster_k2.iloc[i]])
    for i in order
]).reshape(1, n, 3)
top_ax.imshow(rgb_h, aspect="auto")
top_ax.set_xticks([]); top_ax.set_yticks([])
for sp in top_ax.spines.values():
    sp.set_visible(False)

# left bar
left_ax = fig.add_axes([pos.x0 - 0.014, pos.y0, 0.014, pos.height])
rgb_v = np.array([
    matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster_k2.iloc[i]])
    for i in order
]).reshape(n, 1, 3)
left_ax.imshow(rgb_v, aspect="auto")
left_ax.set_xticks([]); left_ax.set_yticks([])
for sp in left_ax.spines.values():
    sp.set_visible(False)

# legend
legend_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[0], label="Luminal (C0, n=503)"),
    mpatches.Patch(color=CLUSTER_COLORS[1], label="Basal (C1, n=141)"),
]
ax.legend(handles=legend_patches, loc="lower right",
          fontsize=8, frameon=True, framealpha=0.85)

fig.suptitle(
    "SNF Fused Affinity Matrix (644 × 644 patients)\n"
    "Ordered by k=2 cluster assignment — block structure confirms two stable patient groups",
    fontsize=10, y=0.98
)

out_path = OUT_SNF / "fig_snf_heatmap.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
log.info("  Saved: %s", out_path)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-modality vs fused similarity matrix panel
# ══════════════════════════════════════════════════════════════════════════════
log.info("Plotting: fig_modality_comparison …")

matrices = {
    "RNA-seq Affinity":          aff_rna,
    "CNV Affinity":              aff_cnv,
    "Methylation Affinity":      aff_meth,
    "Fused (SNF) Affinity":      aff_fused,
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
plt.subplots_adjust(wspace=0.06, left=0.04, right=0.96, top=0.85, bottom=0.08)

# Use shared vmax across all four for fair visual comparison
global_vmax = np.percentile(np.concatenate([m.ravel() for m in matrices.values()]), 99.5)

for ax, (title, mat) in zip(axes, matrices.items()):
    # Use modality-specific order for the per-modality panels; fused order for fused
    if "Fused" in title:
        ord_i = order
    else:
        ord_i = cluster_order(cluster_k2, mat)

    M = mat[np.ix_(ord_i, ord_i)]
    im = ax.imshow(M, cmap="hot_r", aspect="auto",
                   vmin=0, vmax=global_vmax, interpolation="nearest")

    # cluster boundary
    ax.axvline(n_c0 - 0.5, color="#3A3A3A", linewidth=0.9, alpha=0.85)
    ax.axhline(n_c0 - 0.5, color="#3A3A3A", linewidth=0.9, alpha=0.85)

    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xticks([]); ax.set_yticks([])

    # thin annotation bar on top
    pos = ax.get_position()
    tbar = fig.add_axes([pos.x0, pos.y1 + 0.003, pos.width, 0.018])
    rgb_t = np.array([
        matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster_k2.iloc[i]])
        for i in ord_i
    ]).reshape(1, n, 3)
    tbar.imshow(rgb_t, aspect="auto")
    tbar.set_xticks([]); tbar.set_yticks([])
    for sp in tbar.spines.values():
        sp.set_visible(False)

# shared colorbar
cbar_ax = fig.add_axes([0.965, 0.10, 0.013, 0.70])
sm = plt.cm.ScalarMappable(cmap="hot_r",
                            norm=plt.Normalize(vmin=0, vmax=global_vmax))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Affinity", fontsize=8)
cb.ax.tick_params(labelsize=7)

legend_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[0], label="Luminal (C0)"),
    mpatches.Patch(color=CLUSTER_COLORS[1], label="Basal (C1)"),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=2, fontsize=8, frameon=True, bbox_to_anchor=(0.50, -0.02))

fig.suptitle(
    "Per-Modality Affinity Matrices vs. SNF Fused Matrix (patients ordered by k=2 cluster)\n"
    "SNF reinforces the block structure present inconsistently across single modalities",
    fontsize=9, y=0.97
)

out_path = OUT_SNF / "fig_modality_comparison.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
log.info("  Saved: %s", out_path)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Top differential CNV loci
# ══════════════════════════════════════════════════════════════════════════════
log.info("Plotting: fig_de_cnv …")

# The pre-saved top_de_cnv.tsv only has 20 rows and all UP_Basal entries share
# one GISTIC2 segment (identical delta). Recompute from the full CNV matrix to
# get segment-aware deduplication across all 5000 features.
cnv_mat = pd.read_parquet(PROCESSED / "cnv_preprocessed.parquet")  # (features x patients)
# align patients with cluster assignments
common_cols = [c for c in cnv_mat.columns if c in cluster_k2.index]
cnv_aligned = cnv_mat[common_cols]
c0_ids = cluster_k2[cluster_k2 == 0].index
c1_ids = cluster_k2[cluster_k2 == 1].index
c0_cols = [c for c in common_cols if c in c0_ids]
c1_cols = [c for c in common_cols if c in c1_ids]

mean_basal   = cnv_aligned[c1_cols].mean(axis=1)
mean_luminal = cnv_aligned[c0_cols].mean(axis=1)
delta_all    = mean_basal - mean_luminal

cnv_full = pd.DataFrame({
    "ensg_id":     delta_all.index,
    "delta_mean":  delta_all.values,
    "mean_basal":  mean_basal.values,
    "mean_luminal": mean_luminal.values,
})

# Load symbol map from the saved table for any known gene symbols
symbol_map = {}
saved_cnv  = pd.read_csv(TABLES / "top_de_cnv.tsv", sep="\t")
for _, row in saved_cnv.iterrows():
    symbol_map[row["ensg_id"]] = row["symbol"]
cnv_full["symbol"] = cnv_full["ensg_id"].map(symbol_map).fillna(cnv_full["ensg_id"])

# Resolve remaining ENSG IDs to gene symbols via mygene
unresolved_mask = cnv_full["symbol"].str.startswith("ENSG")
if unresolved_mask.any():
    mg = mygene.MyGeneInfo()
    ensg_ids = cnv_full.loc[unresolved_mask, "ensg_id"].tolist()
    ensg_bare = [e.split(".")[0] for e in ensg_ids]
    log.info("  Resolving %d ENSG IDs via mygene …", len(ensg_bare))
    results = mg.querymany(ensg_bare, scopes="ensembl.gene",
                           fields="symbol", species="human",
                           returnall=False, verbose=False)
    resolved = {}
    for r in results:
        if "symbol" in r:
            resolved[r["query"]] = r["symbol"]
    # apply back using bare ENSG (strip version)
    def apply_resolved(row):
        if row["symbol"].startswith("ENSG"):
            bare = row["ensg_id"].split(".")[0]
            return resolved.get(bare, row["symbol"])
        return row["symbol"]
    cnv_full["symbol"] = cnv_full.apply(apply_resolved, axis=1)
    n_resolved = (~cnv_full["symbol"].str.startswith("ENSG")).sum()
    log.info("  Symbol resolution: %d / %d features now have gene symbols",
             n_resolved, len(cnv_full))

# Deduplicate: GISTIC2 gives identical scores to genes within the same segment.
# Round delta to 5 decimal places and keep best-annotated gene per unique segment.
cnv_full["delta_round"] = cnv_full["delta_mean"].round(5)

def sym_quality(s):
    if pd.isna(s) or s.startswith("ENSG"): return 0
    if s.startswith("LOC") or s.startswith("LINC"): return 1
    return 2

cnv_full["_sq"] = cnv_full["symbol"].map(sym_quality)
cnv_full = cnv_full.sort_values("_sq", ascending=False)
cnv_dedup = cnv_full.drop_duplicates(subset=["delta_round"]).drop(columns=["_sq"])

# Top 10 per direction
top_basal   = cnv_dedup.nlargest(10, "delta_mean").copy()
top_luminal = cnv_dedup.nsmallest(10, "delta_mean").copy()
top_basal["direction"]   = "UP_Basal"
top_luminal["direction"] = "UP_Luminal"

# Display: prefer symbols, fall back to short ENSG ID
def make_display(row):
    s = row["symbol"]
    if s.startswith("ENSG"):
        return row["ensg_id"].split(".")[0][:14]  # short ENSG, strip version
    return s  # use whatever symbol we have (including LOC/LINC — they are real names)

top_basal["display"]   = top_basal.apply(make_display, axis=1)
top_luminal["display"] = top_luminal.apply(make_display, axis=1)

log.info("  CNV: %d unique Basal segments, %d unique Luminal segments (from %d total deduped)",
         len(top_basal), len(top_luminal), len(cnv_dedup))

plot_df = pd.concat([top_luminal, top_basal]).reset_index(drop=True)
colors  = [CLUSTER_COLORS[1] if d == "UP_Basal" else CLUSTER_COLORS[0]
           for d in plot_df["direction"]]

fig, ax = plt.subplots(figsize=(7, 6))
y_pos = np.arange(len(plot_df))

ax.hlines(y_pos, 0, plot_df["delta_mean"], colors=colors, linewidth=1.5)
ax.scatter(plot_df["delta_mean"], y_pos, c=colors, s=55, zorder=5)
ax.axvline(0, color="black", linewidth=0.9, linestyle="-")

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["display"], fontsize=8)
ax.set_xlabel("Δ mean CNV score (Basal − Luminal)", fontsize=9)
ax.set_title(
    "Top Differential Copy Number Variation Loci\n"
    "Top 10 amplified in Basal (C1) · Top 10 amplified in Luminal (C0)\n"
    "(deduplicated by GISTIC2 segment — one representative gene per region)",
    fontsize=9
)
boundary_y = len(top_luminal) - 0.5
ax.axhline(boundary_y, color="#AAAAAA", linewidth=0.7, linestyle="--")

xmin = plot_df["delta_mean"].min()
xmax = plot_df["delta_mean"].max()
xrange = xmax - xmin
ax.text(xmin - xrange * 0.02, len(top_luminal) / 2,
        "← Higher in\nLuminal (C0)",
        ha="right", va="center", fontsize=7.5, color=CLUSTER_COLORS[0])
ax.text(xmax + xrange * 0.02, len(top_luminal) + len(top_basal) / 2,
        "Higher in\nBasal (C1) →",
        ha="left", va="center", fontsize=7.5, color=CLUSTER_COLORS[1])

legend_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[1], label="Higher in Basal (C1)"),
    mpatches.Patch(color=CLUSTER_COLORS[0], label="Higher in Luminal (C0)"),
]
ax.legend(handles=legend_patches, fontsize=8, frameon=False, loc="lower right")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
out_path = OUT_VAL / "fig_de_cnv.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
log.info("  Saved: %s", out_path)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Top differential methylation probes
# ══════════════════════════════════════════════════════════════════════════════
log.info("Plotting: fig_de_meth …")

meth_df = pd.read_csv(TABLES / "top_de_meth.tsv", sep="\t")

# Directions: HYPERmeth_Basal (high in basal) and HYPOMeth_Basal (low in basal = high in luminal)
hyper = meth_df[meth_df["direction"].str.startswith("HYPER")].nlargest(10, "delta_corrected_meth")
hypo  = meth_df[meth_df["direction"].str.startswith("HYPO")].nsmallest(10, "delta_corrected_meth")

# If no HYPO entries, take bottom-10 overall
if len(hypo) == 0:
    hypo = meth_df.nsmallest(10, "delta_corrected_meth")

plot_df = pd.concat([hypo, hyper]).reset_index(drop=True)
colors  = [CLUSTER_COLORS[1] if "Basal" in d else CLUSTER_COLORS[0]
           for d in plot_df["direction"]]

fig, ax = plt.subplots(figsize=(7, 6))
y_pos = np.arange(len(plot_df))

ax.hlines(y_pos, 0, plot_df["delta_corrected_meth"], colors=colors, linewidth=1.5)
ax.scatter(plot_df["delta_corrected_meth"], y_pos, c=colors, s=55, zorder=5)
ax.axvline(0, color="black", linewidth=0.9, linestyle="-")

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["probe_id"], fontsize=8)
ax.set_xlabel("Δ corrected methylation (Basal − Luminal)", fontsize=9)
ax.set_title(
    "Top Differential Methylation Probes\n"
    "Top 10 hypermethylated in Basal (C1) · Top 10 hypomethylated in Basal (C0)",
    fontsize=9
)

if len(hypo) > 0:
    ax.axhline(len(hypo) - 0.5, color="#AAAAAA", linewidth=0.7, linestyle="--")

legend_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[1], label="Hypermethylated in Basal (C1)"),
    mpatches.Patch(color=CLUSTER_COLORS[0], label="Hypomethylated in Basal / higher in Luminal (C0)"),
]
ax.legend(handles=legend_patches, fontsize=8, frameon=False, loc="lower right")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
out_path = OUT_VAL / "fig_de_meth.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
log.info("  Saved: %s", out_path)

log.info("All four missing figures generated successfully.")
