"""
Step 10 – Publication Figures
==============================
Creates four composite publication-quality panels plus a master results table.

Outputs (results/figures/panels/):
  fig1_overview.pdf       — UMAP coloured by cluster & PAM50, + KM k=2
  fig2_benchmarks.pdf     — Clustering metrics + C-index comparison
  fig3_survival_delta.pdf — Delta C-index forest + covariate-adjusted C-index
  fig4_sensitivity.pdf    — K / feature / metric sensitivity sweeps

Outputs (results/tables/):
  master_results_table.tsv — merged table of all key metrics
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from umap import UMAP

warnings.filterwarnings("ignore")

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
META = ROOT / "metadata"
RES_TAB = ROOT / "results" / "tables"
OUT_DIR = ROOT / "results" / "figures" / "panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,          # editable text in PDF
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PAM50_COLORS = {
    "BRCA_LumA":   "#4878D0",
    "BRCA_LumB":   "#6ACC65",
    "BRCA_Her2":   "#EE854A",
    "BRCA_Basal":  "#D65F5F",
    "BRCA_Normal": "#956CB4",
    "Unknown":     "#AAAAAA",
}
CLUSTER_COLORS = {0: "#4878D0", 1: "#D65F5F"}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading data…")

    clinical = pd.read_parquet(PROC / "clinical_preprocessed.parquet")

    cluster_all = pd.read_parquet(PROC / "cluster_assignments.parquet")

    spectral_emb = pd.read_parquet(PROC / "spectral_embedding.parquet")
    assert spectral_emb.shape == (644, 50), spectral_emb.shape

    # PAM50 from clinical (already merged in preprocessing)
    pam50 = clinical["pam50_subtype"].fillna("Unknown")

    # Cluster k=2 labels (1-indexed)
    cluster_k2 = cluster_all["cluster_k2"]

    # Survival columns
    os_time  = clinical["os_time"]
    os_event = clinical["os_event"].astype(bool)

    # Tables
    surv    = pd.read_csv(RES_TAB / "survival_metrics.tsv", sep="\t", index_col=0)
    delta   = pd.read_csv(RES_TAB / "delta_cindex.tsv", sep="\t")
    adj     = pd.read_csv(RES_TAB / "covariate_adjusted_cindex.tsv", sep="\t",
                          index_col=0)
    bcl     = pd.read_csv(RES_TAB / "baselines_clustering_metrics.tsv", sep="\t",
                          index_col=0)
    sens    = pd.read_csv(RES_TAB / "sensitivity_sweep.tsv", sep="\t")
    clust_k = pd.read_csv(RES_TAB / "clustering_metrics.tsv",  sep="\t")

    log.info("  clinical: %s", clinical.shape)
    log.info("  spectral_embedding: %s", spectral_emb.shape)
    log.info("  cluster_k2 distribution:\n%s", cluster_k2.value_counts().to_string())

    return dict(
        clinical=clinical,
        cluster_all=cluster_all,
        spectral_emb=spectral_emb,
        pam50=pam50,
        cluster_k2=cluster_k2,
        os_time=os_time,
        os_event=os_event,
        surv=surv,
        delta=delta,
        adj=adj,
        bcl=bcl,
        sens=sens,
        clust_k=clust_k,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UMAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_umap(spectral_emb: pd.DataFrame, seed: int = 42) -> np.ndarray:
    log.info("Computing UMAP (n_components=2, n_neighbors=30, min_dist=0.3)…")
    reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=seed,
        verbose=False,
    )
    emb2d = reducer.fit_transform(spectral_emb.values)
    log.info("  UMAP done: shape=%s", emb2d.shape)
    return emb2d


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  UMAP overview + KM k=2
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig1_overview(data: dict, umap2d: np.ndarray):
    log.info("Plotting Figure 1: overview panel…")

    cluster_k2 = data["cluster_k2"]
    pam50      = data["pam50"]
    os_time    = data["os_time"] / 30.4375   # days → months
    os_event   = data["os_event"]
    cluster_k5 = data["cluster_all"]["cluster_k5"]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.50)

    # ── A: UMAP coloured by cluster ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for cl, col in CLUSTER_COLORS.items():
        mask = (cluster_k2 == cl).values
        ax_a.scatter(umap2d[mask, 0], umap2d[mask, 1],
                     c=col, s=6, alpha=0.7, linewidths=0,
                     label=f"Cluster {cl+1} (n={mask.sum()})")
    ax_a.set_xlabel("UMAP 1"); ax_a.set_ylabel("UMAP 2")
    ax_a.set_title("(A) SNF clusters (k=2)")
    ax_a.legend(markerscale=2.5, frameon=False, loc="best")

    # ── B: UMAP coloured by PAM50 ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    subtypes = pam50.unique()
    for st in ["BRCA_Basal", "BRCA_LumA", "BRCA_LumB", "BRCA_Her2",
               "BRCA_Normal", "Unknown"]:
        if st not in subtypes:
            continue
        mask = (pam50 == st).values
        ax_b.scatter(umap2d[mask, 0], umap2d[mask, 1],
                     c=PAM50_COLORS.get(st, "#AAAAAA"),
                     s=6, alpha=0.7, linewidths=0,
                     label=st.replace("BRCA_", ""))
    ax_b.set_xlabel("UMAP 1"); ax_b.set_ylabel("")
    ax_b.set_title("(B) PAM50 subtypes")
    ax_b.legend(markerscale=2.5, frameon=False, loc="best")

    # ── C: KM curve k=2 ──────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    kmfs = {}
    for cl, col in CLUSTER_COLORS.items():
        mask = (cluster_k2 == cl).values
        t = os_time.values[mask].astype(float)
        e = os_event.values[mask].astype(bool)
        kmf = KaplanMeierFitter()
        kmf.fit(t, e, label=f"Cluster {cl+1}")
        kmf.plot_survival_function(ax=ax_c, ci_show=True,
                                   color=col, at_risk_counts=True)
        kmfs[cl] = (t, e)

    lr = logrank_test(kmfs[0][0], kmfs[1][0], kmfs[0][1], kmfs[1][1])
    ax_c.set_title(f"(C) Overall survival: SNF k=2\n(log-rank p={lr.p_value:.4f})")
    ax_c.set_xlabel("Time (months)"); ax_c.set_ylabel("Survival probability")
    ax_c.legend(frameon=False, loc="upper right")

    # ── D: k-selection ─────────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    clust_k = data["clust_k"]
    k_vals  = clust_k["k"].tolist()
    sil_vals = clust_k["silhouette"].tolist()
    stab_vals = clust_k["stability_mean_ari"].tolist()

    ax_d.plot(k_vals, sil_vals, "o-", color="#4878D0", linewidth=1.8,
              markersize=6, label="Silhouette")
    ax_d.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax_d.axvline(2, color="#D65F5F", linewidth=1.4, linestyle="--",
                 zorder=0, label="Pre-specified k=2")

    ax_d2 = ax_d.twinx()
    # fill in missing stability values (k=2 only has stability computed)
    stab_plot = [s if not pd.isna(s) else np.nan for s in stab_vals]
    ax_d2.plot(k_vals, stab_plot, "s--", color="#EE854A", linewidth=1.5,
               markersize=5, label="Stability ARI")
    ax_d2.set_ylabel("Stability (cross-seed ARI)", color="#EE854A")
    ax_d2.tick_params(axis="y", labelcolor="#EE854A")
    ax_d2.set_ylim(-0.1, 1.25)
    ax_d2.spines["top"].set_visible(False)

    ax_d.set_xticks(k_vals)
    ax_d.set_xlabel("Number of clusters (k)")
    ax_d.set_ylabel("Silhouette score")
    ax_d.set_title("(D) k-selection: silhouette & stability\n"
                   "(k=2 pre-specified by eigengap heuristic)")
    lines1, lbls1 = ax_d.get_legend_handles_labels()
    lines2, lbls2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, lbls1 + lbls2,
                frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Figure 1 – SNF multi-omics clustering of TCGA-BRCA (n=644)",
        fontsize=11, y=1.01
    )
    out = OUT_DIR / "fig1_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  Clustering & survival benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig2_benchmarks(data: dict):
    log.info("Plotting Figure 2: benchmarks panel…")

    bcl  = data["bcl"]
    surv = data["surv"]

    # Desired display order
    cl_order = ["SNF (proposed)", "Single-omics RNA",
                "Single-omics Methylation", "Single-omics CNV",
                "Early Integration (concat PCA)"]
    sur_order = ["SNF", "RNA-only", "Meth-only", "CNV-only",
                 "Early-concat", "Late-C1 (risk avg)", "Late-C2 (concat)"]

    bcl_plot = bcl.loc[bcl.index.intersection(cl_order)]
    bcl_plot = bcl_plot.reindex([m for m in cl_order if m in bcl_plot.index])

    surv_plot = surv.loc[surv.index.intersection(sur_order)]
    surv_plot = surv_plot.reindex([m for m in sur_order if m in surv_plot.index])

    x_cl  = np.arange(len(bcl_plot))
    x_sur = np.arange(len(surv_plot))
    w     = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.40)

    # ── A: ARI + NMI ─────────────────────────────────────────────────────────
    ax = axes[0]
    bars_ari = ax.bar(x_cl - w/2, bcl_plot["ari_pam50"], w,
                      color="#4878D0", alpha=0.85, label="ARI")
    bars_nmi = ax.bar(x_cl + w/2, bcl_plot["nmi_pam50"], w,
                      color="#EE854A", alpha=0.85, label="NMI")

    # Highlight SNF bar
    for bar in [bars_ari[0], bars_nmi[0]]:
        bar.set_edgecolor("black"); bar.set_linewidth(1.5)

    ax.set_xticks(x_cl)
    ax.set_xticklabels(
        [m.replace("Single-omics ", "").replace(" (concat PCA)", "\n(concat PCA)")
         for m in bcl_plot.index],
        rotation=30, ha="right"
    )
    ax.set_ylabel("Score")
    ax.set_title("(A) Clustering quality vs PAM50")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.65)
    ax.axhline(0, color="black", linewidth=0.6)
    # Annotate reference lines
    ax.axvline(-0.5, color="k", linewidth=0.5, linestyle=":")

    # ── B: OOF C-index with 95% CI ──────────────────────────────────────────
    ax = axes[1]
    colors = ["#D65F5F" if m == "SNF" else "#4878D0" for m in surv_plot.index]
    yerr_lo = surv_plot["oof_cindex"] - surv_plot["ci_lower_95"]
    yerr_hi = surv_plot["ci_upper_95"] - surv_plot["oof_cindex"]

    ax.bar(x_sur, surv_plot["oof_cindex"],
           color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x_sur, surv_plot["oof_cindex"],
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=2,
               label="Random (C=0.5)")
    ax.set_xticks(x_sur)
    ax.set_xticklabels(
        [m.replace(" (risk avg)", "\n(risk avg)").replace(" (concat)", "\n(concat)")
         for m in surv_plot.index],
        rotation=30, ha="right"
    )
    ax.set_ylabel("C-index (OOF)")
    ax.set_title("(B) Survival prediction (Ridge Cox, 5-fold CV)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0.40, 0.85)

    # Legend patch for SNF
    snf_patch  = mpatches.Patch(color="#D65F5F", alpha=0.85, label="SNF (proposed)")
    base_patch = mpatches.Patch(color="#4878D0", alpha=0.85, label="Baseline")
    ax.legend(handles=[snf_patch, base_patch,
                        Line2D([0], [0], color="gray", linestyle="--",
                               label="Random (C=0.5)")],
              frameon=False, loc="lower right", fontsize=7.5)

    fig.suptitle(
        "Figure 2 – Benchmarking SNF against single-omics and integration baselines",
        fontsize=11
    )
    out = OUT_DIR / "fig2_benchmarks.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  Delta C-index forest + covariate-adjusted C-index
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3_survival_delta(data: dict):
    log.info("Plotting Figure 3: survival delta & covariate-adjusted panel…")

    delta = data["delta"]
    adj   = data["adj"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.45)

    # ── A: Delta C-index forest plot ─────────────────────────────────────────
    ax = axes[0]
    comps = delta["comparison"].str.replace("SNF − ", "", regex=False)
    y     = np.arange(len(comps))

    for i, (_, row) in enumerate(delta.iterrows()):
        color = "#D65F5F" if row["ci_excludes_zero"] else "#4878D0"
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i],
                color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter([row["delta"]], [i], color=color, zorder=5, s=40)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(comps)
    ax.set_xlabel("Δ C-index (SNF − baseline)")
    ax.set_title("(A) Δ C-index vs SNF\n(paired bootstrap 95% CI, N=1000)")
    # legend
    sig_patch  = mpatches.Patch(color="#D65F5F", label="CI excludes 0 (sign.)")
    nsig_patch = mpatches.Patch(color="#4878D0", label="CI includes 0")
    ax.legend(handles=[sig_patch, nsig_patch], frameon=False,
              loc="lower right", fontsize=7.5)

    # ── B: Covariate-adjusted C-index ────────────────────────────────────────
    ax = axes[1]
    methods = adj.index.tolist()
    x = np.arange(len(methods))
    w = 0.32

    adj_colors = {"SNF": "#D65F5F", "RNA-only": "#4878D0"}
    for i, m in enumerate(methods):
        col = adj_colors.get(m, "#888888")
        ax.bar(i - w/2, adj.loc[m, "unadj_oof_cindex"], w,
               color=col, alpha=0.55, label="Unadjusted" if i == 0 else "")
        ax.bar(i + w/2, adj.loc[m, "adj_oof_cindex"],   w,
               color=col, alpha=0.95, label="+ Age/Stage" if i == 0 else "")
        # Arrow showing gain
        ax.annotate(
            f"+{adj.loc[m,'adj_oof_cindex'] - adj.loc[m,'unadj_oof_cindex']:.3f}",
            xy=(i + w/2, adj.loc[m, "adj_oof_cindex"] + 0.005),
            ha="center", va="bottom", fontsize=7.5, color=col
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("C-index (OOF)")
    ax.set_title("(B) Covariate-adjusted C-index\n(+ age at diagnosis + tumour stage)")
    ax.set_ylim(0.50, 0.85)

    # Manually build legend
    unadj_ari = mpatches.Patch(color="#888888", alpha=0.55, label="Unadjusted")
    adj_patch  = mpatches.Patch(color="#888888", alpha=0.95, label="+ Age/Stage")
    ax.legend(handles=[unadj_ari, adj_patch], frameon=False)

    fig.suptitle(
        "Figure 3 – Survival performance: delta C-index and covariate adjustment",
        fontsize=11
    )
    out = OUT_DIR / "fig3_survival_delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4_sensitivity(data: dict):
    log.info("Plotting Figure 4: sensitivity analysis panel…")

    sens = data["sens"]

    # Split rows by sweep type
    ref_row  = sens[sens["config"] == "K=20 · 5000feat · euclidean"].iloc[0]
    k_rows   = sens[sens["config"].str.contains("5000feat") &
                    sens["config"].str.contains("euclidean")].copy()
    k_vals   = [int(r.split("K=")[1].split(" ")[0]) for r in k_rows["config"]]
    k_rows["K"] = k_vals
    k_rows   = k_rows.sort_values("K")

    feat_rows = sens[sens["config"].str.contains("K=20") &
                     sens["config"].str.contains("euclidean")].copy()
    feat_rows = feat_rows.sort_values("cv_cindex", ascending=True)  # 2k first

    metric_rows = sens[sens["config"].str.contains("K=20") &
                       sens["config"].str.contains("5000feat")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── A: K sweep ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(k_rows["K"], k_rows["ari_pam50"],   "o-", color="#4878D0",
            label="ARI", linewidth=1.8, markersize=6)
    ax.plot(k_rows["K"], k_rows["nmi_pam50"],   "s-", color="#EE854A",
            label="NMI", linewidth=1.8, markersize=6)
    ax2 = ax.twinx()
    ax2.plot(k_rows["K"], k_rows["cv_cindex"], "^--", color="#D65F5F",
             label="C-index", linewidth=1.8, markersize=6)
    ax.set_xticks(k_rows["K"].tolist())
    ax.set_xlabel("K (nearest neighbours)")
    ax.set_ylabel("Clustering score")
    ax2.set_ylabel("C-index")
    ax2.spines["top"].set_visible(False)
    ax.set_title("(A) K sweep\n(n_feat=5000, metric=euclidean)")
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.5,
              loc="lower right")
    # Reference line
    ax.axvline(20, color="gray", linewidth=0.8, linestyle=":", zorder=0)

    # ── B: Feature sweep ──────────────────────────────────────────────────────
    ax = axes[1]
    feat_labels = [r.split("·")[1].strip() for r in feat_rows["config"]]
    x = np.arange(len(feat_rows))
    w = 0.28

    ax_twin = ax.twinx()
    ax.bar(x - w, feat_rows["ari_pam50"].values, w, color="#4878D0", alpha=0.85,
           label="ARI")
    ax.bar(x,     feat_rows["nmi_pam50"].values, w, color="#EE854A", alpha=0.85,
           label="NMI")
    ax_twin.bar(x + w, feat_rows["cv_cindex"].values, w, color="#D65F5F",
                alpha=0.85, label="C-index")

    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels)
    ax.set_ylabel("Clustering score")
    ax_twin.set_ylabel("C-index")
    ax_twin.spines["top"].set_visible(False)
    ax.set_title("(B) Feature count sweep\n(K=20, metric=euclidean)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.5,
              loc="lower right")
    # Annotate reference
    feat_labels_list = feat_labels if isinstance(feat_labels, list) else list(feat_labels)
    ref_idx = feat_labels_list.index("5000feat") if "5000feat" in feat_labels_list else len(feat_labels_list) - 1
    ax.annotate("★ ref", xy=(ref_idx - w/2, 0.02),
                ha="center", fontsize=7, color="#333")

    # ── C: Metric sweep ───────────────────────────────────────────────────────
    ax = axes[2]
    m_labels = [r.split("·")[-1].strip() for r in metric_rows["config"]]
    x = np.arange(len(metric_rows))

    ax_twin2 = ax.twinx()
    ax.bar(x - w, metric_rows["ari_pam50"].values, w, color="#4878D0",
           alpha=0.85, label="ARI")
    ax.bar(x,     metric_rows["nmi_pam50"].values, w, color="#EE854A",
           alpha=0.85, label="NMI")
    ax_twin2.bar(x + w, metric_rows["cv_cindex"].values, w, color="#D65F5F",
                 alpha=0.85, label="C-index")

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.set_ylabel("Clustering score")
    ax_twin2.set_ylabel("C-index")
    ax_twin2.spines["top"].set_visible(False)
    ax.set_title("(C) Affinity metric sweep\n(K=20, n_feat=5000)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.5,
              loc="lower right")

    fig.suptitle(
        "Figure 4 – Sensitivity analysis: one-at-a-time hyperparameter sweep",
        fontsize=11
    )
    out = OUT_DIR / "fig4_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def save_master_table(data: dict):
    log.info("Building master results table…")

    bcl  = data["bcl"].copy()
    surv = data["surv"].copy()

    # Merge clustering + survival on a shared 'method' key
    # Clustering table uses full names; rename to short key
    rename_cl = {
        "SNF (proposed)":               "SNF",
        "Single-omics RNA":              "RNA-only",
        "Single-omics CNV":              "CNV-only",
        "Single-omics Methylation":      "Meth-only",
        "Early Integration (concat PCA)":"Early-concat",
    }
    bcl_merged = bcl.rename(index=rename_cl)
    bcl_merged = bcl_merged[["ari_pam50", "nmi_pam50",
                               "stability_mean_ari", "logrank_p"]]
    bcl_merged.columns = ["ARI_PAM50", "NMI_PAM50", "Stability", "LR_p"]

    surv_sel = surv.loc[surv.index.intersection(
        ["SNF", "RNA-only", "CNV-only", "Meth-only",
         "Early-concat", "Late-C1 (risk avg)", "Late-C2 (concat)"]
    )][[  "oof_cindex", "ci_lower_95", "ci_upper_95"]]
    surv_sel.columns = ["C_index", "CI_lower", "CI_upper"]

    master = bcl_merged.join(surv_sel, how="outer")
    master.index.name = "Method"

    # Add delta table
    delta = data["delta"].set_index("comparison")
    delta.index = delta.index.str.replace("SNF − ", "", regex=False)
    master_delta = pd.DataFrame({
        "Delta_Cindex": delta["delta"],
        "Delta_CI_lo":  delta["ci_lower"],
        "Delta_CI_hi":  delta["ci_upper"],
        "Sig":          delta["ci_excludes_zero"],
    })

    # Covariate adjusted
    adj = data["adj"][["unadj_oof_cindex", "adj_oof_cindex"]]
    adj.columns = ["Unadj_Cindex", "Adj_Cindex(+Age+Stage)"]

    out = RES_TAB / "master_results_table.tsv"
    master.to_csv(out, sep="\t", float_format="%.4f")

    out_delta = RES_TAB / "master_delta_table.tsv"
    master_delta.to_csv(out_delta, sep="\t", float_format="%.4f")

    out_adj = RES_TAB / "master_adjusted_table.tsv"
    adj.to_csv(out_adj, sep="\t", float_format="%.4f")

    log.info("  Master table → %s", out)
    log.info("  Delta table  → %s", out_delta)
    log.info("  Adj. table   → %s", out_adj)

    # Print for quick review
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        log.info("\n%s", master.to_string(float_format="%.4f"))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║              STEP 10: PUBLICATION FIGURES               ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    data   = load_data()
    umap2d = compute_umap(data["spectral_emb"])

    plot_fig1_overview(data, umap2d)
    plot_fig2_benchmarks(data)
    plot_fig3_survival_delta(data)
    plot_fig4_sensitivity(data)
    save_master_table(data)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║              STEP 10 COMPLETE                           ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Figures saved to results/figures/panels/               ║")
    log.info("║    fig1_overview.pdf                                    ║")
    log.info("║    fig2_benchmarks.pdf                                  ║")
    log.info("║    fig3_survival_delta.pdf                              ║")
    log.info("║    fig4_sensitivity.pdf                                 ║")
    log.info("║  Tables saved to results/tables/                        ║")
    log.info("║    master_results_table.tsv                             ║")
    log.info("║    master_delta_table.tsv                               ║")
    log.info("║    master_adjusted_table.tsv                            ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
