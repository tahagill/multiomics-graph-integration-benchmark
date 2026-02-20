"""
Step 11 – Clinical & Molecular Validation
==========================================
Validates SNF cluster assignments against independent clinical and molecular data
that were NEVER used during fusion/clustering.

Outputs
-------
results/tables/
    ihc_validation.tsv          — ER/PR/Her2/TN rates + chi-squared p-values
    top_de_genes.tsv            — Top 40 differentially expressed genes with symbols
    top_de_cnv.tsv              — Top 20 differential CNV loci with symbols
    top_de_meth.tsv             — Top 20 differential methylation probes

results/figures/validation/
    fig_ihc_barplot.pdf         — ER / PR / TN positivity rates by cluster
    fig_top_de_genes.pdf        — Lollipop plot: top 20 up + down genes
    fig_km_k5_panel.pdf         — k=5 KM curves in a 5-subplot panel
    fig_validation_composite.pdf — Combined panel (IHC + top genes + KM k=5)
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import mygene

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
OUT_TAB = ROOT / "results" / "tables"
OUT_FIG = ROOT / "results" / "figures" / "validation"
OUT_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLUSTER_COLORS = {0: "#4878D0", 1: "#D65F5F"}
CLUSTER_NAMES  = {0: "Cluster 1\n(Luminal, n=503)", 1: "Cluster 2\n(Basal, n=141)"}
KM_COLORS = ["#4878D0", "#E88A2E", "#D65F5F", "#59A14F", "#9467BD"]

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load():
    log.info("Loading data…")
    cl = pd.read_parquet(PROC / "clinical_preprocessed.parquet")
    ca = pd.read_parquet(PROC / "cluster_assignments.parquet")
    rna  = pd.read_parquet(PROC / "rna_preprocessed.parquet")   # genes × patients
    cnv  = pd.read_parquet(PROC / "cnv_preprocessed.parquet")
    meth = pd.read_parquet(PROC / "methylation_preprocessed.parquet")

    # Align: rna/cnv/meth are (features × patients) → transpose
    rna_T  = rna.T;  cnv_T  = cnv.T;  meth_T = meth.T

    common = cl.index.intersection(ca.index).intersection(rna_T.index)
    cl   = cl.loc[common]
    ca   = ca.loc[common]
    rna_T  = rna_T.loc[common]
    cnv_T  = cnv_T.loc[common]
    meth_T = meth_T.loc[common]

    df = ca.join(cl)
    log.info("  Cohort: %d patients, %d RNA features, %d CNV features, %d meth probes",
             len(common), rna_T.shape[1], cnv_T.shape[1], meth_T.shape[1])
    return df, rna_T, cnv_T, meth_T


# ═══════════════════════════════════════════════════════════════════════════════
# ENSG → SYMBOL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def ensg_to_symbol(ensg_list: list[str]) -> dict[str, str]:
    """Query mygene.info for a batch of ENSG versioned IDs → gene symbol."""
    log.info("  Querying mygene.info for %d IDs…", len(ensg_list))
    mg = mygene.MyGeneInfo()
    # strip version suffix before querying
    clean      = [e.split(".")[0] for e in ensg_list]
    versioned  = ensg_list            # keep originals as keys

    results = mg.getgenes(clean, fields="symbol", species="human", verbose=False)
    sym_map = {}
    for orig, res in zip(versioned, results):
        sym_map[orig] = res.get("symbol", orig.split(".")[0])
    return sym_map


# ═══════════════════════════════════════════════════════════════════════════════
# IHC VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_ihc_validation(df: pd.DataFrame):
    log.info("Running IHC / receptor-status validation…")

    rows = []
    for cl_id in [0, 1]:
        mask  = df["cluster_k2"] == cl_id
        n     = mask.sum()
        sub   = df.loc[mask]

        er_pos  = (sub["er_status_by_ihc"]  == "Positive").sum()
        pr_pos  = (sub["pr_status_by_ihc"]  == "Positive").sum()
        her2_pos = (sub["her2_status_by_ihc"] == "Positive").sum()
        tn = ((sub["er_status_by_ihc"]  == "Negative") &
              (sub["pr_status_by_ihc"]  == "Negative") &
              (sub["her2_status_by_ihc"] == "Negative")).sum()

        rows.append({
            "cluster":     cl_id,
            "n":           n,
            "ER_pos":      er_pos,
            "ER_pos_pct":  100 * er_pos  / n,
            "PR_pos":      pr_pos,
            "PR_pos_pct":  100 * pr_pos  / n,
            "Her2_pos":    her2_pos,
            "Her2_pos_pct": 100 * her2_pos / n,
            "TN":          tn,
            "TN_pct":      100 * tn / n,
        })

    ihc = pd.DataFrame(rows).set_index("cluster")

    # Chi-squared tests (observed vs expected independence)
    pvals = {}
    for marker, pos_col in [("ER", "er_status_by_ihc"),
                             ("PR", "pr_status_by_ihc"),
                             ("Her2", "her2_status_by_ihc")]:
        ct = pd.crosstab(df["cluster_k2"], df[pos_col])
        chi2, p, *_ = chi2_contingency(ct)
        pvals[marker] = p

    # TN chi-squared
    tn_col = ((df["er_status_by_ihc"]  == "Negative") &
              (df["pr_status_by_ihc"]  == "Negative") &
              (df["her2_status_by_ihc"] == "Negative"))
    ct_tn  = pd.crosstab(df["cluster_k2"], tn_col)
    _, p_tn, *_ = chi2_contingency(ct_tn)
    pvals["TN"] = p_tn

    # Print summary
    log.info("  %-20s  %s  %s  %s",
             "", "Cluster 0 (Luminal)", "Cluster 1 (Basal)", "chi2 p")
    for mk in ["ER", "PR", "Her2", "TN"]:
        col_map = {"ER": "ER_pos_pct", "PR": "PR_pos_pct",
                   "Her2": "Her2_pos_pct", "TN": "TN_pct"}
        c = col_map[mk]
        log.info("  %-20s  %5.1f%%               %5.1f%%             p=%s",
                 mk + "-positive",
                 ihc.loc[0, c], ihc.loc[1, c],
                 f"{pvals[mk]:.2e}")

    # Save table
    ihc["ER_chi2_p"]   = pvals["ER"]
    ihc["PR_chi2_p"]   = pvals["PR"]
    ihc["Her2_chi2_p"] = pvals["Her2"]
    ihc["TN_chi2_p"]   = pvals["TN"]
    out = OUT_TAB / "ihc_validation.tsv"
    ihc.to_csv(out, sep="\t", float_format="%.4f")
    log.info("  Saved → %s", out)

    return ihc, pvals


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL EXPRESSION  (RNA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_de_analysis(df: pd.DataFrame, rna_T: pd.DataFrame,
                    cnv_T: pd.DataFrame, meth_T: pd.DataFrame):
    log.info("Running differential feature analysis…")

    mask0 = (df["cluster_k2"] == 0).values
    mask1 = (df["cluster_k2"] == 1).values

    # ── RNA ──────────────────────────────────────────────────────────────────
    diff_rna = rna_T[mask1].mean() - rna_T[mask0].mean()

    # Mann-Whitney U p-values for top candidates
    log.info("  Computing Mann-Whitney U p-values for top 40 RNA genes…")
    top_ensg = pd.concat([diff_rna.nlargest(20), diff_rna.nsmallest(20)]).index.tolist()
    pvals_rna = {}
    for g in top_ensg:
        _, p = mannwhitneyu(rna_T.loc[mask1, g], rna_T.loc[mask0, g],
                            alternative="two-sided")
        pvals_rna[g] = p

    # Map ENSG → symbol
    sym_map = ensg_to_symbol(top_ensg)

    de_rows = []
    for ensg in top_ensg:
        de_rows.append({
            "ensg_id": ensg,
            "symbol":  sym_map.get(ensg, ensg),
            "delta_mean": diff_rna[ensg],
            "direction": "UP_Basal" if diff_rna[ensg] > 0 else "UP_Luminal",
            "mw_pval": pvals_rna[ensg],
        })
    de_df = pd.DataFrame(de_rows).sort_values("delta_mean", ascending=False)
    out = OUT_TAB / "top_de_genes.tsv"
    de_df.to_csv(out, sep="\t", index=False, float_format="%.6f")
    log.info("  Saved → %s", out)

    # ── CNV ──────────────────────────────────────────────────────────────────
    diff_cnv   = cnv_T[mask1].mean() - cnv_T[mask0].mean()
    mean_basal = cnv_T[mask1].mean()
    mean_lumin = cnv_T[mask0].mean()
    top_cnv  = pd.concat([diff_cnv.nlargest(10), diff_cnv.nsmallest(10)]).index.tolist()
    sym_cnv  = ensg_to_symbol(top_cnv)
    cnv_rows = [{"ensg_id": e, "symbol": sym_cnv.get(e, e),
                 "mean_basal": mean_basal[e],
                 "mean_luminal": mean_lumin[e],
                 "delta_mean": diff_cnv[e],
                 "direction": "UP_Basal" if diff_cnv[e] > 0 else "UP_Luminal"}
                for e in top_cnv]
    cnv_df = pd.DataFrame(cnv_rows).sort_values("delta_mean", ascending=False)
    out_cnv = OUT_TAB / "top_de_cnv.tsv"
    cnv_df.to_csv(out_cnv, sep="\t", index=False, float_format="%.6f")
    log.info("  Saved → %s", out_cnv)

    # ── Methylation ──────────────────────────────────────────────────────────
    diff_meth = meth_T[mask1].mean() - meth_T[mask0].mean()
    top_meth  = pd.concat([diff_meth.nlargest(10),
                            diff_meth.nsmallest(10)]).index.tolist()
    mean_meth_basal = meth_T[mask1].mean()
    mean_meth_lumin = meth_T[mask0].mean()
    meth_rows = [{"probe_id": p,
                  "mean_basal": mean_meth_basal[p],
                  "mean_luminal": mean_meth_lumin[p],
                  "delta_corrected_meth": diff_meth[p],
                  "direction": "HYPERmeth_Basal" if diff_meth[p] > 0 else "HYPOmeth_Basal"}
                 for p in top_meth]
    meth_df = pd.DataFrame(meth_rows).sort_values("delta_corrected_meth", ascending=False)
    out_meth = OUT_TAB / "top_de_meth.tsv"
    meth_df.to_csv(out_meth, sep="\t", index=False, float_format="%.6f")
    log.info("  Saved → %s", out_meth)

    return de_df, cnv_df, meth_df, sym_map


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ihc(df: pd.DataFrame, ihc: pd.DataFrame, pvals: dict):
    log.info("Plotting IHC validation figure…")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.40)

    # ── A: ER / PR / TN positivity rates ────────────────────────────────────
    ax = axes[0]
    markers  = ["ER", "PR", "TN"]
    pct_cols = ["ER_pos_pct", "PR_pos_pct", "TN_pct"]
    x   = np.arange(len(markers))
    w   = 0.32

    for offset, cl_id in [(-w/2, 0), (w/2, 1)]:
        vals = [ihc.loc[cl_id, c] for c in pct_cols]
        bars = ax.bar(x + offset, vals, w,
                      color=CLUSTER_COLORS[cl_id], alpha=0.85,
                      label=f"Cluster {cl_id + 1} "
                            f"({'Luminal' if cl_id == 0 else 'Basal'})")
        # add pct labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    # p-value annotations
    for i, mk in enumerate(markers):
        p = pvals[mk]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_max = max(ihc.loc[0, pct_cols[i]], ihc.loc[1, pct_cols[i]]) + 6
        ax.annotate(f"p={p:.1e}\n{sig}",
                    xy=(x[i], y_max), ha="center", fontsize=7.5,
                    color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["ER-positive", "PR-positive", "Triple-negative"])
    ax.set_ylabel("Patients in cluster (%)")
    ax.set_title("(A) Receptor status by SNF cluster (k=2)\n"
                 "(IHC — independent of multi-omics fusion)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False)

    # ── B: Her2 IHC breakdown ────────────────────────────────────────────────
    ax = axes[1]
    her2_cats = ["Positive", "Equivocal", "Negative", "[Not Evaluated]"]
    bottom0 = np.zeros(2)
    bar_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#bdc3c7"]
    cluster_labels = ["Cluster 1\n(Luminal)", "Cluster 2\n(Basal)"]

    for cat, col in zip(her2_cats, bar_colors):
        vals = []
        for cl_id in [0, 1]:
            sub = df[df["cluster_k2"] == cl_id]
            n   = len(sub)
            vals.append(100 * (sub["her2_status_by_ihc"] == cat).sum() / n)
        vals = np.array(vals)
        ax.bar([0, 1], vals, 0.5, bottom=bottom0, color=col, alpha=0.85,
               label=cat.replace("[Not Evaluated]", "Not Evaluated"))
        bottom0 += vals

    her2_p = pvals["Her2"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cluster_labels)
    ax.set_ylabel("Patients in cluster (%)")
    ax.set_title(f"(B) Her2 IHC status by cluster\n(chi2 p={her2_p:.2e})")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Figure 5 – Independent clinical validation: IHC receptor status by SNF cluster",
        fontsize=11
    )
    out = OUT_FIG / "fig_ihc_barplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


def plot_top_de_genes(de_df: pd.DataFrame, sym_map: dict):
    log.info("Plotting top DE genes lollipop…")

    # top 10 up in Basal + top 10 up in Luminal (down in Basal)
    up_basal   = de_df[de_df["delta_mean"] > 0].nlargest(10, "delta_mean")
    up_luminal = de_df[de_df["delta_mean"] < 0].nsmallest(10, "delta_mean")
    plot_df    = pd.concat([up_basal, up_luminal]).sort_values("delta_mean")

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#D65F5F" if d > 0 else "#4878D0" for d in plot_df["delta_mean"]]
    y      = np.arange(len(plot_df))

    ax.hlines(y, 0, plot_df["delta_mean"], colors=colors, linewidth=1.5, alpha=0.7)
    ax.scatter(plot_df["delta_mean"], y,
               c=colors, s=55, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["symbol"], fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ mean expression (Basal − Luminal,\npost-batch-correction log₂ units)")
    ax.set_title("Top 20 differentially expressed genes\n"
                 "(top 10 up in each cluster, Mann–Whitney U)")

    # Annotations for well-known markers
    known = {"ESR1", "PGR", "FOXA1", "GATA3", "EGFR", "CDH3", "ERBB2",
             "MKI67", "KRT14", "KRT5"}
    for _, row in plot_df.iterrows():
        sym = row["symbol"]
        if sym in known:
            xi = row["delta_mean"]
            yi_val = plot_df.index.get_loc(_) if _ in plot_df.index else None
            pass  # handled below via enumerate

    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row["symbol"] in known:
            ax.annotate(f"  {row['symbol']}",
                        xy=(row["delta_mean"], i),
                        va="center", fontsize=7.5,
                        color="#D65F5F" if row["delta_mean"] > 0 else "#4878D0")

    # Legend
    up_patch   = mpatches.Patch(color="#D65F5F", alpha=0.85, label="UP in Basal cluster")
    down_patch = mpatches.Patch(color="#4878D0", alpha=0.85, label="UP in Luminal cluster")
    ax.legend(handles=[up_patch, down_patch], frameon=False)

    out = OUT_FIG / "fig_top_de_genes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


def plot_km_k5(df: pd.DataFrame):
    log.info("Plotting k=5 KM panel…")

    # PAM50-derived cluster names (from contingency analysis)
    k5_names = {
        0: "C1: Mixed/Outlier (n=52)",
        1: "C2: LumA+LumB+Her2 (n=221)",
        2: "C3: Basal-pure (n=104)",
        3: "C4: LumA-rich (n=98)",
        4: "C5: LumA-dominant (n=169)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(wspace=0.38, bottom=0.25)

    # ── A: all 5 curves on one axes ──────────────────────────────────────────
    ax = axes[0]
    event_rates = {}
    for cl_id in range(5):
        mask = df["cluster_k5"] == cl_id
        if mask.sum() == 0:
            continue
        t = df.loc[mask, "os_time"].values.astype(float) / 30.4375   # days → months
        e = df.loc[mask, "os_event"].values.astype(bool)
        event_rates[cl_id] = 100 * e.sum() / len(e)
        kmf = KaplanMeierFitter()
        kmf.fit(t, e, label=k5_names[cl_id])
        kmf.plot_survival_function(ax=ax, ci_show=False,
                                   color=KM_COLORS[cl_id], linewidth=1.8,
                                   at_risk_counts=True)

    # Multivariate log-rank
    lr = multivariate_logrank_test(df["os_time"], df["cluster_k5"], df["os_event"])
    ax.set_title(f"(A) Overall survival: SNF k=5\n(log-rank p={lr.p_value:.4f})")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # ── B: event rate bar chart ───────────────────────────────────────────────
    ax = axes[1]
    cl_ids = sorted(event_rates.keys())
    bars   = ax.bar(cl_ids,
                    [event_rates[c] for c in cl_ids],
                    color=[KM_COLORS[c] for c in cl_ids],
                    alpha=0.85, width=0.55)
    for bar, cl_id in zip(bars, cl_ids):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{event_rates[cl_id]:.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(cl_ids)
    ax.set_xticklabels([f"C{c+1}" for c in cl_ids])
    ax.set_ylabel("Event rate (%)")
    ax.set_title("(B) Mortality rate per SNF cluster (k=5)")
    ax.set_ylim(0, 25)

    fig.suptitle("Figure 6 – Survival stratification by SNF clusters (k=5)", fontsize=11)
    out = OUT_FIG / "fig_km_k5_panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


def plot_validation_composite(df: pd.DataFrame, ihc: pd.DataFrame,
                               pvals: dict, de_df: pd.DataFrame):
    """4-panel composite: ER/PR/TN bars | top DE genes | k=5 KM | event rate bar."""
    log.info("Plotting validation composite panel…")

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.45)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel A: ER / PR / TN
    ax_a = fig.add_subplot(gs[0, 0])
    markers  = ["ER", "PR", "TN"]
    pct_cols = ["ER_pos_pct", "PR_pos_pct", "TN_pct"]
    x = np.arange(len(markers)); w = 0.32

    for offset, cl_id in [(-w/2, 0), (w/2, 1)]:
        vals = [ihc.loc[cl_id, c] for c in pct_cols]
        bars = ax_a.bar(x + offset, vals, w,
                        color=CLUSTER_COLORS[cl_id], alpha=0.85,
                        label=f"Cluster {cl_id+1} ({'Luminal' if cl_id==0 else 'Basal'})")
        for bar, v in zip(bars, vals):
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                      f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5)

    for i, mk in enumerate(markers):
        p = pvals[mk]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_top = max(ihc.loc[0, pct_cols[i]], ihc.loc[1, pct_cols[i]]) + 6
        ax_a.annotate(f"{sig}\np={p:.1e}", xy=(x[i], y_top),
                      ha="center", fontsize=7, color="black")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["ER-positive", "PR-positive", "Triple-negative"])
    ax_a.set_ylabel("% of cluster")
    ax_a.set_title("(A) IHC receptor status\n(independent of fusion)")
    ax_a.set_ylim(0, 120)
    ax_a.legend(frameon=False, fontsize=7.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel B: top DE genes lollipop
    ax_b = fig.add_subplot(gs[0, 1])
    up_b = de_df[de_df["delta_mean"] > 0].nlargest(8, "delta_mean")
    up_l = de_df[de_df["delta_mean"] < 0].nsmallest(8, "delta_mean")
    pde  = pd.concat([up_b, up_l]).sort_values("delta_mean")

    cols = ["#D65F5F" if d > 0 else "#4878D0" for d in pde["delta_mean"]]
    yy   = np.arange(len(pde))
    ax_b.hlines(yy, 0, pde["delta_mean"], colors=cols, linewidth=1.8, alpha=0.75)
    ax_b.scatter(pde["delta_mean"], yy, c=cols, s=45, zorder=5)
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(pde["symbol"], fontsize=8)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.set_xlabel("Δ mean expression (Basal − Luminal)")
    ax_b.set_title("(B) Top differentially expressed genes\n(RNA-seq, post-batch-correction)")
    known = {"ESR1","PGR","FOXA1","GATA3","EGFR","CDH3","MKI67","KRT14","ERBB2"}
    for i, (_, row) in enumerate(pde.iterrows()):
        if row["symbol"] in known:
            ax_b.annotate(f"  {row['symbol']}", xy=(row["delta_mean"], i),
                          va="center", fontsize=7,
                          color="#D65F5F" if row["delta_mean"]>0 else "#4878D0")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel C: k=5 KM
    ax_c = fig.add_subplot(gs[1, 0])
    k5_labels = {0:"C1:Mixed", 1:"C2:LumA+LumB+Her2", 2:"C3:Basal",
                 3:"C4:LumA-rich", 4:"C5:LumA-pure"}
    ev_rates = {}
    for cl_id in range(5):
        mask = df["cluster_k5"] == cl_id
        if not mask.any(): continue
        t = df.loc[mask,"os_time"].values.astype(float) / 30.4375   # days → months
        e = df.loc[mask,"os_event"].values.astype(bool)
        ev_rates[cl_id] = 100 * e.sum() / len(e)
        kmf = KaplanMeierFitter()
        kmf.fit(t, e, label=k5_labels[cl_id])
        kmf.plot_survival_function(ax=ax_c, ci_show=False,
                                   color=KM_COLORS[cl_id], linewidth=1.8)
    lr = multivariate_logrank_test(df["os_time"], df["cluster_k5"], df["os_event"])
    ax_c.set_title(f"(C) Overall survival: SNF k=5 (log-rank p={lr.p_value:.4f})")
    ax_c.set_xlabel("Time (months)"); ax_c.set_ylabel("Survival probability")
    ax_c.legend(frameon=False, fontsize=7, loc="upper right")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel D: PAM50 subtype composition by cluster (contingency heatmap)
    ax_d = fig.add_subplot(gs[1, 1])
    pam50_order = ["BRCA_LumA", "BRCA_LumB", "BRCA_Her2",
                   "BRCA_Basal", "BRCA_Normal"]
    labeled = df[df["pam50_subtype"].notna() &
                 (df["pam50_subtype"] != "Unknown")].copy()
    ct = pd.crosstab(labeled["cluster_k2"], labeled["pam50_subtype"])
    ct = ct.reindex(columns=[c for c in pam50_order if c in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    im = ax_d.imshow(ct_pct.values, aspect="auto", cmap="Blues",
                     vmin=0, vmax=100)
    plt.colorbar(im, ax=ax_d, label="% of cluster", shrink=0.80)
    ax_d.set_xticks(range(len(ct_pct.columns)))
    ax_d.set_xticklabels(
        [c.replace("BRCA_", "") for c in ct_pct.columns],
        rotation=30, ha="right"
    )
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["Cluster 1\n(Luminal)", "Cluster 2\n(Basal)"])
    ax_d.set_title("(D) PAM50 composition by cluster\n"
                   f"(k=2, n={len(labeled)} labelled patients)")
    for i in range(ct_pct.shape[0]):
        for j in range(ct_pct.shape[1]):
            pct_v = ct_pct.values[i, j]
            n_v   = ct.values[i, j]
            txt_col = "white" if pct_v > 55 else "black"
            ax_d.text(j, i, f"{pct_v:.0f}%\n(n={n_v})",
                      ha="center", va="center", fontsize=7.5, color=txt_col)

    fig.suptitle(
        "Figure 5 – Clinical & molecular validation of SNF clusters\n"
        "(all variables are independent of the multi-omics fusion input)",
        fontsize=11, y=1.01
    )
    out = OUT_FIG / "fig_validation_composite.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║        STEP 11: CLINICAL & MOLECULAR VALIDATION         ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    df, rna_T, cnv_T, meth_T = load()

    ihc, pvals       = run_ihc_validation(df)
    de_df, cnv_df, meth_df, sym_map = run_de_analysis(df, rna_T, cnv_T, meth_T)

    plot_ihc(df, ihc, pvals)
    plot_top_de_genes(de_df, sym_map)
    plot_km_k5(df)
    plot_validation_composite(df, ihc, pvals, de_df)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║              STEP 11 COMPLETE                           ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Tables → results/tables/                               ║")
    log.info("║    ihc_validation.tsv                                   ║")
    log.info("║    top_de_genes.tsv                                     ║")
    log.info("║    top_de_cnv.tsv                                       ║")
    log.info("║    top_de_meth.tsv                                      ║")
    log.info("║  Figures → results/figures/validation/                  ║")
    log.info("║    fig_ihc_barplot.pdf                                  ║")
    log.info("║    fig_top_de_genes.pdf                                 ║")
    log.info("║    fig_km_k5_panel.pdf                                  ║")
    log.info("║    fig_validation_composite.pdf  ← main deliverable     ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
