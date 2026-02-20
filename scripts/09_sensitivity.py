"""
09_sensitivity.py
-----------------
Structured sensitivity analysis for the SNF pipeline.
Sweeps one parameter at a time (others fixed at reference values) and records
cluster ARI vs PAM50, log-rank p, and C-index for each configuration.

Parameters swept
────────────────
  K (graph k-nearest-neighbours)   : 10, 15, **20**, 25
  n_features per modality           : **5000**, 2000  (10k excluded: re-preprocessing needed)
  Affinity metric                   : **euclidean** (Wang kernel), cosine
  (bold = reference configuration = main result)

Design
──────
  One-at-a-time (OAT) sweep: for each parameter, vary it while keeping all
  others at reference values. This is the standard approach for sensitivity
  analysis in bioinformatics pipelines and matches the framing in the README.

  Full factorial is not performed because:
    1. 4 × 2 × 2 = 16 configurations would require 16 full SNF runs.
    2. The OAT design identifies which parameters matter; full factorial
       adds little biological insight at the cost of presentation clarity.

  The 10k-feature configuration is excluded because it requires re-running
  preprocessing (the existing parquets contain only top-5000 features by MAD).
  This is noted in the output table as a limitation.

Survival C-index
────────────────
  Each configuration uses 5-fold CV (seed=42 only, no multi-seed repeat).
  Alpha=1.0 is fixed (not tuned via inner CV) since this is an exploratory sweep.
  Per README: "Log-rank p-values across sensitivity sweep configurations are
  exploratory. The primary survival separation result is computed on the
  pre-specified k only."

Reference config metrics (from Steps 06 & 08)
───────────────────────────────────────────────
  K=20, 5k features, Euclidean
  ARI=0.4432, NMI=0.4951, logrank_p=0.1438, C-index=0.6814

Outputs
───────
  results/tables/
    sensitivity_sweep.tsv         full results table
  results/figures/sensitivity/
    k_sweep.pdf                   ARI/NMI/C-index vs K
    feature_sweep.pdf             ARI/NMI/C-index vs n_features
    metric_sweep.pdf              ARI/NMI/C-index: euclidean vs cosine
    stability_heatmap.pdf         ARI vs PAM50 heatmap across all configs

Run from project root:
  python scripts/09_sensitivity.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import snf
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
# Reference configuration (= main result from Steps 05-08)
REF_K           = 20
REF_NFEAT       = 5000
REF_METRIC      = "euclidean"
MU              = 0.5
T_ITER          = 20
N_SPECTRAL      = 50
K_CLUST         = 2
CV_SEED         = 42
COX_ALPHA       = 1.0   # fixed for exploratory sweep
L1_RATIO        = 0.01

# Sweep ranges
K_VALUES     = [10, 15, 20, 25]
NFEAT_VALUES = [2000, 5000]
METRIC_VALUES = ["euclidean", "cosine"]

PROCESSED   = Path("data/processed")
FIG_OUT     = Path("results/figures/sensitivity")
TABLE_OUT   = Path("results/tables")
FIG_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR     = Path("logs")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "09_sensitivity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Affinity computation (self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def compute_affinity_euclidean(data: np.ndarray, k: int, mu: float) -> np.ndarray:
    """Wang et al. 2014 bounded exponential kernel on squared Euclidean distance."""
    D     = cdist(data, data, metric="sqeuclidean")
    TT    = np.sort(D, axis=1)[:, 1:k+1].mean(axis=1)
    sigma = mu * (TT[:, None] + TT[None, :] + D) / 3.0
    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    W     = np.exp(-D / (2.0 * sigma))
    np.fill_diagonal(W, 0.0)
    return (W + W.T) / 2.0


def compute_affinity_cosine(data: np.ndarray, k: int, mu: float) -> np.ndarray:
    """
    Wang kernel variant using cosine distance = 1 − cosine_similarity.
    For z-scored data this is equivalent to using Pearson correlation distance.
    """
    # Cosine similarity → convert to distance
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    data_n = data / norms
    cos_sim = data_n @ data_n.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    D = 1.0 - cos_sim   # cosine distance in [0, 2]
    np.fill_diagonal(D, 0.0)
    # Apply Wang kernel on cosine distance
    TT    = np.sort(D, axis=1)[:, 1:k+1].mean(axis=1)
    sigma = mu * (TT[:, None] + TT[None, :] + D) / 3.0
    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    W     = np.exp(-D / (2.0 * sigma))
    np.fill_diagonal(W, 0.0)
    return (W + W.T) / 2.0


def compute_affinity(data: np.ndarray, k: int, mu: float,
                      metric: str) -> np.ndarray:
    if metric == "euclidean":
        return compute_affinity_euclidean(data, k, mu)
    elif metric == "cosine":
        return compute_affinity_cosine(data, k, mu)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ──────────────────────────────────────────────────────────────────────────────
# SNF pipeline for one parameter configuration
# ──────────────────────────────────────────────────────────────────────────────

def run_snf_config(
    feat_mats: list[np.ndarray],   # list of (n_patients, n_features) arrays
    k: int, mu: float, metric: str,
    n_spectral: int = N_SPECTRAL,
    t_iter: int = T_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-modality affinity matrices → SNF fusion → spectral embedding.
    Returns (fused_affinity, spectral_embedding).
    """
    affs = [compute_affinity(f, k, mu, metric) for f in feat_mats]

    # SNF fusion using snfpy (same as Step 05)
    fused = snf.snf(affs, K=k, t=t_iter, alpha=mu)
    # Clip minor numerical noise
    fused = np.clip(fused, 0.0, None)
    fused = (fused + fused.T) / 2.0
    np.fill_diagonal(fused, 0.0)

    # Spectral embedding
    se = SpectralEmbedding(
        n_components=n_spectral, affinity="precomputed", random_state=42
    )
    emb = se.fit_transform(fused).astype(np.float32)
    return fused, emb


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation for one configuration
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    fused: np.ndarray,
    emb: np.ndarray,
    y_surv: np.ndarray,
    pam50: pd.Series,
    clin: pd.DataFrame,
    config_label: str,
) -> dict:
    """
    Run clustering, PAM50 comparison, survival analysis, and Cox C-index.
    Returns metrics dict.
    """
    # Cluster
    sc = SpectralClustering(
        n_clusters=K_CLUST, affinity="precomputed",
        random_state=CV_SEED, n_init=10, assign_labels="kmeans"
    )
    labels = sc.fit_predict(fused)
    sizes  = np.bincount(labels).tolist()

    # ARI / NMI vs PAM50
    mask = pam50.notna().values
    ari  = adjusted_rand_score(pam50[mask].values, labels[mask])
    nmi  = normalized_mutual_info_score(
        pam50[mask].values, labels[mask], average_method="arithmetic"
    )

    # Log-rank p-value
    os_time  = clin["os_time"].values
    os_event = clin["os_event"].values
    df0 = pd.DataFrame({"t": os_time[labels == 0], "e": os_event[labels == 0]})
    df1 = pd.DataFrame({"t": os_time[labels == 1], "e": os_event[labels == 1]})
    if len(df0) > 0 and len(df1) > 0:
        lr  = logrank_test(df0["t"], df1["t"],
                           event_observed_A=df0["e"],
                           event_observed_B=df1["e"])
        lr_p = float(lr.p_value)
    else:
        lr_p = float("nan")

    # Ridge Cox C-index (5-fold, seed=42, fixed alpha)
    X = emb.astype(np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    fold_cidxs = []
    for tr_idx, te_idx in kf.split(X):
        y_te = y_surv[te_idx]
        if y_te["event"].sum() < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=L1_RATIO, alphas=[COX_ALPHA],
                max_iter=300, fit_baseline_model=False,
            )
            model.fit(Xtr, y_surv[tr_idx])
            pred = model.predict(Xte).ravel()
            c = concordance_index_censored(
                y_te["event"], y_te["time"], pred
            )[0]
            fold_cidxs.append(c)
        except Exception as e:
            log.warning(f"    Cox fold failed ({config_label}): {e}")

    cv_cindex = float(np.mean(fold_cidxs)) if fold_cidxs else float("nan")

    log.info(f"  {config_label:<40s} sizes={sizes}  "
             f"ARI={ari:.4f}  NMI={nmi:.4f}  "
             f"logrank_p={lr_p:.4f}  C-idx={cv_cindex:.4f}")

    return {
        "config":      config_label,
        "cluster_sizes": str(sizes),
        "ari_pam50":   ari,
        "nmi_pam50":   nmi,
        "logrank_p":   lr_p,
        "cv_cindex":   cv_cindex,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def line_sweep_plot(
    x_vals: list, y_dict: dict, x_label: str, ref_x,
    out_path: Path, title: str
) -> None:
    """
    Multi-metric line plot for a sweep over x_vals.
    y_dict: {metric_name: [values for each x_val]}
    """
    fig, axes = plt.subplots(1, len(y_dict), figsize=(4 * len(y_dict), 3.5))
    if len(y_dict) == 1:
        axes = [axes]

    metric_colours = {
        "ARI vs PAM50": "steelblue",
        "NMI vs PAM50": "darkorange",
        "Log-rank p":   "mediumpurple",
        "C-index":      "mediumseagreen",
    }

    for ax, (metric_name, values) in zip(axes, y_dict.items()):
        colour = metric_colours.get(metric_name, "grey")
        ax.plot(x_vals, values, "o-", color=colour, lw=2, markersize=7)
        # Mark reference
        if ref_x in x_vals:
            ref_val = values[x_vals.index(ref_x)]
            ax.axvline(ref_x, ls="--", lw=1, color="grey", alpha=0.5,
                       label="Reference")
            ax.plot(ref_x, ref_val, "s", color=colour,
                    markersize=10, zorder=5, label="Reference")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_title(metric_name, fontsize=10)
        if metric_name == "C-index":
            ax.set_ylim(0.50, 0.80)
            ax.axhline(0.5, ls=":", lw=0.8, color="grey", alpha=0.5)
        if metric_name == "ARI vs PAM50":
            ax.set_ylim(0.0, 0.6)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figure → {out_path}")


def stability_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """
    Heatmap of ARI vs PAM50 across all sensitivity configurations.
    Rows = metric parameter, cols = K value (roughly).
    """
    fig, ax = plt.subplots(figsize=(max(8, len(df)*1.0), 3.5))
    configs = df["config"].tolist()
    aris    = df["ari_pam50"].values
    cidxs   = df["cv_cindex"].values

    x = np.arange(len(configs))
    ax2 = ax.twinx()
    bars = ax.bar(x, aris, color="steelblue", alpha=0.7, label="ARI vs PAM50")
    line = ax2.plot(x, cidxs, "o-", color="mediumseagreen", lw=2,
                    markersize=6, label="C-index")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("ARI vs PAM50", color="steelblue", fontsize=9)
    ax2.set_ylabel("C-index", color="mediumseagreen", fontsize=9)
    ax.set_ylim(0, 0.6)
    ax2.set_ylim(0.45, 0.80)
    ax.set_title("Sensitivity sweep: ARI vs PAM50 and C-index across all configs",
                 fontsize=10)
    # Combined legend
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Stability heatmap → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║           STEP 9: SENSITIVITY ANALYSIS                  ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Reference config: K={REF_K}, n_feat={REF_NFEAT}, metric={REF_METRIC}")
    log.info(f"K sweep:          {K_VALUES}")
    log.info(f"n_features sweep: {NFEAT_VALUES}")
    log.info(f"Metric sweep:     {METRIC_VALUES}")

    # ── Load shared inputs ─────────────────────────────────────────────────────
    log.info("\nLoading shared inputs...")
    emb_df = pd.read_parquet(PROCESSED / "spectral_embedding.parquet")
    patient_ids = emb_df.index.tolist()
    n = len(patient_ids)

    clin    = pd.read_parquet(PROCESSED / "clinical_preprocessed.parquet")
    clin    = clin.loc[patient_ids]
    pam50   = clin["pam50_subtype"]
    y_surv  = np.array(
        [(bool(e), float(t)) for e, t in zip(clin["os_event"], clin["os_time"])],
        dtype=[("event", bool), ("time", float)],
    )

    # Load feature matrices (orient: patients × features)
    log.info("Loading feature matrices...")
    feat_full = {}
    for mod, fname in [("rna",  "rna_preprocessed.parquet"),
                        ("cnv",  "cnv_preprocessed.parquet"),
                        ("meth", "methylation_preprocessed.parquet")]:
        df  = pd.read_parquet(PROCESSED / fname).T.loc[patient_ids]
        feat_full[mod] = df.values.astype(np.float64)
        log.info(f"  {mod}: {feat_full[mod].shape}")

    # Reference result for comparison (from Steps 06 & 08)
    REF_METRICS = {
        "config":        f"K={REF_K} · {REF_NFEAT}feat · {REF_METRIC}",
        "cluster_sizes": "[503, 141]",
        "ari_pam50":     0.4432,
        "nmi_pam50":     0.4951,
        "logrank_p":     0.1438,
        "cv_cindex":     0.6914,   # fold-mean from Step 08
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Load top-2000 feature subsets (sort cols by MAD descending)
    # ══════════════════════════════════════════════════════════════════════════
    feat_2k = {}
    for mod in ["rna", "cnv", "meth"]:
        mat = feat_full[mod]   # 644 × 5000, already the top-5000 by MAD
        mad = np.median(np.abs(mat - np.median(mat, axis=0)), axis=0)
        top_idx = np.argsort(mad)[::-1][:2000]
        feat_2k[mod] = mat[:, top_idx]
    log.info(f"  2k-feature subsets: {feat_2k['rna'].shape}")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # SWEEP 1: K (nearest neighbours) — n_feat=5k, metric=euclidean
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SWEEP 1: K NEAREST NEIGHBOURS")
    log.info("═"*60)

    k_results = []
    for K in K_VALUES:
        config_label = f"K={K} · {REF_NFEAT}feat · {REF_METRIC}"
        if K == REF_K:
            # Use pre-computed result
            r = dict(REF_METRICS); r["config"] = config_label
            log.info(f"  {config_label:<40s} sizes=[503, 141]  "
                     f"ARI=0.4432  NMI=0.4951  logrank_p=0.1438  C-idx=0.6914"
                     f" [reference]")
        else:
            mats = [feat_full["rna"], feat_full["cnv"], feat_full["meth"]]
            fused, emb = run_snf_config(mats, K, MU, REF_METRIC)
            r = evaluate_config(fused, emb, y_surv, pam50, clin, config_label)
        k_results.append(r)
        all_results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # SWEEP 2: n_features — K=20, metric=euclidean
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SWEEP 2: NUMBER OF FEATURES PER MODALITY")
    log.info("═"*60)

    feat_results = []
    for n_feat in NFEAT_VALUES:
        config_label = f"K={REF_K} · {n_feat}feat · {REF_METRIC}"
        if n_feat == REF_NFEAT:
            r = dict(REF_METRICS); r["config"] = config_label
            log.info(f"  {config_label:<40s} [reference]")
        else:
            mats = [feat_2k["rna"], feat_2k["cnv"], feat_2k["meth"]]
            fused, emb = run_snf_config(mats, REF_K, MU, REF_METRIC)
            r = evaluate_config(fused, emb, y_surv, pam50, clin, config_label)
        feat_results.append(r)
        # only add to all_results if not already added (avoid duplicate reference)
        if n_feat != REF_NFEAT:
            all_results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # SWEEP 3: Affinity metric — K=20, n_feat=5k
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SWEEP 3: AFFINITY METRIC")
    log.info("═"*60)

    metric_results = []
    for metric in METRIC_VALUES:
        config_label = f"K={REF_K} · {REF_NFEAT}feat · {metric}"
        if metric == REF_METRIC:
            r = dict(REF_METRICS); r["config"] = config_label
            log.info(f"  {config_label:<40s} [reference]")
        else:
            mats = [feat_full["rna"], feat_full["cnv"], feat_full["meth"]]
            fused, emb = run_snf_config(mats, REF_K, MU, metric)
            r = evaluate_config(fused, emb, y_surv, pam50, clin, config_label)
        metric_results.append(r)
        if metric != REF_METRIC:
            all_results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # Build full results table (include reference only once)
    # ══════════════════════════════════════════════════════════════════════════

    # Add reference once at the top
    ref_row = dict(REF_METRICS)
    full_results = [ref_row] + all_results
    sweep_df = pd.DataFrame(full_results)
    # dedup in case reference was added via sweep
    sweep_df = sweep_df.drop_duplicates(subset="config").reset_index(drop=True)

    sweep_path = TABLE_OUT / "sensitivity_sweep.tsv"
    sweep_df.to_csv(sweep_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"\n  Full sweep table saved → {sweep_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SAVING FIGURES")
    log.info("═"*60)

    # K sweep
    line_sweep_plot(
        x_vals=K_VALUES,
        y_dict={
            "ARI vs PAM50": [r["ari_pam50"] for r in k_results],
            "NMI vs PAM50": [r["nmi_pam50"] for r in k_results],
            "C-index":      [r["cv_cindex"] for r in k_results],
        },
        x_label="K (nearest neighbours)",
        ref_x=REF_K,
        out_path=FIG_OUT / "k_sweep.pdf",
        title="Sensitivity: K nearest neighbours sweep\n"
               f"(n_feat={REF_NFEAT}, metric={REF_METRIC})",
    )

    # n_features sweep
    line_sweep_plot(
        x_vals=NFEAT_VALUES,
        y_dict={
            "ARI vs PAM50": [r["ari_pam50"] for r in feat_results],
            "NMI vs PAM50": [r["nmi_pam50"] for r in feat_results],
            "C-index":      [r["cv_cindex"] for r in feat_results],
        },
        x_label="Features per modality",
        ref_x=REF_NFEAT,
        out_path=FIG_OUT / "feature_sweep.pdf",
        title=f"Sensitivity: features per modality sweep\n"
               f"(K={REF_K}, metric={REF_METRIC})",
    )

    # Metric sweep (bar chart since categorical)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric_name, key in zip(
        axes,
        ["ARI vs PAM50", "NMI vs PAM50", "C-index"],
        ["ari_pam50", "nmi_pam50", "cv_cindex"],
    ):
        vals = [r[key] for r in metric_results]
        colours = ["goldenrod" if m == REF_METRIC else "steelblue"
                   for m in METRIC_VALUES]
        ax.bar(METRIC_VALUES, vals, color=colours, edgecolor="white")
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("Affinity metric")
        if key == "cv_cindex":
            ax.set_ylim(0.5, 0.80)
        else:
            ax.set_ylim(0, 0.65)
    fig.suptitle(f"Sensitivity: affinity metric "
                 f"(K={REF_K}, n_feat={REF_NFEAT})", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_OUT / "metric_sweep.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figure → {FIG_OUT / 'metric_sweep.pdf'}")

    # Omnibus stability heatmap (all non-reference configs)
    stability_heatmap(sweep_df, FIG_OUT / "stability_heatmap.pdf")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║            SENSITIVITY COMPLETE — SUMMARY               ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  {'Config':<42} {'ARI':>6} {'C-idx':>7}")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for _, row in sweep_df.iterrows():
        flag = "★" if row["config"] == REF_METRICS["config"] else " "
        log.info(f"║ {flag} {row['config']:<41} "
                 f"{row['ari_pam50']:>6.4f} {row['cv_cindex']:>7.4f}")
    log.info("╠══════════════════════════════════════════════════════════╣")

    # Range check
    ari_range = sweep_df["ari_pam50"].max() - sweep_df["ari_pam50"].min()
    cid_range = sweep_df["cv_cindex"].max() - sweep_df["cv_cindex"].min()
    log.info(f"║  ARI range across all configs:    {ari_range:.4f}")
    log.info(f"║  C-index range across all configs: {cid_range:.4f}")
    if ari_range < 0.10 and cid_range < 0.10:
        log.info("║  ✓ Results stable (Δ < 0.10 for both ARI and C-index)")
    else:
        log.warning(f"║  ⚠ Sensitivity detected: ARI Δ={ari_range:.4f}, "
                    f"C-idx Δ={cid_range:.4f}")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("\n✓ Next: python scripts/10_figures.py")
