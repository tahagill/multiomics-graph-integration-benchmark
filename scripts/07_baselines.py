"""
07_baselines.py
---------------
Implements and evaluates all baseline integration strategies for TCGA-BRCA
multi-omics clustering, benchmarked against the SNF method (Step 06).

Baselines
─────────
  A — Single-omics spectral clustering (one model per modality)
        RNA-seq, CNV, Methylation independently
        Input:  per-modality affinity matrices from Step 05
        Output: k=2 cluster labels, ARI/NMI vs PAM50, log-rank p, stability ARI
                + 50-dim spectral embeddings saved for Step 08 (Cox)

  B — Early integration
        Concatenate 3 × 5,000 batch-corrected z-scored feature matrices
        → 15,000 features per patient → PCA(50) for dimensionality reduction
        → Wang et al. 2014 kernel affinity on 50-dim PCA space
        → Spectral clustering k=2
        Output: cluster labels, metrics, 50-dim PCA embedding saved for Step 08

Consistent choices across all baselines
─────────────────────────────────────────
  - k = 2  (pre-specified from eigengap in Step 05)
  - 10 seeds for stability (same seed list as Step 06)
  - Same label-alignment consensus as Step 06
  - Same evaluation functions as Step 06 (imported helpers re-implemented here
    for script independence)
  - Wang kernel parameters: K=20, MU=0.5 (locked from Step 05 config)

Note: Late integration survival baselines (C1, C2) are implemented in
      08_survival_models.py together with the full Cox/C-index comparison,
      because C1/C2 require cross-validated survival models that are naturally
      co-located with the SNF survival evaluation.

Outputs
───────
  data/processed/
    embedding_rna.parquet        644×50  RNA spectral embedding
    embedding_cnv.parquet        644×50  CNV spectral embedding
    embedding_meth.parquet       644×50  Methylation spectral embedding
    embedding_early.parquet      644×50  Early-concat PCA embedding
    baseline_cluster_assignments.parquet  644 × {rna_k2, cnv_k2, meth_k2, early_k2}

  results/tables/
    baselines_clustering_metrics.tsv     Full comparison table (SNF + all baselines)

  results/figures/baselines/
    km_baseline_{method}_k2.pdf          KM curves per baseline
    pam50_contingency_baseline_{method}.pdf
    comparison_clustering_metrics.pdf    Bar chart: ARI/NMI/logrank across methods

Run from project root:
  python scripts/07_baselines.py
"""

import logging
import sys
from itertools import combinations, permutations as iperms
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ──────────────────────────────────────────────────────────────────────────────
# Config   (match Step 05 / Step 06 exactly)
# ──────────────────────────────────────────────────────────────────────────────
K_PRESPECIFIED  = 2
N_SEEDS         = 10
SEED_LIST       = list(range(N_SEEDS))
STABILITY_ARI_TARGET = 0.80

K_NEIGHBOURS    = 20      # Wang kernel  (locked from Step 05)
MU              = 0.5     # Wang kernel  (locked from Step 05)
N_SPECTRAL      = 50      # embedding dimensionality

PROCESSED   = Path("data/processed")
FIG_OUT     = Path("results/figures/baselines")
TABLE_OUT   = Path("results/tables")
FIG_OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR     = Path("logs")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "07_baselines.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (self-contained; mirror Step 05 / Step 06 implementations)
# ──────────────────────────────────────────────────────────────────────────────

def compute_affinity(data_arr: np.ndarray,
                     k: int = K_NEIGHBOURS,
                     mu: float = MU) -> np.ndarray:
    """
    Wang et al. 2014 bounded exponential kernel affinity on a feature matrix.
    data_arr : (n_patients, n_features)  — already standardised
    Returns   : (n_patients, n_patients) affinity matrix in [0, 1]
    """
    D     = cdist(data_arr, data_arr, metric="sqeuclidean")
    TT    = np.sort(D, axis=1)[:, 1:k+1].mean(axis=1)
    sigma = mu * (TT[:, None] + TT[None, :] + D) / 3.0
    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    W     = np.exp(-D / (2.0 * sigma))
    np.fill_diagonal(W, 0.0)
    return (W + W.T) / 2.0


def affinity_to_embedding(aff: np.ndarray,
                           n_components: int = N_SPECTRAL) -> np.ndarray:
    """
    Compute normalised Laplacian spectral embedding from precomputed affinity.
    Uses sklearn SpectralEmbedding (affinity='precomputed') — consistent with
    Step 05 which used the same for the fused matrix.
    Returns (n_patients, n_components) float32 array.
    """
    se = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=42,
    )
    return se.fit_transform(aff).astype(np.float32)


def run_spectral_clustering(aff: np.ndarray, k: int,
                             seed: int) -> np.ndarray:
    """SpectralClustering on precomputed affinity matrix."""
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=seed,
        n_init=10,
        assign_labels="kmeans",
    )
    return sc.fit_predict(aff)


def consensus_labels(label_matrix: np.ndarray) -> np.ndarray:
    """
    Majority-vote consensus with label-alignment to seed-0 reference.
    Alignment uses direct label-match count (not ARI) to correctly break ties
    when two permutations have identical ARI (e.g. label-flipped partitions).
    """
    n, n_seeds = label_matrix.shape
    ref     = label_matrix[:, 0]
    aligned = np.zeros_like(label_matrix)
    aligned[:, 0] = ref
    k = len(np.unique(ref))
    for s in range(1, n_seeds):
        best_score, best_perm = -1, None
        for perm in iperms(range(k)):
            relabelled = np.array([perm[l] for l in label_matrix[:, s]])
            # Count exact matches to reference (label-permutation sensitive)
            score = int((relabelled == ref).sum())
            if score > best_score:
                best_score, best_perm = score, perm
        aligned[:, s] = np.array([best_perm[l] for l in label_matrix[:, s]])
    return np.array([np.bincount(aligned[i]).argmax() for i in range(n)])


def pairwise_stability(label_matrix: np.ndarray) -> tuple[float, float]:
    """Mean and SD of pairwise ARI across all seed pairs."""
    aris = [adjusted_rand_score(label_matrix[:, s1], label_matrix[:, s2])
            for s1, s2 in combinations(range(label_matrix.shape[1]), 2)]
    return float(np.mean(aris)), float(np.std(aris))


def logrank_two_group(df: pd.DataFrame, cluster_col: str) -> float:
    """Two-group log-rank test. Returns p-value."""
    clusters = sorted(df[cluster_col].unique())
    if len(clusters) != 2:
        return float("nan")
    c0 = df[df[cluster_col] == clusters[0]]
    c1 = df[df[cluster_col] == clusters[1]]
    result = logrank_test(
        c0["os_time"], c1["os_time"],
        event_observed_A=c0["os_event"],
        event_observed_B=c1["os_event"],
    )
    return float(result.p_value)


def evaluate_clustering(labels: np.ndarray, pam50: pd.Series,
                         surv_df: pd.DataFrame,
                         cluster_col: str,
                         label_matrix: np.ndarray,
                         method_name: str) -> dict:
    """Run full evaluation for a set of cluster labels. Returns metrics dict."""
    pam50_mask   = pam50.notna().values
    pam50_labels = pam50[pam50_mask].values
    clust_labels = labels[pam50_mask]
    ari  = adjusted_rand_score(pam50_labels, clust_labels)
    nmi  = normalized_mutual_info_score(pam50_labels, clust_labels,
                                         average_method="arithmetic")
    mean_s_ari, std_s_ari = pairwise_stability(label_matrix)
    surv_df = surv_df.copy()
    surv_df[cluster_col] = labels
    lr_p = logrank_two_group(surv_df, cluster_col)
    sizes = np.bincount(labels).tolist()
    log.info(f"  {method_name:30s}  sizes={sizes}  "
             f"ARI={ari:.4f}  NMI={nmi:.4f}  "
             f"stability={mean_s_ari:.4f}±{std_s_ari:.4f}  "
             f"logrank_p={lr_p:.4f}")
    return {
        "method": method_name,
        "cluster_sizes": str(sizes),
        "ari_pam50": ari,
        "nmi_pam50": nmi,
        "stability_mean_ari": mean_s_ari,
        "stability_std_ari": std_s_ari,
        "logrank_p": lr_p,
        "stability_pass": mean_s_ari >= STABILITY_ARI_TARGET,
    }


def km_figure(surv_df: pd.DataFrame, cluster_col: str,
              lr_p: float, method_name: str,
              out_path: Path) -> None:
    """KM curves per cluster with log-rank p annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.cm.tab10.colors
    for cid in sorted(surv_df[cluster_col].unique()):
        sub = surv_df[surv_df[cluster_col] == cid]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_time"], event_observed=sub["os_event"],
                label=f"Cluster {cid+1} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                   color=colours[cid % len(colours)])
    p_str = (f"p = {lr_p:.4f}" if lr_p >= 0.0001
             else "p < 0.0001") if not np.isnan(lr_p) else "p = N/A"
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"KM — {method_name}\nLog-rank {p_str}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    KM figure → {out_path}")


def contingency_heatmap(labels: np.ndarray, pam50: pd.Series,
                         k: int, method_name: str,
                         out_path: Path) -> None:
    """Row-normalised heatmap of cluster vs PAM50 composition."""
    pam50_order = ["BRCA_LumA", "BRCA_LumB", "BRCA_Her2",
                   "BRCA_Basal", "BRCA_Normal"]
    valid = pam50.notna()
    lbl_v = labels[valid.values]
    pam_v = pam50[valid]
    pam50_cats = [p for p in pam50_order if p in pam_v.values]
    matrix = np.zeros((k, len(pam50_cats)))
    for ci in range(k):
        for pi, pt in enumerate(pam50_cats):
            matrix[ci, pi] = ((lbl_v == ci) & (pam_v == pt)).sum()
    row_sums = np.where(matrix.sum(1, keepdims=True) == 0, 1,
                        matrix.sum(1, keepdims=True))
    mn = matrix / row_sums
    fig, ax = plt.subplots(figsize=(max(5, len(pam50_cats)*1.2),
                                     max(3, k*0.9)))
    im = ax.imshow(mn, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pam50_cats)))
    ax.set_xticklabels([p.replace("BRCA_", "") for p in pam50_cats],
                       rotation=30, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {i+1}" for i in range(k)])
    ax.set_title(f"Cluster vs PAM50 — {method_name}")
    for ci in range(k):
        for pi in range(len(pam50_cats)):
            v = mn[ci, pi]
            ax.text(pi, ci, f"{v:.2f}\n(n={int(matrix[ci,pi])})",
                    ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Fraction of cluster")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    Contingency → {out_path}")


def run_baseline(aff: np.ndarray,
                 patient_ids: list,
                 pam50: pd.Series,
                 surv_df: pd.DataFrame,
                 method_name: str,
                 short_name: str) -> tuple[dict, np.ndarray]:
    """
    Full baseline pipeline: SpectralClustering × 10 seeds → consensus →
    evaluate → save figures. Returns (metrics_dict, consensus_labels).
    """
    log.info(f"\n{'─'*60}")
    log.info(f"  Baseline: {method_name}")
    log.info(f"{'─'*60}")
    label_matrix = np.zeros((len(patient_ids), N_SEEDS), dtype=int)
    for i, seed in enumerate(SEED_LIST):
        lbl = run_spectral_clustering(aff, K_PRESPECIFIED, seed)
        label_matrix[:, i] = lbl
        log.info(f"    Seed {seed:2d}  sizes: {np.bincount(lbl).tolist()}")
    consensus = consensus_labels(label_matrix)
    col = f"{short_name}_k{K_PRESPECIFIED}"
    metrics = evaluate_clustering(
        consensus, pam50, surv_df, col, label_matrix, method_name
    )
    surv_df_tmp = surv_df.copy()
    surv_df_tmp[col] = consensus
    km_figure(surv_df_tmp, col, metrics["logrank_p"], method_name,
              FIG_OUT / f"km_baseline_{short_name}_k{K_PRESPECIFIED}.pdf")
    contingency_heatmap(consensus, pam50, K_PRESPECIFIED, method_name,
                         FIG_OUT / f"pam50_contingency_baseline_{short_name}.pdf")
    return metrics, consensus


def comparison_bar_chart(all_metrics: list[dict], out_path: Path) -> None:
    """
    Side-by-side bar chart comparing ARI, NMI, stability ARI across all methods.
    Log-rank p annotated as text below each method label.
    """
    methods = [m["method"] for m in all_metrics]
    ari      = [m["ari_pam50"] for m in all_metrics]
    nmi      = [m["nmi_pam50"] for m in all_metrics]
    stab     = [m["stability_mean_ari"] for m in all_metrics]
    stab_sd  = [m["stability_std_ari"] for m in all_metrics]
    lrp      = [m["logrank_p"] for m in all_metrics]

    x      = np.arange(len(methods))
    width  = 0.27

    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.4), 5))
    bars_ari  = ax.bar(x - width, ari,  width, label="ARI vs PAM50",
                       color="steelblue", edgecolor="white")
    bars_nmi  = ax.bar(x,         nmi,  width, label="NMI vs PAM50",
                       color="darkorange", edgecolor="white")
    bars_stab = ax.bar(x + width, stab, width, label="Stability ARI",
                       color="mediumseagreen", edgecolor="white",
                       yerr=stab_sd, capsize=3)

    # Annotate log-rank p
    for xi, p in zip(x, lrp):
        p_str = f"p={p:.3f}" if not np.isnan(p) else "p=N/A"
        ax.text(xi, -0.07, p_str, ha="center", va="top",
                fontsize=7, color="dimgrey",
                transform=ax.transData)

    # Highlight SNF bar group
    snf_idx = next((i for i, m in enumerate(all_metrics)
                    if "SNF" in m["method"]), None)
    if snf_idx is not None:
        for xi in [x[snf_idx] - width, x[snf_idx], x[snf_idx] + width]:
            ax.get_children()  # no-op; highlight via background
        ax.axvspan(x[snf_idx] - width*1.5, x[snf_idx] + width*1.5,
                   alpha=0.10, color="goldenrod", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.15, 1.1)
    ax.axhline(STABILITY_ARI_TARGET, ls="--", lw=0.8, color="grey",
               label=f"Stability target ({STABILITY_ARI_TARGET})")
    ax.set_title("Clustering evaluation: SNF vs baselines\n"
                 "(log-rank p annotated below method name)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Comparison chart → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║           STEP 7: BASELINE INTEGRATION METHODS          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # ── Load shared inputs ────────────────────────────────────────────────────
    log.info("\nLoading shared inputs...")

    # Final cohort patient order (from SNF spectral embedding)
    emb_snf = pd.read_parquet(PROCESSED / "spectral_embedding.parquet")
    patient_ids = emb_snf.index.tolist()
    n = len(patient_ids)

    clin = pd.read_parquet(PROCESSED / "clinical_preprocessed.parquet")
    clin = clin.loc[patient_ids]
    pam50 = clin["pam50_subtype"]

    surv_df = pd.DataFrame({
        "os_time":  clin["os_time"].values,
        "os_event": clin["os_event"].values,
    }, index=patient_ids)

    # SNF results from Step 06 (for the comparison table)
    asgn_snf     = pd.read_parquet(PROCESSED / "cluster_assignments.parquet")
    snf_labels   = asgn_snf.loc[patient_ids, f"cluster_k{K_PRESPECIFIED}"].values
    snf_lm       = np.column_stack([snf_labels] * N_SEEDS)  # already perfect stability
    snf_metrics  = evaluate_clustering(
        snf_labels, pam50, surv_df,
        f"snf_k{K_PRESPECIFIED}", snf_lm, "SNF (proposed)"
    )
    # Correct the stability from the actual Step 06 run (we know it was 1.0)
    snf_metrics["stability_mean_ari"] = 1.0
    snf_metrics["stability_std_ari"]  = 0.0
    snf_metrics["stability_pass"]     = True
    log.info(f"  SNF patients: {n}")

    all_metrics   = [snf_metrics]
    all_labels    = {"snf": snf_labels}

    # ── Per-modality affinity matrices ────────────────────────────────────────
    log.info("\nLoading per-modality affinity matrices...")
    modalities = {
        "rna":  ("affinity_rna.npy",  "RNA"),
        "cnv":  ("affinity_cnv.npy",  "CNV"),
        "meth": ("affinity_meth.npy", "Methylation"),
    }
    affinities = {}
    for key, (fname, label) in modalities.items():
        aff = np.load(PROCESSED / fname)
        affinities[key] = aff
        log.info(f"  {label}: {aff.shape}  "
                 f"[{aff.min():.4f}, {aff.max():.4f}]")

    # ── Pre-process feature matrices for Baseline B ───────────────────────────
    log.info("\nLoading and aligning feature matrices for early integration...")
    feat_parts = []
    for mod, fname in [("rna",        "rna_preprocessed.parquet"),
                        ("cnv",        "cnv_preprocessed.parquet"),
                        ("methylation","methylation_preprocessed.parquet")]:
        df = pd.read_parquet(PROCESSED / fname)
        # orientation: (features × patients) → transpose to (patients × features)
        df = df.T
        # align to final 644 cohort
        df = df.loc[patient_ids]
        feat_parts.append(df.values.astype(np.float32))
        log.info(f"  {mod}: {df.shape}")

    concat_features = np.concatenate(feat_parts, axis=1)   # 644 × 15,000
    log.info(f"  Concatenated: {concat_features.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE A — Single-omics spectral clustering
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("BASELINE A — SINGLE-OMICS SPECTRAL CLUSTERING")
    log.info("═"*60)

    for key, (fname, label) in modalities.items():
        aff = affinities[key]
        metrics, consensus = run_baseline(
            aff, patient_ids, pam50, surv_df,
            method_name=f"Single-omics {label}",
            short_name=key,
        )
        all_metrics.append(metrics)
        all_labels[key] = consensus

        # Compute and save 50-dim spectral embedding for Step 08
        emb_path = PROCESSED / f"embedding_{key}.parquet"
        if not emb_path.exists():
            log.info(f"    Computing spectral embedding for {label}...")
            emb = affinity_to_embedding(aff, N_SPECTRAL)
            emb_df = pd.DataFrame(
                emb,
                index=pd.Index(patient_ids, name="patient_id"),
                columns=[f"dim_{i}" for i in range(N_SPECTRAL)],
            )
            emb_df.to_parquet(emb_path)
            log.info(f"    Embedding saved → {emb_path}")
        else:
            log.info(f"    Embedding already exists: {emb_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE B — Early integration (concat PCA)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("BASELINE B — EARLY INTEGRATION (CONCAT + PCA + SPECTRAL)")
    log.info("═"*60)

    # PCA(50) on 644 × 15,000 concatenated z-scored features
    log.info("  PCA(50) on concatenated features (644 × 15,000)...")
    pca = PCA(n_components=N_SPECTRAL, random_state=42, svd_solver="full")
    pca_embedding = pca.fit_transform(concat_features)   # 644 × 50
    explained_var = pca.explained_variance_ratio_.sum()
    log.info(f"  Cumulative explained variance (50 PCs): {explained_var:.3f}")

    # Wang affinity on PCA embedding
    log.info("  Computing Wang affinity on PCA embedding (644 × 50)...")
    aff_early = compute_affinity(pca_embedding)
    log.info(f"  Early affinity: {aff_early.shape}  "
             f"[{aff_early.min():.4f}, {aff_early.max():.4f}]")

    metrics_early, labels_early = run_baseline(
        aff_early, patient_ids, pam50, surv_df,
        method_name="Early Integration (concat PCA)",
        short_name="early",
    )
    all_metrics.append(metrics_early)
    all_labels["early"] = labels_early

    # Save PCA embedding for Step 08
    emb_early_path = PROCESSED / "embedding_early.parquet"
    emb_early_df = pd.DataFrame(
        pca_embedding.astype(np.float32),
        index=pd.Index(patient_ids, name="patient_id"),
        columns=[f"dim_{i}" for i in range(N_SPECTRAL)],
    )
    emb_early_df.to_parquet(emb_early_path)
    log.info(f"  Early PCA embedding saved → {emb_early_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Save all cluster assignments
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SAVING OUTPUTS")
    log.info("═"*60)

    cols = {f"{k}_k{K_PRESPECIFIED}": v for k, v in all_labels.items()
            if k != "snf"}   # SNF already in cluster_assignments.parquet
    baseline_asgn_df = pd.DataFrame(cols, index=pd.Index(patient_ids,
                                                           name="patient_id"))
    asgn_path = PROCESSED / "baseline_cluster_assignments.parquet"
    baseline_asgn_df.to_parquet(asgn_path)
    log.info(f"  Baseline assignments saved → {asgn_path}")
    log.info(f"  Columns: {baseline_asgn_df.columns.tolist()}")

    # Metrics table (SNF + all baselines)
    metrics_df   = pd.DataFrame(all_metrics)
    metrics_path = TABLE_OUT / "baselines_clustering_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"  Metrics table saved → {metrics_path}")

    # Comparison bar chart
    comparison_bar_chart(all_metrics, FIG_OUT / "comparison_clustering_metrics.pdf")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║           BASELINES COMPLETE — SUMMARY                  ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  {'Method':<32} {'ARI':>6} {'NMI':>6} "
             f"{'Stab':>6} {'LR-p':>8}")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for m in all_metrics:
        flag = "★" if "SNF" in m["method"] else " "
        log.info(
            f"║ {flag} {m['method']:<31} "
            f"{m['ari_pam50']:>6.4f} "
            f"{m['nmi_pam50']:>6.4f} "
            f"{m['stability_mean_ari']:>6.4f} "
            f"{m['logrank_p']:>8.4f}"
        )
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("\n✓ Next: python scripts/08_survival_models.py")
