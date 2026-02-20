"""
06_clustering.py
----------------
Spectral clustering on the SNF-fused patient similarity matrix.
Evaluates k via silhouette score (k=2..6), confirms eigengap pre-specified k,
runs clustering across 10 seeds for stability, evaluates biological validity
via ARI/NMI against PAM50, and produces Kaplan-Meier survival curves.

Pipeline
────────
  1.  Load fused affinity matrix (644×644) and spectral embedding (644×50)
  2.  k-selection: silhouette score for k=2..6 on embedding space
      → compare with eigengap pre-specified k=2
  3.  Primary clustering: SpectralClustering(k=2) on fused affinity × 10 seeds
      → consensus assignment = majority vote across seeds
  4.  Stability: pairwise ARI across all 10 seeds → report mean ± SD
  5.  Biological validation: ARI + NMI vs PAM50 (568 labelled patients)
  6.  Survival analysis:
        a. Kaplan-Meier curves per cluster (+ 95% CI band)
        b. Log-rank test; p-value annotated on figure
        c. Median OS per cluster with 95% CI
  7.  Additional k-sweep clustering (k=3..6) for supplementary comparison
  8.  Save primary cluster assignments → data/processed/cluster_assignments.parquet
  9.  Save all figures → results/figures/clustering/
  10. Save metrics table → results/tables/clustering_metrics.tsv

Design decisions
────────────────
  - SpectralClustering operates on precomputed affinity matrix (not embedding).
    sklearn's SpectralClustering(affinity='precomputed') takes the fused 644×644
    matrix directly, applies the normalised Laplacian, and runs k-means internally.
    This is the correct spectral clustering pipeline.
  - Silhouette scores are computed in the embedding space (644×k_dims using the
    top-k eigenvectors), not on the full 50-dim embedding. This matches the
    internal k-means space of SpectralClustering.
  - Stability (10 seeds): seed range [0..9] passed to SpectralClustering via
    random_state which controls k-means initialisation.
  - Consensus labels: for each patient, the cluster label assigned most
    frequently across 10 seed runs (majority vote). Ties resolved by lowest
    label index.
  - PAM50 evaluation is on the 568-patient labelled subset only.

Outputs
───────
  data/processed/
    cluster_assignments.parquet   644 × {cluster_k2, ..., cluster_k6}
  results/tables/
    clustering_metrics.tsv        ARI, NMI, logrank p, median OS per cluster,
                                  stability mean ARI, silhouette scores
  results/figures/clustering/
    silhouette_kselection.pdf     Silhouette vs k bar chart
    km_cluster_k{k}.pdf           KM curves for k=2..6
    pam50_contingency_k{k}.pdf    Heatmap: cluster vs PAM50 (k=2..6)

Run from project root:
  python scripts/06_clustering.py
"""

import logging
import sys
from itertools import combinations
from itertools import permutations as iperms
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
K_PRESPECIFIED  = 2          # from eigengap heuristic (Step 05, pre-specified)
K_RANGE         = range(2, 7)  # k=2..6 evaluated for silhouette + supplementary
N_SEEDS         = 10         # repeat clustering across this many random seeds
SEED_LIST       = list(range(N_SEEDS))
STABILITY_ARI_TARGET = 0.80  # README success criterion

PROCESSED   = Path("data/processed")
FIG_OUT     = Path("results/figures/clustering")
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
        logging.FileHandler(LOG_DIR / "06_clustering.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def silhouette_in_embedding(aff: np.ndarray, labels: np.ndarray,
                             k: int) -> float:
    """
    Compute silhouette score in the normalised Laplacian eigenvector space
    (the same space used internally by SpectralClustering).
    Uses top-k eigenvectors of the normalised Laplacian built from aff.
    """
    # Degree matrix
    d = aff.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    # Normalised random-walk affinity: D^{-1/2} A D^{-1/2}
    norm_aff = d_inv_sqrt[:, None] * aff * d_inv_sqrt[None, :]
    # Top-k eigenvectors (largest eigenvalues = most connected)
    svd = TruncatedSVD(n_components=k, random_state=42)
    eigvecs = svd.fit_transform(norm_aff)   # 644 × k
    # Row-normalise (standard in spectral clustering)
    norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    eigvecs_norm = eigvecs / norms
    return float(silhouette_score(eigvecs_norm, labels, metric="euclidean"))


def run_spectral_clustering(aff: np.ndarray, k: int,
                             seed: int) -> np.ndarray:
    """Run SpectralClustering on precomputed affinity, return label array."""
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
    Majority-vote consensus across N_SEEDS cluster runs.
    label_matrix shape: (n_patients, N_SEEDS).
    Returns (n_patients,) integer array.
    Alignment uses direct label-match count (not ARI) to correctly break ties
    when two permutations have identical ARI (e.g. label-flipped partitions).
    """
    n, n_seeds = label_matrix.shape
    ref = label_matrix[:, 0]
    aligned = np.zeros_like(label_matrix)
    aligned[:, 0] = ref
    k = len(np.unique(ref))

    for s in range(1, n_seeds):
        best_score, best_perm = -1, None
        for perm in iperms(range(k)):
            relabelled = np.array([perm[l] for l in label_matrix[:, s]])
            score = int((relabelled == ref).sum())
            if score > best_score:
                best_score, best_perm = score, perm
        aligned[:, s] = np.array([best_perm[l] for l in label_matrix[:, s]])

    # Majority vote per patient
    consensus = np.array([
        np.bincount(aligned[i]).argmax() for i in range(n)
    ])
    return consensus


def pairwise_stability(label_matrix: np.ndarray) -> tuple[float, float]:
    """
    Mean and SD of pairwise ARI across all N_SEEDS*(N_SEEDS-1)/2 pairs.
    Returns (mean_ari, std_ari).
    """
    n_seeds = label_matrix.shape[1]
    aris = []
    for s1, s2 in combinations(range(n_seeds), 2):
        aris.append(adjusted_rand_score(label_matrix[:, s1],
                                        label_matrix[:, s2]))
    return float(np.mean(aris)), float(np.std(aris))


def km_plot(df: pd.DataFrame, cluster_col: str, k: int,
            logrank_p: float, median_os: dict,
            out_path: Path) -> None:
    """
    Kaplan-Meier curves per cluster with 95% CI, log-rank p annotation.
    df must have columns: os_time, os_event, cluster_col.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.cm.tab10.colors

    for cid in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cid]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_time"], event_observed=sub["os_event"],
                label=f"Cluster {cid + 1} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                   color=colours[cid % len(colours)])
        med = median_os.get(cid, float("nan"))
        med_str = f"{med:.0f}d" if not np.isnan(med) else "NR"
        ax.axvline(x=med, ls=":", lw=1.0, color=colours[cid % len(colours)],
                   alpha=0.5)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    p_str = f"p = {logrank_p:.4f}" if logrank_p >= 0.0001 else "p < 0.0001"
    ax.set_title(f"Kaplan-Meier — k={k} clusters\nLog-rank {p_str}",
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  KM figure saved → {out_path}")


def contingency_heatmap(labels: np.ndarray, pam50: pd.Series,
                         k: int, out_path: Path) -> None:
    """
    Heatmap of cluster assignment vs PAM50 subtype (labelled patients only).
    Row-normalised (fraction of each cluster in each PAM50 subtype).
    """
    pam50_order = ["BRCA_LumA", "BRCA_LumB", "BRCA_Her2",
                   "BRCA_Basal", "BRCA_Normal"]
    valid_mask = pam50.notna()
    lbl_sub   = labels[valid_mask]
    pam_sub   = pam50[valid_mask]

    n_clusters        = k
    pam50_cats        = [p for p in pam50_order if p in pam_sub.values]
    matrix            = np.zeros((n_clusters, len(pam50_cats)))
    for ci in range(n_clusters):
        for pi, ptype in enumerate(pam50_cats):
            matrix[ci, pi] = ((lbl_sub == ci) & (pam_sub == ptype)).sum()

    # Row-normalise
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    matrix_normed = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(5, len(pam50_cats) * 1.2),
                                     max(3, n_clusters * 0.9)))
    im = ax.imshow(matrix_normed, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(pam50_cats)))
    ax.set_xticklabels([p.replace("BRCA_", "") for p in pam50_cats],
                       rotation=30, ha="right")
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {i+1}" for i in range(n_clusters)])
    ax.set_xlabel("PAM50 subtype")
    ax.set_ylabel("SNF cluster")
    ax.set_title(f"Cluster vs PAM50 composition (k={k}, row-normalised)")
    for ci in range(n_clusters):
        for pi in range(len(pam50_cats)):
            val = matrix_normed[ci, pi]
            n_val = int(matrix[ci, pi])
            ax.text(pi, ci, f"{val:.2f}\n(n={n_val})",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if val > 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Fraction of cluster")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Contingency heatmap saved → {out_path}")


def logrank_multigroup(df: pd.DataFrame, cluster_col: str) -> float:
    """
    For k=2: standard two-group log-rank test.
    For k>2: overall multivariate log-rank test (lifelines multivariate_logrank_test).
    Returns p-value (float).
    """
    clusters = sorted(df[cluster_col].unique())
    if len(clusters) == 2:
        c0 = df[df[cluster_col] == clusters[0]]
        c1 = df[df[cluster_col] == clusters[1]]
        result = logrank_test(
            c0["os_time"], c1["os_time"],
            event_observed_A=c0["os_event"],
            event_observed_B=c1["os_event"],
        )
        return float(result.p_value)
    else:
        result = multivariate_logrank_test(
            df["os_time"], df[cluster_col], df["os_event"]
        )
        return float(result.p_value)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║       STEP 6: SPECTRAL CLUSTERING + EVALUATION          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Pre-specified k (eigengap): {K_PRESPECIFIED}")
    log.info(f"k range evaluated:          k={K_RANGE.start}..{K_RANGE.stop-1}")
    log.info(f"Seeds:                      {N_SEEDS}  (seed list: {SEED_LIST})")
    log.info(f"Stability ARI target:       ≥ {STABILITY_ARI_TARGET}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("Loading data...")
    aff_fused = np.load(PROCESSED / "affinity_fused.npy")
    log.info(f"  Fused affinity: {aff_fused.shape}  "
             f"range=[{aff_fused.min():.4f}, {aff_fused.max():.4f}]")

    emb_df = pd.read_parquet(PROCESSED / "spectral_embedding.parquet")
    patient_ids = emb_df.index.tolist()
    log.info(f"  Spectral embedding: {emb_df.shape}")

    clin = pd.read_parquet(PROCESSED / "clinical_preprocessed.parquet")
    clin = clin.loc[patient_ids]   # align to embedding order
    pam50 = clin["pam50_subtype"]
    n_pam50 = int(pam50.notna().sum())
    log.info(f"  Clinical: {clin.shape}  PAM50 labelled: {n_pam50}/{len(clin)}")

    # ── Silhouette k-selection ────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SILHOUETTE K-SELECTION")
    log.info("=" * 60)
    log.info("Computing silhouette score in normalised Laplacian space for "
             f"k={K_RANGE.start}..{K_RANGE.stop-1}  (single run, seed=0)")

    sil_scores = {}
    for k in K_RANGE:
        labels_k = run_spectral_clustering(aff_fused, k, seed=0)
        sil = silhouette_in_embedding(aff_fused, labels_k, k)
        sil_scores[k] = sil
        log.info(f"  k={k}  silhouette={sil:.4f}")

    best_sil_k = max(sil_scores, key=sil_scores.get)
    log.info(f"  Best silhouette k = {best_sil_k}  "
             f"(score={sil_scores[best_sil_k]:.4f})")
    log.info(f"  Eigengap pre-specified k = {K_PRESPECIFIED}")
    if best_sil_k == K_PRESPECIFIED:
        log.info(f"  ✓ Silhouette confirms pre-specified k={K_PRESPECIFIED}")
    else:
        log.warning(f"  ⚠ Silhouette suggests k={best_sil_k} "
                    f"vs eigengap k={K_PRESPECIFIED}. "
                    f"Primary analysis uses pre-specified k={K_PRESPECIFIED}; "
                    f"k={best_sil_k} reported in supplementary.")

    # Silhouette figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = list(sil_scores.keys())
    vals = [sil_scores[k] for k in ks]
    colours = ["tab:red" if k == K_PRESPECIFIED else "tab:blue" for k in ks]
    ax.bar(ks, vals, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("k-selection: silhouette score\n"
                 f"(red = eigengap pre-specified k={K_PRESPECIFIED})")
    ax.set_xticks(ks)
    for k, v in zip(ks, vals):
        ax.text(k, v + 0.001, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
    plt.tight_layout()
    sil_fig_path = FIG_OUT / "silhouette_kselection.pdf"
    fig.savefig(sil_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Silhouette figure saved → {sil_fig_path}")

    # ── Primary clustering: k=K_PRESPECIFIED, 10 seeds ───────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(f"PRIMARY CLUSTERING  k={K_PRESPECIFIED}, {N_SEEDS} seeds")
    log.info("=" * 60)

    label_matrix = np.zeros((len(patient_ids), N_SEEDS), dtype=int)
    seed_sil = []
    for i, seed in enumerate(SEED_LIST):
        labels_s = run_spectral_clustering(aff_fused, K_PRESPECIFIED, seed)
        label_matrix[:, i] = labels_s
        s_sil = silhouette_in_embedding(aff_fused, labels_s, K_PRESPECIFIED)
        seed_sil.append(s_sil)
        log.info(f"  Seed {seed:2d}  cluster sizes: "
                 + str(np.bincount(labels_s).tolist())
                 + f"  silhouette={s_sil:.4f}")

    # Stability: pairwise ARI
    mean_ari, std_ari = pairwise_stability(label_matrix)
    log.info(f"  Pairwise ARI across {N_SEEDS} seeds: "
             f"mean={mean_ari:.4f}  SD={std_ari:.4f}")
    if mean_ari >= STABILITY_ARI_TARGET:
        log.info(f"  ✓ Stability criterion met (ARI ≥ {STABILITY_ARI_TARGET})")
    else:
        log.warning(f"  ⚠ Stability criterion NOT met "
                    f"(ARI={mean_ari:.4f} < {STABILITY_ARI_TARGET})")

    # Consensus labels (majority vote after label-alignment)
    log.info("  Computing consensus labels (majority vote, seed-0 reference)...")
    consensus = consensus_labels(label_matrix)
    log.info(f"  Consensus cluster sizes: {np.bincount(consensus).tolist()}")

    # ── ARI / NMI vs PAM50 ───────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("BIOLOGICAL VALIDATION vs PAM50")
    log.info("=" * 60)

    pam50_mask   = pam50.notna().values
    pam50_labels = pam50[pam50_mask].values
    clust_labels = consensus[pam50_mask]

    ari_pam50 = adjusted_rand_score(pam50_labels, clust_labels)
    nmi_pam50 = normalized_mutual_info_score(pam50_labels, clust_labels,
                                              average_method="arithmetic")
    log.info(f"  k={K_PRESPECIFIED} consensus vs PAM50 ({n_pam50} patients):")
    log.info(f"    ARI = {ari_pam50:.4f}")
    log.info(f"    NMI = {nmi_pam50:.4f}")

    # Contingency heatmap
    contingency_heatmap(
        consensus, pam50, K_PRESPECIFIED,
        FIG_OUT / f"pam50_contingency_k{K_PRESPECIFIED}.pdf"
    )

    # ── Survival analysis ─────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SURVIVAL ANALYSIS")
    log.info("=" * 60)

    surv_df      = clin[["os_time", "os_event"]].copy()
    surv_df.index = pd.Index(patient_ids)
    cluster_col  = f"cluster_k{K_PRESPECIFIED}"
    surv_df[cluster_col] = consensus

    # Median OS per cluster with 95% CI
    median_os = {}
    log.info(f"  Median OS per cluster (k={K_PRESPECIFIED}):")
    for cid in sorted(surv_df[cluster_col].unique()):
        sub = surv_df[surv_df[cluster_col] == cid]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_time"], event_observed=sub["os_event"])
        med = kmf.median_survival_time_
        median_os[cid] = med
        events  = int(sub["os_event"].sum())
        med_str = f"{med:.0f}" if not np.isnan(med) else "not reached"
        log.info(f"    Cluster {cid+1}: n={len(sub)}  events={events}  "
                 f"median OS = {med_str} days")

    logrank_p = logrank_multigroup(surv_df, cluster_col)
    log.info(f"  Log-rank p = {logrank_p:.6f}")

    # KM figure
    km_plot(
        surv_df, cluster_col, K_PRESPECIFIED, logrank_p, median_os,
        FIG_OUT / f"km_cluster_k{K_PRESPECIFIED}.pdf"
    )

    # ── Supplementary: k=3..6 ────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SUPPLEMENTARY CLUSTERING  k=3..6")
    log.info("=" * 60)

    all_assignments = {cluster_col: consensus}
    supp_metrics    = []

    for k in K_RANGE:
        if k == K_PRESPECIFIED:
            # Already done
            supp_metrics.append({
                "k": k, "primary": True,
                "silhouette": sil_scores[k],
                "stability_mean_ari": mean_ari,
                "stability_std_ari": std_ari,
                "ari_pam50": ari_pam50,
                "nmi_pam50": nmi_pam50,
                "logrank_p": logrank_p,
            })
            continue

        log.info(f"  k={k} supplementary run (seed=0 only)...")
        labels_k  = run_spectral_clustering(aff_fused, k, seed=0)
        col_k     = f"cluster_k{k}"
        all_assignments[col_k] = labels_k

        # ARI/NMI vs PAM50
        lbl_k_pam = labels_k[pam50_mask]
        ari_k = adjusted_rand_score(pam50_labels, lbl_k_pam)
        nmi_k = normalized_mutual_info_score(pam50_labels, lbl_k_pam,
                                              average_method="arithmetic")
        log.info(f"    Cluster sizes: {np.bincount(labels_k).tolist()}")
        log.info(f"    ARI vs PAM50 = {ari_k:.4f}   NMI = {nmi_k:.4f}")

        # Log-rank
        surv_df[col_k]  = labels_k
        lr_p_k          = logrank_multigroup(surv_df, col_k)
        log.info(f"    Log-rank p = {lr_p_k:.6f}")

        # Figures
        med_os_k = {}
        for cid in sorted(np.unique(labels_k)):
            sub = surv_df[surv_df[col_k] == cid]
            kmf = KaplanMeierFitter()
            kmf.fit(sub["os_time"], event_observed=sub["os_event"])
            med_os_k[cid] = kmf.median_survival_time_

        km_plot(surv_df, col_k, k, lr_p_k, med_os_k,
                FIG_OUT / f"km_cluster_k{k}.pdf")
        contingency_heatmap(labels_k, pam50, k,
                             FIG_OUT / f"pam50_contingency_k{k}.pdf")

        supp_metrics.append({
            "k": k, "primary": False,
            "silhouette": sil_scores[k],
            "stability_mean_ari": None,
            "stability_std_ari": None,
            "ari_pam50": ari_k,
            "nmi_pam50": nmi_k,
            "logrank_p": lr_p_k,
        })

    # ── Save outputs ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SAVING OUTPUTS")
    log.info("=" * 60)

    # Cluster assignments parquet
    assignments_df = pd.DataFrame(all_assignments, index=patient_ids)
    assignments_df.index.name = "patient_id"
    asgn_path = PROCESSED / "cluster_assignments.parquet"
    assignments_df.to_parquet(asgn_path)
    log.info(f"  Cluster assignments saved → {asgn_path}")
    log.info(f"  Columns: {assignments_df.columns.tolist()}")

    # Metrics table
    metrics_df = pd.DataFrame(sorted(supp_metrics, key=lambda x: x["k"]))
    metrics_path = TABLE_OUT / "clustering_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"  Metrics table saved → {metrics_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║              CLUSTERING COMPLETE — SUMMARY              ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Pre-specified k:        {K_PRESPECIFIED}")
    log.info(f"║  Best silhouette k:      {best_sil_k}  "
             f"(score={sil_scores[best_sil_k]:.4f})")
    log.info(f"║  Consensus cluster sizes: "
             f"{np.bincount(consensus).tolist()}")
    log.info(f"║  Stability (10 seeds):")
    log.info(f"║    Mean pairwise ARI = {mean_ari:.4f}  SD={std_ari:.4f}  "
             f"{'✓ PASS' if mean_ari >= STABILITY_ARI_TARGET else '✗ FAIL'}")
    log.info(f"║  PAM50 validation ({n_pam50} patients):")
    log.info(f"║    ARI = {ari_pam50:.4f}   NMI = {nmi_pam50:.4f}")
    log.info(f"║  Survival separation:")
    log.info(f"║    Log-rank p = {logrank_p:.6f}  "
             f"{'✓ p<0.05' if logrank_p < 0.05 else '✗ p≥0.05'}")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Silhouette scores by k:")
    for k, s in sil_scores.items():
        log.info(f"║    k={k}: {s:.4f}"
                 + (" ← pre-specified" if k == K_PRESPECIFIED else ""))
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"\n✓ Next: python scripts/07_baselines.py")
