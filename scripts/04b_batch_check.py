"""
04b_batch_check.py
------------------
Check for Tissue Source Site (TSS) batch structure in each preprocessed
modality. If meaningful batch structure is detected, apply pycombat correction
and overwrite the preprocessed parquet in-place.

Background
──────────
TCGA samples are collected at multiple Tissue Source Sites (TSS) — different
hospitals/biobanks identified by the 2nd field in the TCGA barcode (e.g. 'BH'
in 'TCGA-BH-A0AU'). Samples processed at different sites and times can carry
systematic technical variation that mimics biological signal, particularly in:
  - Methylation 450K (array batch / bisulphite treatment efficiency)
  - RNA-seq      (library prep protocol variation)
  - CNV          (smaller effect; segmentation algorithm is harmonised by GDC)

A batch effect in the top PCs means cluster structure downstream is at least
partially driven by which hospital sent the sample, not by tumour biology.

Method
──────
For each modality (RNA, CNV, methylation):
  1. Load preprocessed matrix; subset to 644-patient final cohort.
  2. PCA → top 10 PCs.
  3. Compute batch R² = variance in PC subspace explained by TSS label
     (η² = SS_between / SS_total, averaged across PC1–PC10).
  4. Plot PC1 vs PC2 coloured by TSS site.
  5. Decision rule:
       R² < 0.02  → no action (log OK)
       0.02–0.05  → weak signal (log warning, no correction)
       R² ≥ 0.05  → apply pycombat, overwrite parquet, delete
                    downstream affinity checkpoints

pycombat operates on features × samples (transpose convention). It adjusts
each feature independently using an empirical Bayes model of batch shift and
scale. We pass the z-scored matrix values directly; after ComBat the values
remain approximately z-scored but with TSS-driven mean/variance removed.

After any pycombat correction, ALL downstream affinity .npy files and the
spectral embedding parquet MUST be deleted so 05_graph_construction.py
recomputes them from the corrected data.

Run from project root:
  python scripts/04b_batch_check.py
"""

import gc
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
N_PCA_COMPONENTS    = 10     # PCs used for batch R² calculation
BATCH_R2_WARN       = 0.02   # R² above this → log warning
BATCH_R2_CORRECT    = 0.05   # R² above this → apply pycombat
TOP_TSS_IN_LEGEND   = 12     # colour top-N TSS sites; rest → "Other"

PROCESSED   = Path("data/processed")
FIG_OUT     = Path("results/figures/batch_check")
LOG_DIR     = Path("logs")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Downstream checkpoints that must be deleted if correction is applied
DOWNSTREAM_FILES = [
    PROCESSED / "affinity_rna.npy",
    PROCESSED / "affinity_cnv.npy",
    PROCESSED / "affinity_meth.npy",
    PROCESSED / "affinity_fused.npy",
    PROCESSED / "spectral_embedding.parquet",
    PROCESSED / "eigenvalues_fused.npy",
]

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04b_batch_check.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_tss(patient_ids: list[str]) -> pd.Series:
    """Extract Tissue Source Site code from TCGA barcode (field 2).
    E.g. 'TCGA-BH-A0AU' → 'BH'.
    """
    return pd.Series(
        [pid.split("-")[1] if "-" in pid else "UNK" for pid in patient_ids],
        index=patient_ids,
        name="tss",
    )


def batch_r2(pca_scores: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Mean η² (eta-squared) across PCs — the fraction of variance in each PC
    explained by the batch grouping variable.
        η² = SS_between / SS_total
    Averaged over all PCs in pca_scores.
    """
    unique_batches = np.unique(batch_labels)
    grand_mean = pca_scores.mean(axis=0)               # (n_pcs,)
    ss_total   = ((pca_scores - grand_mean) ** 2).sum(axis=0)  # (n_pcs,)
    ss_between = np.zeros(pca_scores.shape[1])
    for b in unique_batches:
        mask = batch_labels == b
        n_b  = mask.sum()
        if n_b == 0:
            continue
        group_mean = pca_scores[mask].mean(axis=0)
        ss_between += n_b * (group_mean - grand_mean) ** 2
    # Avoid divison by zero for constant PCs (shouldn't exist)
    eta2_per_pc = np.where(ss_total > 0, ss_between / ss_total, 0.0)
    return float(eta2_per_pc.mean())


def pca_and_plot(arr: np.ndarray, patient_ids: list,
                 tss: pd.Series, label: str,
                 r2: float) -> None:
    """Plot PC1 vs PC2 coloured by TSS; save figure."""
    pca  = PCA(n_components=2, random_state=42)
    pcs  = pca.fit_transform(arr)      # patients × 2

    tss_arr = tss.loc[patient_ids].values
    top_sites = (
        pd.Series(tss_arr).value_counts().head(TOP_TSS_IN_LEGEND).index.tolist()
    )

    # Build colour map
    cmap   = plt.cm.get_cmap("tab20", len(top_sites))
    colour_map = {site: cmap(i) for i, site in enumerate(top_sites)}
    colours = [
        colour_map.get(s, (0.6, 0.6, 0.6, 0.5))   # grey for "Other"
        for s in tss_arr
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colours, s=14, alpha=0.75,
               linewidths=0, rasterized=True)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    title_colour = "red" if r2 >= BATCH_R2_CORRECT else \
                   "darkorange" if r2 >= BATCH_R2_WARN else "black"
    ax.set_title(
        f"{label} — TSS batch check\n"
        f"Batch R² (η², PC1–{N_PCA_COMPONENTS}) = {r2:.3f}",
        color=title_colour
    )

    # Legend: top TSS + "Other"
    patches = [
        mpatches.Patch(color=colour_map[s], label=s) for s in top_sites
    ]
    patches.append(mpatches.Patch(color=(0.6, 0.6, 0.6), label="Other"))
    ax.legend(handles=patches, title="TSS", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=7, title_fontsize=8)

    plt.tight_layout()
    out_path = FIG_OUT / f"pca_tss_{label.lower()}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figure saved → {out_path}")


def apply_combat(arr: np.ndarray, patient_ids: list,
                 tss: pd.Series, label: str,
                 parquet_path: Path) -> np.ndarray:
    """
    Apply pycombat to the modality matrix.
    pycombat expects features × samples DataFrame.
    After correction, rewrite parquet as features × ALL-755-patients
    (only 644-cohort patients are corrected; non-cohort columns unchanged).

    Singleton batches (n=1 sample): pycombat cannot estimate within-batch
    variance for a single sample, producing NaN when it attempts variance
    normalisation. We detect this automatically and pass mean_only=True,
    which removes only the batch mean shift without rescaling variance.
    A post-correction NaN-fill with feature column means acts as a final
    safety net regardless.
    """
    log.info(f"  Applying pycombat to {label}...")
    try:
        from combat.pycombat import pycombat
    except ImportError:
        log.error("  pycombat not installed — run: pip install combat")
        return arr

    batch_labels = tss.loc[patient_ids].values
    batch_counts = pd.Series(batch_labels).value_counts()
    n_singletons = int((batch_counts == 1).sum())
    if n_singletons:
        log.warning(f"  {n_singletons} singleton batch(es) detected — "
                    f"using mean_only=True (skips variance normalisation)")
    mean_only = n_singletons > 0

    # pycombat needs pandas DataFrame, features × samples
    # arr is patients × features
    df_input = pd.DataFrame(
        arr.T.astype(np.float64),     # features × patients, float64
        columns=patient_ids
    )
    batch_series = pd.Series(batch_labels, index=patient_ids)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df_corrected = pycombat(
            df_input, batch_series, mean_only=mean_only
        )   # features × patients

    # Safety net: NaN-fill with feature column means (should not occur with
    # mean_only=True, but included as a belt-and-suspenders guard)
    n_nan = int(df_corrected.isna().sum().sum())
    if n_nan:
        log.warning(f"  pycombat produced {n_nan} NaN values — "
                    f"filling with feature column means")
        col_means = df_corrected.mean(axis=1)   # feature-wise mean
        df_corrected = df_corrected.T.fillna(col_means).T
        remaining_nan = int(df_corrected.isna().sum().sum())
        if remaining_nan:
            log.error(f"  {remaining_nan} NaN values remain after fill — "
                      f"filling with 0 as last resort")
            df_corrected = df_corrected.fillna(0.0)

    log.info(f"  pycombat done.  Shape: {df_corrected.shape}  "
             f"NaN remaining: {int(df_corrected.isna().sum().sum())}")

    # Load original full parquet (features × 755) and patch corrected columns
    log.info(f"  Patching original parquet at {parquet_path}...")
    df_full = pd.read_parquet(parquet_path)
    for col in patient_ids:
        if col in df_full.columns:
            df_full[col] = df_corrected[col].values
    df_full.to_parquet(parquet_path)
    log.info(f"  Parquet overwritten with pycombat-corrected values.")

    return df_corrected.values.T.astype(np.float32)  # patients × features


def delete_downstream_checkpoints():
    """Remove all affinity/embedding checkpoints so Step 05 recomputes."""
    log.info("  Deleting downstream affinity/embedding checkpoints...")
    deleted = []
    for f in DOWNSTREAM_FILES:
        if f.exists():
            f.unlink()
            deleted.append(f.name)
    if deleted:
        log.info(f"  Deleted: {', '.join(deleted)}")
    else:
        log.info("  No downstream checkpoints found (already clean).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
MODALITIES = [
    ("RNA",         PROCESSED / "rna_preprocessed.parquet"),
    ("CNV",         PROCESSED / "cnv_preprocessed.parquet"),
    ("Methylation", PROCESSED / "methylation_preprocessed.parquet"),
]

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║     STEP 04b: BATCH EFFECT CHECK (TSS)              ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"Batch R² warning threshold:    {BATCH_R2_WARN}")
    log.info(f"Batch R² correction threshold: {BATCH_R2_CORRECT}")
    log.info(f"PCs used for R² calculation:   PC1–{N_PCA_COMPONENTS}")

    # ── Load cohort ──────────────────────────────────────────────────────────
    cohort_path = PROCESSED / "final_cohort.txt"
    if not cohort_path.exists():
        log.error("final_cohort.txt not found — run 04_preprocess.py first")
        sys.exit(1)
    with open(cohort_path) as f:
        cohort = [l.strip() for l in f if l.strip()]
    log.info(f"Final cohort: {len(cohort)} patients")

    # Extract TSS batch labels
    tss = extract_tss(cohort)
    log.info(f"Unique TSS sites: {tss.nunique()}")
    top5 = tss.value_counts().head(5)
    log.info(f"Top 5 TSS sites (patient count):\n{top5.to_string()}")

    # ── Per-modality check ───────────────────────────────────────────────────
    any_corrected = False
    results_summary = []

    for label, parquet_path in MODALITIES:
        log.info("")
        log.info(f"{'='*55}")
        log.info(f"BATCH CHECK: {label}")
        log.info(f"{'='*55}")

        # Load and subset to cohort
        df = pd.read_parquet(parquet_path)
        present = [p for p in cohort if p in df.columns]
        missing = len(cohort) - len(present)
        if missing:
            log.warning(f"  {missing} cohort patients missing from {label} matrix")
        arr = df[present].values.T.astype(np.float32)   # patients × features
        log.info(f"  Loaded: {arr.shape[0]} patients × {arr.shape[1]} features")
        del df; gc.collect()

        # PCA for batch R²
        log.info(f"  Running PCA (n_components={N_PCA_COMPONENTS})...")
        pca_full = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
        pca_scores = pca_full.fit_transform(arr)       # patients × N_PCA
        var_explained = pca_full.explained_variance_ratio_
        log.info(f"  Variance explained: "
                 + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var_explained)))

        # Batch R² (η²)
        tss_arr = tss.loc[present].values
        r2 = batch_r2(pca_scores, tss_arr)
        log.info(f"  Batch R² (η² across PC1–{N_PCA_COMPONENTS}): {r2:.4f}")

        # Interpret
        if r2 < BATCH_R2_WARN:
            log.info(f"  → No meaningful batch structure (R²={r2:.3f} < {BATCH_R2_WARN}). OK.")
            status = "OK"
        elif r2 < BATCH_R2_CORRECT:
            log.warning(f"  → Weak batch signal (R²={r2:.3f}). Logging warning; no correction.")
            status = "WARN"
        else:
            log.warning(f"  → Batch structure detected (R²={r2:.3f} ≥ {BATCH_R2_CORRECT}). "
                        f"Applying pycombat.")
            arr = apply_combat(arr, present, tss, label, parquet_path)
            status = "CORRECTED"
            any_corrected = True

        # PCA figure (always generate — needed for paper)
        log.info(f"  Generating PC1 vs PC2 figure...")
        pca_and_plot(arr, present, tss, label, r2)

        results_summary.append((label, r2, status))
        del arr; gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║              BATCH CHECK SUMMARY                    ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    for label, r2, status in results_summary:
        flag = "✓" if status == "OK" else ("⚠" if status == "WARN" else "✗ CORRECTED")
        log.info(f"║  {label:12s}  R²={r2:.3f}   {flag}")
    log.info("╠══════════════════════════════════════════════════════╣")

    if any_corrected:
        log.info("║  pycombat was applied to ≥1 modality.               ║")
        log.info("║  Deleting downstream affinity checkpoints...         ║")
        log.info("╚══════════════════════════════════════════════════════╝")
        delete_downstream_checkpoints()
        log.info("  Re-run: python scripts/05_graph_construction.py")
    else:
        log.info("║  No pycombat correction applied.                     ║")
        log.info("║  If any downstream .npy files exist from a prior     ║")
        log.info("║  (broken) run, delete them before step 05.           ║")
        log.info("╚══════════════════════════════════════════════════════╝")
        log.info("  Proceed: python scripts/05_graph_construction.py")

    log.info("")
    log.info(f"  Figures saved to: {FIG_OUT}/")
