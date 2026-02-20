"""
05_graph_construction.py
------------------------
Build per-modality affinity graphs, fuse them via SNF, and produce
spectral embeddings for downstream clustering and survival modelling.

Pipeline
────────
  For each modality (RNA, CNV, methylation):
    1. Load preprocessed matrix (5,000 features × 755 patients)
    2. Subset to final_cohort (644 patients with valid OS data)
    3. Winsorise at ±5 σ  (prevents CNV focal-amp extremes from dominating cosine)
    4. Transpose → patients × features  (required by snf.make_affinity)

  Graph construction & fusion (locked hyperparameters):
    5. affinity_robust(*3 data matrices, K=20, mu=0.5)
       → 3 patient-similarity affinity matrices  (644 × 644)
       Uses Wang et al. 2014 bounded kernel: exp(−d / 2σ)
       NOT snf.make_affinity() — see compute_affinity() docstring.
    6. snf.snf(*3 affinity matrices, K=20, t=20)
       → 1 fused affinity matrix  (644 × 644)

  Spectral embedding:
    7. Eigendecomposition of normalised graph Laplacian of fused matrix
       → retain top 50 eigenvectors  (patients × 50)

  k selection support:
    8. Compute eigengap on fused affinity spectrum  (2 ≤ k ≤ 10)
       → log suggested k; save eigenvalue array for paper figure

SNF hyperparameters (locked — NOT swept, see README):
  K  = 20   (neighbours)
  t  = 20   (diffusion iterations)
  mu = 0.5  (scaling — passed to make_affinity)

Outputs  (data/processed/)
──────────────────────────
  affinity_rna.npy          644 × 644 float64
  affinity_cnv.npy          644 × 644 float64
  affinity_meth.npy         644 × 644 float64
  affinity_fused.npy        644 × 644 float64
  spectral_embedding.parquet   644 × 50  float32   patients × dims
  eigenvalues_fused.npy     eigenvalue array for eigengap figure

Run from project root:
  python scripts/05_graph_construction.py
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import snf
from scipy.spatial.distance import cdist
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────────────────────────────────────
# Config — locked SNF hyperparameters (Wang et al. 2014 defaults)
# ──────────────────────────────────────────────────────────────────────────────
K_NEIGHBOURS  = 20      # k-NN graph degree
T_ITERATIONS  = 20      # diffusion steps
MU            = 0.5     # affinity scaling (passed to make_affinity)
N_SPECTRAL    = 50      # eigenvectors to retain for embedding
WINSOR_SIGMA  = 5.0     # clip z-scores beyond ±5σ before similarity
K_RANGE       = range(2, 11)   # k values to evaluate for eigengap

PROCESSED = Path("data/processed")
LOG_DIR   = Path("logs")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_graph_construction.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def checkpoint_exists(path: Path) -> bool:
    if path.exists():
        log.info(f"CHECKPOINT HIT — skipping: {path}")
        return True
    return False


def report_ram():
    try:
        with open("/proc/meminfo") as f:
            avail = [l for l in f if l.startswith("MemAvailable")][0].split()[1]
        log.info(f"  [RAM] MemAvailable = {int(avail)//1024} MB")
    except Exception:
        pass


def winsorise(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Clip values beyond ±sigma standard deviations (computed column-wise).
    arr shape: patients × features. Because matrices are already z-scored
    row-wise (per-feature), simply clip on the global ±sigma bound."""
    clipped = np.clip(arr, -sigma, sigma)
    n_clipped = int(np.sum(arr != clipped))
    if n_clipped:
        log.info(f"  Winsorised {n_clipped:,} values beyond ±{sigma}σ "
                 f"({n_clipped / arr.size * 100:.3f}% of all values)")
    return clipped


def load_modality(parquet_path: Path, cohort: list,
                  label: str) -> np.ndarray:
    """
    Load a preprocessed (features × patients) parquet, subset to cohort,
    transpose to (patients × features), winsorise, L2-normalise each patient
    vector, then return float64 ndarray.

    L2 normalisation rationale:
      Per-gene z-scoring gives each gene mean=0 / std=1 across patients, but
      patient L2 norms vary greatly (especially CNV — diploid patients have
      near-zero vectors while amplified samples have large norms). Large norm
      variation causes snf.make_affinity's KNN-bandwidth estimate to collapse
      to ~0 for near-identical patients, overflowing the RBF kernel.
      Projecting each patient to the unit hypersphere makes sqeuclidean
      equivalent to 2*(1 − cosine_similarity), bounding all distances in
      [0, 2] and eliminating bandwidth collapse. This is methodologically
      identical to using cosine similarity, which is pre-specified in
      the project README as an acceptable metric.
    """
    log.info(f"Loading {label} — {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log.info(f"  Shape (features × all patients): {df.shape}")

    # Subset columns (patients) to final cohort, preserving order
    present = [p for p in cohort if p in df.columns]
    missing = len(cohort) - len(present)
    if missing:
        log.warning(f"  {missing} cohort patients not found in {label} matrix")
    df = df[present]

    arr = df.values.T.astype(np.float64)   # → patients × features, float64
    log.info(f"  Shape (patients × features): {arr.shape}")

    arr = winsorise(arr, WINSOR_SIGMA)

    # L2-normalise: project each patient onto unit hypersphere
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms_min, norms_max = float(norms.min()), float(norms.max())
    log.info(f"  Patient L2 norms before normalisation: "
             f"min={norms_min:.2f}  max={norms_max:.2f}  "
             f"mean={float(norms.mean()):.2f}")
    # Guard against any zero-norm patient (shouldn't exist, but be safe)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms
    log.info(f"  After L2 normalisation: "
             f"norm range=[{np.linalg.norm(arr,axis=1).min():.6f}, "
             f"{np.linalg.norm(arr,axis=1).max():.6f}]  (should be ~1.0)")

    report_ram()
    del df
    gc.collect()
    return arr, present


def compute_affinity(data_arr: np.ndarray, label: str) -> np.ndarray:
    """
    Compute a patient×patient affinity matrix using the Wang et al. 2014
    bounded exponential kernel — NOT snf.make_affinity().

    WHY NOT snf.make_affinity():
      snfpy 0.2.2 implements the kernel as:
          W(i,j) = scipy.stats.norm.pdf(d_ij, loc=0, scale=μ·σ_ij)
      stats.norm.pdf is an UNBOUNDED probability density:
          pdf = 1/(σ·√2π) · exp(−d²/2σ²)
      When σ→0 (patients with zero KNN distance, e.g. near-duplicate CNV
      profiles on the unit hypersphere), pdf → ∞. This produced a CNV
      affinity matrix max of 3.59×10¹⁵ with 69 patients having TT=0.

    CORRECT Wang et al. 2014 kernel (bounded in (0, 1]):
          W(i,j) = exp(−d_ij / (2·σ_ij))
      where d_ij   = squared-Euclidean distance between patients i and j
            TT_i   = mean squared-Euclidean distance to K nearest neighbours
            σ_ij   = μ · (TT_i + TT_j + d_ij) / 3
      σ is clipped to np.finfo(float64).tiny to prevent zero-division.
      W is bounded in (0, 1] for all finite inputs — safe for all modalities.

    snf.snf() (the diffusion step) is still used — only make_affinity replaced.

    Returns float64 ndarray, shape (n_patients, n_patients).
    """
    log.info(f"Computing affinity matrix: {label}  "
             f"(K={K_NEIGHBOURS}, mu={MU}, robust bounded kernel)")

    n = data_arr.shape[0]

    # Pairwise squared-Euclidean distances — O(n²·p), float64
    D = cdist(data_arr, data_arr, metric="sqeuclidean")

    # KNN mean distance per patient (exclude self: column 0 is always 0)
    D_sorted = np.sort(D, axis=1)                        # each row sorted asc
    TT = D_sorted[:, 1:K_NEIGHBOURS + 1].mean(axis=1)   # mean of K nearest

    # Bandwidth matrix: σ_ij = μ · (TT_i + TT_j + d_ij) / 3
    sigma = MU * (TT[:, None] + TT[None, :] + D) / 3.0

    # Clip to prevent zero-division (σ=0 for near-duplicate patients)
    eps = np.finfo(np.float64).tiny
    sigma = np.maximum(sigma, eps)

    # Bounded kernel: exp(−d / 2σ)  —  values in (0, 1]
    W = np.exp(-D / (2.0 * sigma))

    # Zero diagonal (self-similarity excluded, matching snfpy convention)
    np.fill_diagonal(W, 0.0)

    # Enforce symmetry (should be exact, but numerical safety)
    W = (W + W.T) / 2.0

    # Diagnostics
    n_zero_tt = int(np.sum(TT < 1e-6))
    if n_zero_tt:
        log.warning(f"  {n_zero_tt} patients had TT < 1e-6 (near-duplicate "
                    f"on unit sphere) — σ clamped to {eps:.2e}")
    log.info(f"  TT range: {TT.min():.3e} – {TT.max():.3e}  "
             f"(mean {TT.mean():.4f})")
    log.info(f"  Affinity matrix shape: {W.shape}  "
             f"dtype: {W.dtype}  "
             f"range: [{W.min():.6f}, {W.max():.6f}]")
    sym_err = float(np.abs(W - W.T).max())
    if sym_err > 1e-10:
        log.warning(f"  Symmetry error unexpectedly large: {sym_err:.2e}")

    return W


def spectral_embed(fused_aff: np.ndarray,
                   patient_ids: list,
                   n_components: int) -> pd.DataFrame:
    """
    Compute spectral embedding of the fused affinity matrix.
    Uses sklearn SpectralEmbedding with precomputed affinity.
    Returns DataFrame (patients × n_components).
    """
    log.info(f"Computing spectral embedding (n_components={n_components})...")

    se = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=42,
        n_jobs=-1,
    )
    embedding = se.fit_transform(fused_aff)   # patients × n_components
    log.info(f"  Embedding shape: {embedding.shape}")

    cols = [f"SE_{i+1}" for i in range(n_components)]
    df   = pd.DataFrame(embedding.astype(np.float32),
                        index=patient_ids, columns=cols)
    df.index.name = "patient_id"
    return df


def eigengap_analysis(fused_aff: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Eigengap heuristic on the normalised graph Laplacian.
    Evaluates k in K_RANGE, returns (suggested_k, eigenvalues array).
    Eigenvalues are sorted ascending; the eigengap is the largest jump
    between consecutive eigenvalues — the index+1 of that jump is k.
    """
    log.info(f"Eigengap analysis (k range: {K_RANGE.start}–{K_RANGE.stop - 1})...")

    # snf.get_n_clusters uses eigengap on the affinity matrix directly
    # Returns sorted eigenvalues; we use the standard eigengap method on
    # the Laplacian for transparency.

    # Degree matrix
    D = np.diag(fused_aff.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D).clip(1e-10)))

    # Normalised Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    L_sym = np.eye(len(fused_aff)) - D_inv_sqrt @ fused_aff @ D_inv_sqrt

    # Eigenvalues of L_sym (ascending — smallest = most connected)
    eigenvalues = np.linalg.eigvalsh(L_sym)
    eigenvalues = np.sort(eigenvalues)

    # Evaluate eigengap for each candidate k in range
    # Gap_k = λ_{k+1} - λ_k  (larger gap → natural break)
    max_k = max(K_RANGE)
    gaps  = np.diff(eigenvalues[1:max_k + 2])   # gaps between consecutive evals
    gap_k_index = int(np.argmax(gaps))           # 0-based index into K_RANGE start
    suggested_k = K_RANGE.start + gap_k_index

    log.info(f"  Eigenvalues[1:{max_k+2}]: "
             + "  ".join(f"λ{i+2}={eigenvalues[i+1]:.4f}" for i in range(max_k)))
    log.info(f"  Eigengaps:   "
             + "  ".join(f"{g:.4f}" for g in gaps))
    log.info(f"  Suggested k (largest eigengap): {suggested_k}")

    return suggested_k, eigenvalues


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

AFF_RNA    = PROCESSED / "affinity_rna.npy"
AFF_CNV    = PROCESSED / "affinity_cnv.npy"
AFF_METH   = PROCESSED / "affinity_meth.npy"
AFF_FUSED  = PROCESSED / "affinity_fused.npy"
EMBED_OUT  = PROCESSED / "spectral_embedding.parquet"
EVALS_OUT  = PROCESSED / "eigenvalues_fused.npy"

if __name__ == "__main__":
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║     STEP 5: GRAPH CONSTRUCTION + SNF + EMBEDDING     ║")
    log.info("╚═══════════════════════════════════════════════════════╝")
    log.info(f"SNF hyperparameters: K={K_NEIGHBOURS}, t={T_ITERATIONS}, mu={MU}")
    log.info(f"Spectral dimensions: {N_SPECTRAL}")
    log.info(f"Winsorisation:       ±{WINSOR_SIGMA}σ")

    # ── Load final cohort ────────────────────────────────────────────────────
    cohort_path = PROCESSED / "final_cohort.txt"
    if not cohort_path.exists():
        log.error("final_cohort.txt not found — run 04_preprocess.py first")
        sys.exit(1)
    with open(cohort_path) as f:
        cohort = [l.strip() for l in f if l.strip()]
    log.info(f"Final cohort: {len(cohort)} patients")

    # ── Per-modality: load, winsorise, compute affinity ──────────────────────
    all_done = (checkpoint_exists(AFF_RNA) and
                checkpoint_exists(AFF_CNV) and
                checkpoint_exists(AFF_METH))

    if not all_done:
        rna_data,  rna_patients  = load_modality(
            PROCESSED / "rna_preprocessed.parquet",  cohort, "RNA")
        cnv_data,  cnv_patients  = load_modality(
            PROCESSED / "cnv_preprocessed.parquet",  cohort, "CNV")
        meth_data, meth_patients = load_modality(
            PROCESSED / "methylation_preprocessed.parquet", cohort, "Methylation")

        # Patient IDs must be identical across modalities going into SNF
        assert rna_patients == cnv_patients == meth_patients, \
            "Patient order mismatch across modalities — cannot proceed"
        patient_ids = rna_patients
        log.info(f"Patient alignment confirmed: {len(patient_ids)} patients "
                 f"across all modalities")

        # Affinity matrices
        log.info("=" * 60)
        log.info("BUILDING AFFINITY MATRICES")
        log.info("=" * 60)
        aff_rna  = compute_affinity(rna_data,  "RNA")
        np.save(AFF_RNA,  aff_rna)
        log.info(f"Saved {AFF_RNA}")
        del rna_data; gc.collect()

        aff_cnv  = compute_affinity(cnv_data,  "CNV")
        np.save(AFF_CNV,  aff_cnv)
        log.info(f"Saved {AFF_CNV}")
        del cnv_data; gc.collect()

        aff_meth = compute_affinity(meth_data, "Methylation")
        np.save(AFF_METH, aff_meth)
        log.info(f"Saved {AFF_METH}")
        del meth_data; gc.collect()

    else:
        log.info("Loading cached affinity matrices...")
        aff_rna   = np.load(AFF_RNA)
        aff_cnv   = np.load(AFF_CNV)
        aff_meth  = np.load(AFF_METH)

        # Recover patient list from clinical parquet (same order as final_cohort)
        clin = pd.read_parquet(PROCESSED / "clinical_preprocessed.parquet")
        patient_ids = [p for p in cohort if p in clin.index]
        log.info(f"Patient list recovered: {len(patient_ids)} patients")

    # ── SNF ──────────────────────────────────────────────────────────────────
    if not checkpoint_exists(AFF_FUSED):
        log.info("=" * 60)
        log.info("RUNNING SNF")
        log.info("=" * 60)
        log.info(f"Fusing {len([aff_rna, aff_cnv, aff_meth])} modalities  "
                 f"(K={K_NEIGHBOURS}, t={T_ITERATIONS})")
        report_ram()

        aff_fused = snf.snf(aff_rna, aff_cnv, aff_meth,
                            K=K_NEIGHBOURS, t=T_ITERATIONS)

        log.info(f"Fused matrix shape: {aff_fused.shape}  "
                 f"range: [{aff_fused.min():.4f}, {aff_fused.max():.4f}]")
        sym_err = float(np.abs(aff_fused - aff_fused.T).max())
        log.info(f"Fused matrix symmetry error: {sym_err:.2e}  "
                 f"({'OK' if sym_err < 1e-6 else 'WARNING: asymmetric'})")

        np.save(AFF_FUSED, aff_fused)
        log.info(f"Saved {AFF_FUSED}")
        report_ram()
    else:
        log.info("Loading cached fused affinity matrix...")
        aff_fused = np.load(AFF_FUSED)

    # ── Eigengap analysis ────────────────────────────────────────────────────
    if not checkpoint_exists(EVALS_OUT):
        suggested_k, eigenvalues = eigengap_analysis(aff_fused)
        np.save(EVALS_OUT, eigenvalues)
        log.info(f"Eigenvalues saved → {EVALS_OUT}")
        log.info(f"PRE-SPECIFIED k = {suggested_k}  "
                 f"(document this before examining clustering results)")
    else:
        log.info("Loading cached eigenvalues...")
        eigenvalues = np.load(EVALS_OUT)
        # Recompute suggested k from saved eigenvalues
        max_k = max(K_RANGE)
        gaps  = np.diff(eigenvalues[1:max_k + 2])
        suggested_k = K_RANGE.start + int(np.argmax(gaps))
        log.info(f"Suggested k from eigengap: {suggested_k}")

    # ── Spectral embedding ───────────────────────────────────────────────────
    if not checkpoint_exists(EMBED_OUT):
        embedding_df = spectral_embed(aff_fused, patient_ids, N_SPECTRAL)
        embedding_df.to_parquet(EMBED_OUT)
        log.info(f"Spectral embedding saved → {EMBED_OUT}")
    else:
        log.info(f"Spectral embedding checkpoint exists: {EMBED_OUT}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║         GRAPH CONSTRUCTION COMPLETE                  ║")
    log.info("╠═══════════════════════════════════════════════════════╣")
    log.info(f"║  Patients:               {len(patient_ids)}")
    log.info(f"║  Affinity matrices:      3 × {aff_rna.shape}  float64")
    log.info(f"║  Fused matrix:           {aff_fused.shape}  float64")
    log.info(f"║  Spectral embedding:     {len(patient_ids)} × {N_SPECTRAL}")
    log.info(f"║  Suggested k (eigengap): {suggested_k}")
    log.info("╠═══════════════════════════════════════════════════════╣")
    log.info("║  IMPORTANT: k is now locked.                         ║")
    log.info("║  Do not change it after examining cluster results.   ║")
    log.info("╚═══════════════════════════════════════════════════════╝")
    log.info("\n✓ Next: python scripts/06_clustering.py")
