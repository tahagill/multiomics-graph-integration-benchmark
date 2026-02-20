"""
04_preprocess.py
----------------
Preprocess all raw matrices into analysis-ready, normalised matrices.

Modality pipeline
─────────────────
  RNA-seq
    1. Subset to final 755-patient cohort
    2. Cast to float32
    3. CPM normalisation (per-sample library-size correction)
    4. log2(CPM + 1) transformation
    5. Remove low-count genes  (CPM < 1 in > 80 % of samples)
    6. Variance filter          → top N_FEATURES genes by MAD
    7. Z-score per gene across patients

  CNV
    1. Subset to final cohort
    2. Variance filter          → top N_FEATURES genes by MAD
    3. Z-score per gene across patients

  Methylation
    1. Subset to final cohort
    2. Download / load Illumina 450K probe annotation (Zhou lab)
    3. Remove probes on sex chromosomes (chrX, chrY)
    4. Remove SNP-proximal probes  (MASK_snp5_common or MASK_general)
    5. Remove rs* control probes (SNP probes, not CpG)
    6. Remove probes with > 20 % missingness
    7. Variance filter          → top N_FEATURES CpGs by MAD
    8. Z-score per probe across patients

  Clinical
    1. Subset to final cohort
    2. Build os_time  (days_to_death if Dead, else days_to_last_followup)
    3. Build os_event (1 = Dead, 0 = Alive/censored)
    4. Exclude patients with os_time < 30 days or os_time NaN
    5. Numeric tumour stage (I→1, II→2, III→3, IV→4)
    6. Save final patient list → data/processed/final_cohort.txt

Outputs  (data/processed/)
──────────────────────────
  rna_preprocessed.parquet
  cnv_preprocessed.parquet
  methylation_preprocessed.parquet
  clinical_preprocessed.parquet
  final_cohort.txt            ← cohort after clinical filter (used by all downstream scripts)

Memory strategy (16 GB RAM / HDD)
──────────────────────────────────
  - Process one modality at a time; explicit del + gc.collect() between steps
  - float32 throughout (not float64)
  - Methylation: load → subset → filter rows aggressively → MAD → z-score → save
  - Checkpointing: skip if output already exists

Run from project root:
  python scripts/04_preprocess.py
"""

import gc
import gzip
import io
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED    = Path("data/processed")
METADATA_DIR = Path("metadata")
LOG_DIR      = Path("logs")

N_FEATURES   = 5_000        # features retained per modality after MAD filter
LOW_CPM_FMT  = 0.80         # fraction of samples below CPM threshold to call "low"
CPM_THRESH   = 1.0          # CPM threshold for low-count filter
MISS_THRESH  = 0.20         # max fraction of missing values allowed per CpG probe
OS_MIN_DAYS  = 30           # minimum OS time; patients below this are excluded

# Illumina 450K annotation (Zhou lab, hg38) — genomic coordinates + chromosome
ANNOT_URL  = (
    "https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1"
    "/main/Anno/HM450/HM450.hg38.manifest.tsv.gz"
)
ANNOT_CACHE = METADATA_DIR / "HM450.hg38.manifest.tsv.gz"

# Illumina 450K SNP-proximal probe list (Zhou lab, hg38)
# Every probe in this file overlaps a known genomic variant → exclude from analysis
SNP_URL   = (
    "https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1"
    "/main/Anno/HM450/HM450.hg38.snp.tsv.gz"
)
SNP_CACHE = METADATA_DIR / "HM450.hg38.snp.tsv.gz"

# Tumour stage → ordinal mapping
STAGE_MAP = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4,
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_preprocess.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def checkpoint_exists(path: Path) -> bool:
    if path.exists():
        log.info(f"CHECKPOINT HIT — skipping, already exists: {path}")
        return True
    return False


def load_final_cohort() -> list:
    """Load the 755-patient intersection list produced by 03_build_matrices.py."""
    path = PROCESSED / "sample_intersection.txt"
    if not path.exists():
        log.error("sample_intersection.txt not found — run 03_build_matrices.py first")
        sys.exit(1)
    with open(path) as f:
        cohort = [line.strip() for line in f if line.strip()]
    log.info(f"Loaded {len(cohort)} patients from sample_intersection.txt")
    return cohort


def mad_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Keep the top-n rows (features) by MAD across columns (samples).
    df must be features × patients.
    Uses float32 values; returns float32 DataFrame.
    """
    log.info(f"  MAD filter: {df.shape[0]:,} features → top {n:,}")
    mad = median_abs_deviation(df.values, axis=1, nan_policy="omit")
    idx = np.argpartition(mad, -n)[-n:]        # indices of top-n (unordered)
    idx = idx[np.argsort(mad[idx])[::-1]]      # sort descending
    result = df.iloc[idx]
    log.info(f"  After MAD filter: {result.shape}")
    return result


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise across patients (axis=1) for each feature (row).
    Features with zero variance are set to 0 (not NaN).
    Returns float32 DataFrame with same row/column labels.
    """
    vals    = df.values.astype(np.float32)
    means   = vals.mean(axis=1, keepdims=True)
    stds    = vals.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0           # avoid divide-by-zero; result will be 0
    zscore  = (vals - means) / stds
    return pd.DataFrame(zscore.astype(np.float32),
                        index=df.index, columns=df.columns)


def report_ram(label: str):
    """Log a rough estimate of currently used RAM via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            lines = {l.split(":")[0]: l.split(":")[1].strip()
                     for l in f.readlines()}
        avail = lines.get("MemAvailable", "?")
        log.info(f"  [RAM] {label}: MemAvailable = {avail}")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 1. Methylation annotation
# ──────────────────────────────────────────────────────────────────────────────

def fetch_methylation_annotation() -> pd.DataFrame | None:
    """
    Download and cache the Zhou lab Illumina HM450 hg38 manifest.
    Returns a DataFrame with index=Probe_ID and at minimum columns:
        CpG_chrm         (chromosome string, e.g. 'chrX')
        MASK_snp5_common (bool-like: True = SNP-proximal)
    Returns None if download fails (caller uses fallback filtering only).
    """
    if not ANNOT_CACHE.exists():
        log.info(f"Downloading 450K probe annotation from Zhou lab...")
        log.info(f"  URL: {ANNOT_URL}")
        log.info(f"  Destination: {ANNOT_CACHE}")
        try:
            urllib.request.urlretrieve(ANNOT_URL, ANNOT_CACHE)
            log.info("  Download complete.")
        except Exception as e:
            log.warning(f"  Annotation download failed: {e}")
            log.warning("  Sex-chromosome and SNP-proximal filtering will be SKIPPED.")
            log.warning("  Only rs* control probes will be removed.")
            return None
    else:
        log.info(f"Using cached annotation: {ANNOT_CACHE}")

    log.info("Loading annotation TSV...")
    try:
        with gzip.open(ANNOT_CACHE, "rb") as gz:
            raw_bytes = gz.read()
        annot = pd.read_csv(
            io.StringIO(raw_bytes.decode("utf-8")),
            sep="\t",
            usecols=lambda c: c in {"Probe_ID", "CpG_chrm",
                                    "MASK_snp5_common", "MASK_general"},
            dtype=str,
            low_memory=False,
        )
        log.info(f"Annotation loaded: {annot.shape[0]:,} probes, "
                 f"columns: {annot.columns.tolist()}")

        # Normalise probe ID column name
        if "Probe_ID" not in annot.columns:
            # Try first column
            annot = annot.rename(columns={annot.columns[0]: "Probe_ID"})

        annot = annot.set_index("Probe_ID")
        return annot

    except Exception as e:
        log.warning(f"Failed to parse annotation file: {e}")
        log.warning("Sex-chromosome and SNP-proximal filtering will be SKIPPED.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1b. SNP-proximal probe list
# ──────────────────────────────────────────────────────────────────────────────

def fetch_snp_probes() -> set | None:
    """
    Download (once) and return the set of 450K probe IDs that overlap a known
    genomic variant according to the Zhou lab hg38 SNP annotation.
    Returns None if download fails (caller skips SNP filtering).
    """
    if not SNP_CACHE.exists():
        log.info("Downloading 450K SNP-proximal probe list from Zhou lab...")
        log.info(f"  URL: {SNP_URL}")
        log.info(f"  Destination: {SNP_CACHE}")
        try:
            urllib.request.urlretrieve(SNP_URL, SNP_CACHE)
            log.info("  Download complete.")
        except Exception as e:
            log.warning(f"  SNP probe download failed: {e}")
            log.warning("  SNP-proximal filtering will be SKIPPED.")
            return None
    else:
        log.info(f"Using cached SNP probe list: {SNP_CACHE}")

    try:
        snp_df = pd.read_csv(SNP_CACHE, sep="\t",
                             usecols=["Probe_ID"], dtype=str)
        snp_probes = set(snp_df["Probe_ID"].dropna().unique())
        log.info(f"SNP-proximal probes to exclude: {len(snp_probes):,}")
        return snp_probes
    except Exception as e:
        log.warning(f"Failed to parse SNP probe file: {e}")
        log.warning("SNP-proximal filtering will be SKIPPED.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. RNA-seq preprocessing
# ──────────────────────────────────────────────────────────────────────────────
RNA_OUT = PROCESSED / "rna_preprocessed.parquet"

def preprocess_rna(cohort: list):
    log.info("=" * 60)
    log.info("PREPROCESSING RNA-SEQ")
    log.info("=" * 60)

    raw_path = PROCESSED / "rna_matrix_raw.parquet"
    log.info(f"Loading {raw_path}...")
    rna = pd.read_parquet(raw_path)
    log.info(f"Loaded: {rna.shape}")
    report_ram("after RNA load")

    # ── Subset to cohort ────────────────────────────────────────────────────
    cohort_present = [p for p in cohort if p in rna.columns]
    log.info(f"Cohort patients found in RNA matrix: "
             f"{len(cohort_present)}/{len(cohort)}")
    rna = rna[cohort_present]

    # ── Cast to float32 ─────────────────────────────────────────────────────
    log.info("Casting to float32...")
    rna = rna.astype(np.float32)
    report_ram("after float32 cast")

    # ── CPM normalisation ───────────────────────────────────────────────────
    log.info("CPM normalisation (library-size correction)...")
    lib_sizes = rna.sum(axis=0)
    log.info(f"  Median library size: {lib_sizes.median():,.0f} counts")
    rna = rna.div(lib_sizes, axis=1) * 1e6           # CPM
    log.info("  CPM done.")

    # ── Low-count gene filter ───────────────────────────────────────────────
    n_samples = rna.shape[1]
    frac_below = (rna < CPM_THRESH).sum(axis=1) / n_samples
    mask_keep  = frac_below <= LOW_CPM_FMT
    n_removed  = (~mask_keep).sum()
    log.info(f"Low-count filter (CPM < {CPM_THRESH} in > "
             f"{LOW_CPM_FMT*100:.0f}% samples): removing {n_removed:,} genes")
    rna = rna.loc[mask_keep]
    log.info(f"  Remaining genes: {rna.shape[0]:,}")

    # ── log2(CPM + 1) ───────────────────────────────────────────────────────
    log.info("log2(CPM + 1) transformation...")
    rna = np.log2(rna + 1.0).astype(np.float32)

    # ── MAD variance filter ─────────────────────────────────────────────────
    rna = mad_top_n(rna, N_FEATURES)

    # ── Z-score per gene ────────────────────────────────────────────────────
    log.info("Z-score normalisation (per gene, across patients)...")
    rna = zscore_rows(rna)

    # ── Save ────────────────────────────────────────────────────────────────
    log.info(f"Saving {RNA_OUT} ...  shape: {rna.shape}")
    rna.to_parquet(RNA_OUT)
    log.info("RNA-seq preprocessing done.")

    # Min/max sanity check
    log.info(f"  Value range: [{rna.values.min():.3f}, {rna.values.max():.3f}]")
    log.info(f"  Mean abs value: {np.abs(rna.values).mean():.4f} (should be ~0.8)")

    del rna
    gc.collect()
    report_ram("after RNA del")


# ──────────────────────────────────────────────────────────────────────────────
# 3. CNV preprocessing
# ──────────────────────────────────────────────────────────────────────────────
CNV_OUT = PROCESSED / "cnv_preprocessed.parquet"

def preprocess_cnv(cohort: list):
    log.info("=" * 60)
    log.info("PREPROCESSING CNV")
    log.info("=" * 60)

    raw_path = PROCESSED / "cnv_matrix_raw.parquet"
    log.info(f"Loading {raw_path}...")
    cnv = pd.read_parquet(raw_path)
    log.info(f"Loaded: {cnv.shape}")

    # ── Subset to cohort ────────────────────────────────────────────────────
    cohort_present = [p for p in cohort if p in cnv.columns]
    log.info(f"Cohort patients found in CNV matrix: "
             f"{len(cohort_present)}/{len(cohort)}")
    cnv = cnv[cohort_present].astype(np.float32)

    # ── MAD variance filter ─────────────────────────────────────────────────
    # Drop rows that are entirely constant first (no MAD signal)
    n_constant = (cnv.std(axis=1) == 0).sum()
    log.info(f"Constant-value genes (all patients same CN): {n_constant:,} — removing")
    cnv = cnv.loc[cnv.std(axis=1) > 0]
    log.info(f"  After constant removal: {cnv.shape[0]:,} genes")

    cnv = mad_top_n(cnv, N_FEATURES)

    # ── Z-score per gene ────────────────────────────────────────────────────
    log.info("Z-score normalisation (per gene, across patients)...")
    cnv = zscore_rows(cnv)

    # ── Save ────────────────────────────────────────────────────────────────
    log.info(f"Saving {CNV_OUT} ...  shape: {cnv.shape}")
    cnv.to_parquet(CNV_OUT)
    log.info(f"CNV preprocessing done.  "
             f"Value range: [{cnv.values.min():.3f}, {cnv.values.max():.3f}]")

    del cnv
    gc.collect()
    report_ram("after CNV del")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Methylation preprocessing
# ──────────────────────────────────────────────────────────────────────────────
METH_OUT = PROCESSED / "methylation_preprocessed.parquet"

def preprocess_methylation(cohort: list):
    log.info("=" * 60)
    log.info("PREPROCESSING METHYLATION")
    log.info("=" * 60)

    # ── Load annotation (chromosome info) and SNP probe list ───────────────
    annot = fetch_methylation_annotation()
    snp_probes = fetch_snp_probes()
    report_ram("after annotation load")

    # ── Load raw methylation matrix ─────────────────────────────────────────
    raw_path = PROCESSED / "methylation_matrix_raw.parquet"
    n_probes_raw = pd.read_parquet(raw_path, columns=[]).shape[0]  # cheap shape check
    log.info(f"Raw methylation: {n_probes_raw:,} probes (will load full matrix)...")
    report_ram("before meth load")

    meth = pd.read_parquet(raw_path)
    log.info(f"Loaded: {meth.shape}    dtype: {meth.dtypes.iloc[0]}")
    report_ram("after meth load")

    # ── Subset to cohort ────────────────────────────────────────────────────
    cohort_present = [p for p in cohort if p in meth.columns]
    log.info(f"Cohort patients found in methylation matrix: "
             f"{len(cohort_present)}/{len(cohort)}")
    meth = meth[cohort_present]
    log.info(f"  After cohort subset: {meth.shape}")
    report_ram("after cohort subset")

    # ── Remove rs* control probes (SNP probes, not CpG sites) ───────────────
    rs_mask = meth.index.str.startswith("rs")
    n_rs    = rs_mask.sum()
    log.info(f"Removing rs* SNP control probes: {n_rs:,}")
    meth = meth.loc[~rs_mask]
    log.info(f"  Remaining: {meth.shape[0]:,} probes")

    # ── Sex chromosome probe removal (uses coordinate manifest) ────────────
    if annot is not None:
        if "CpG_chrm" in annot.columns:
            sex_probes = annot.loc[annot["CpG_chrm"].isin(["chrX", "chrY"])].index
            sex_probes_in_data = meth.index.intersection(sex_probes)
            log.info(f"Removing sex-chromosome probes (chrX/chrY): "
                     f"{len(sex_probes_in_data):,}")
            meth = meth.loc[~meth.index.isin(sex_probes_in_data)]
            log.info(f"  Remaining: {meth.shape[0]:,} probes")
        else:
            log.warning("CpG_chrm column not found in annotation — "
                        "sex-chromosome filtering skipped")
        del annot
        gc.collect()
        report_ram("after sex-chr filter + annotation del")

    else:
        log.warning("No annotation available — sex-chromosome filtering skipped")

    # ── SNP-proximal probe removal ────────────────────────────────────────
    if snp_probes is not None:
        snp_in_data = meth.index.intersection(snp_probes)
        log.info(f"Removing SNP-proximal probes: {len(snp_in_data):,}")
        meth = meth.loc[~meth.index.isin(snp_in_data)]
        log.info(f"  Remaining: {meth.shape[0]:,} probes")
        del snp_probes
        gc.collect()
    else:
        log.warning("SNP probe list unavailable — SNP-proximal filtering skipped")

    # ── Missingness filter: remove probes with > 20 % NaN ───────────────────
    n_samples  = meth.shape[1]
    miss_frac  = meth.isnull().sum(axis=1) / n_samples
    n_miss_rm  = (miss_frac > MISS_THRESH).sum()
    log.info(f"Removing probes with > {MISS_THRESH*100:.0f}% missing: {n_miss_rm:,}")
    meth = meth.loc[miss_frac <= MISS_THRESH]
    log.info(f"  Remaining: {meth.shape[0]:,} probes")

    # After missingness filter, impute any remaining NaN with probe median
    # (only a tiny fraction of values at this point)
    n_nan = meth.isnull().sum().sum()
    if n_nan > 0:
        log.info(f"Imputing {n_nan:,} remaining NaN values with probe median...")
        meth = meth.apply(lambda row: row.fillna(row.median()), axis=1)

    report_ram("after missingness filter")

    # ── MAD variance filter: top N_FEATURES CpGs ────────────────────────────
    meth = meth.astype(np.float32)
    meth = mad_top_n(meth, N_FEATURES)

    # ── Z-score per probe ───────────────────────────────────────────────────
    log.info("Z-score normalisation (per probe, across patients)...")
    meth = zscore_rows(meth)

    # ── Save ────────────────────────────────────────────────────────────────
    log.info(f"Saving {METH_OUT} ...  shape: {meth.shape}")
    meth.to_parquet(METH_OUT)
    log.info(f"Methylation preprocessing done.  "
             f"Value range: [{meth.values.min():.3f}, {meth.values.max():.3f}]")

    del meth
    gc.collect()
    report_ram("after methylation del")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Clinical preprocessing
# ──────────────────────────────────────────────────────────────────────────────
CLIN_OUT         = PROCESSED / "clinical_preprocessed.parquet"
FINAL_COHORT_OUT = PROCESSED / "final_cohort.txt"

def preprocess_clinical(cohort: list) -> list:
    """
    Returns the final patient list after OS-based exclusions.
    Saves clinical_preprocessed.parquet and final_cohort.txt.
    """
    log.info("=" * 60)
    log.info("PREPROCESSING CLINICAL")
    log.info("=" * 60)

    clin = pd.read_parquet(PROCESSED / "clinical_matrix.parquet")
    log.info(f"Loaded clinical matrix: {clin.shape}")

    # ── Subset to cohort ────────────────────────────────────────────────────
    cohort_present = [p for p in cohort if p in clin.index]
    log.info(f"Cohort patients found in clinical matrix: "
             f"{len(cohort_present)}/{len(cohort)}")
    clin = clin.loc[cohort_present].copy()

    # ── Build OS endpoint ───────────────────────────────────────────────────
    # os_event: 1 = Dead, 0 = Alive/censored
    clin["os_event"] = (clin["vital_status"] == "Dead").astype(np.int8)

    # os_time: days_to_death for Dead patients; days_to_last_followup for alive
    clin["os_time"] = np.where(
        clin["os_event"] == 1,
        clin["days_to_death"],
        clin["days_to_last_followup"],
    ).astype("float64")

    log.info(f"OS event distribution:\n{clin['os_event'].value_counts().to_string()}")
    log.info(f"OS time — median: {clin['os_time'].median():.0f} days, "
             f"  NaN: {clin['os_time'].isna().sum()}")

    # ── Exclusion: OS time < 30 days or NaN ────────────────────────────────
    n_before = len(clin)
    clin = clin.loc[clin["os_time"].notna() & (clin["os_time"] >= OS_MIN_DAYS)]
    n_excluded = n_before - len(clin)
    log.info(f"Excluded {n_excluded} patients (OS < {OS_MIN_DAYS} days or NaN OS time)")
    log.info(f"Remaining patients: {len(clin)}")

    # ── Tumour stage: ordinal numeric ───────────────────────────────────────
    clin["tumor_stage_numeric"] = clin["ajcc_pathologic_tumor_stage"].map(STAGE_MAP)
    n_missing_stage = clin["tumor_stage_numeric"].isna().sum()
    log.info(f"Tumour stage mapping: {n_missing_stage} patients have "
             f"unmappable stage (will be NaN in Cox covariate model)")

    # ── Enforce consistent column types ────────────────────────────────────
    clin["age_at_diagnosis"]   = pd.to_numeric(clin["age_at_diagnosis"], errors="coerce")
    clin["days_to_death"]      = pd.to_numeric(clin["days_to_death"], errors="coerce")
    clin["days_to_last_followup"] = pd.to_numeric(clin["days_to_last_followup"],
                                                   errors="coerce")

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("\nFinal clinical summary:")
    log.info(f"  Patients:             {len(clin)}")
    log.info(f"  Events (Dead):        {clin['os_event'].sum()}")
    log.info(f"  Censored:             {(clin['os_event'] == 0).sum()}")
    log.info(f"  Median OS (days):     {clin['os_time'].median():.0f}")
    log.info(f"  OS range (days):      {clin['os_time'].min():.0f} – "
             f"{clin['os_time'].max():.0f}")
    log.info(f"  Mean age:             {clin['age_at_diagnosis'].mean():.1f} years")
    log.info(f"  PAM50 labelled:       {clin['pam50_subtype'].notna().sum()}")
    if clin["pam50_subtype"].notna().any():
        log.info(f"  PAM50 distribution:\n"
                 f"{clin['pam50_subtype'].value_counts().to_string()}")
    log.info(f"  Stage distribution:\n"
             f"{clin['tumor_stage_numeric'].value_counts().sort_index().to_string()}")

    # ── Save ────────────────────────────────────────────────────────────────
    log.info(f"Saving {CLIN_OUT} ...")
    clin.to_parquet(CLIN_OUT)

    # Save final cohort list (used by all downstream scripts)
    final_patients = sorted(clin.index.tolist())
    with open(FINAL_COHORT_OUT, "w") as f:
        for pid in final_patients:
            f.write(pid + "\n")
    log.info(f"Saved final cohort → {FINAL_COHORT_OUT}  "
             f"({len(final_patients)} patients)")
    log.info("Clinical preprocessing done.")

    return final_patients


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║         STEP 4: PREPROCESS ALL MODALITIES            ║")
    log.info("╚═══════════════════════════════════════════════════════╝")

    cohort = load_final_cohort()   # 755 patients from modality intersection

    # ── Clinical first — establishes final cohort after OS exclusions ───────
    # (other modalities use the full 755; downstream scripts use final_cohort.txt)
    if not checkpoint_exists(CLIN_OUT) or not checkpoint_exists(FINAL_COHORT_OUT):
        final_patients = preprocess_clinical(cohort)
    else:
        final_patients_path = PROCESSED / "final_cohort.txt"
        with open(final_patients_path) as f:
            final_patients = [l.strip() for l in f if l.strip()]
        log.info(f"Final cohort loaded from checkpoint: {len(final_patients)} patients")

    # ── RNA ─────────────────────────────────────────────────────────────────
    if not checkpoint_exists(RNA_OUT):
        preprocess_rna(cohort)

    # ── CNV ─────────────────────────────────────────────────────────────────
    if not checkpoint_exists(CNV_OUT):
        preprocess_cnv(cohort)

    # ── Methylation (heaviest step) ─────────────────────────────────────────
    if not checkpoint_exists(METH_OUT):
        preprocess_methylation(cohort)

    log.info("")
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║         PREPROCESSING COMPLETE                       ║")
    log.info("╠═══════════════════════════════════════════════════════╣")

    # ── Final verification: confirm all three matrices have same patients ───
    # Preprocessed matrices are tiny (~15 MB each) — safe to load fully for check
    log.info("Verifying patient alignment across preprocessed matrices...")
    rna_cols  = pd.read_parquet(RNA_OUT).columns.tolist()
    cnv_cols  = pd.read_parquet(CNV_OUT).columns.tolist()
    meth_cols = pd.read_parquet(METH_OUT).columns.tolist()

    rna_set  = set(rna_cols)
    cnv_set  = set(cnv_cols)
    meth_set = set(meth_cols)
    clin_set = set(final_patients)

    log.info(f"║  RNA  patients:   {len(rna_set)}")
    log.info(f"║  CNV  patients:   {len(cnv_set)}")
    log.info(f"║  METH patients:   {len(meth_set)}")
    log.info(f"║  CLIN patients:   {len(clin_set)}")

    all_common = rna_set & cnv_set & meth_set & clin_set
    log.info(f"║  Shared (all 4):  {len(all_common)}")

    if rna_set != cnv_set or rna_set != meth_set:
        log.warning("Patient sets differ across modalities — "
                    "downstream scripts must intersect with final_cohort.txt")
    else:
        log.info("║  ✓ All modalities share the same patient set")

    log.info("╚═══════════════════════════════════════════════════════╝")
    log.info("\n✓ Next: python scripts/05_graph_construction.py")
