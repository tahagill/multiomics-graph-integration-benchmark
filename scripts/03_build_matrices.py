"""
03_build_matrices.py
--------------------
Parse all downloaded raw files and build four unified matrices:
  1. RNA-seq    → genes × samples    (unstranded STAR counts)
  2. Methylation → CpGs × samples   (beta values)
  3. CNV         → genes × samples   (ASCAT3 copy number)
  4. Clinical    → samples × fields  (survival + covariates)

Then intersect all modalities to a common patient set.

Format notes (from data validation):
  - RNA-seq:     TSV, skip '#' comment line + 4 N_* summary rows, use 'unstranded'
  - Methylation: Plain text, NO header, cols: probe_id  beta_value
  - CNV:         TSV with header, use 'copy_number'; ~815 missing/file → impute with 2
  - Clinical:    BCR Biotab TSV, 3 leading rows (names, duplicate, CDE IDs) → skiprows=[1,2]

ID mapping:
  - Files live in data/raw/<modality>/<file_uuid>/<filename>
  - Manifests (metadata/*_meta.tsv) map file_id → submitter_id (TCGA-XX-XXXX-01A-...)
  - Patient ID = first 12 chars of submitter_id (TCGA-XX-XXXX)

Outputs (data/processed/):
  - rna_matrix_raw.parquet      (genes × patients, raw counts)
  - methylation_matrix_raw.parquet (CpGs × patients, beta values)
  - cnv_matrix_raw.parquet      (genes × patients, copy number)
  - clinical_matrix.parquet     (patients × clinical fields)
  - sample_intersection.txt     (final matched patient IDs)

Run from project root:
  python scripts/03_build_matrices.py
"""

import os
import sys
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
RAW_DIR      = Path("data/raw")
PROCESSED    = Path("data/processed")
METADATA_DIR = Path("metadata")
LOG_DIR      = Path("logs")

PROCESSED.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_build_matrices.log"),
        logging.StreamHandler(sys.stdout),
    ]
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


def patient_id(barcode: str) -> str:
    """Extract 12-char patient ID from TCGA barcode (TCGA-XX-XXXX-...)."""
    return "-".join(barcode.split("-")[:3])


def load_manifest(path: Path) -> dict:
    """
    Load a *_meta.tsv manifest and return {file_id: patient_id} dict.
    Handles duplicate patient IDs by keeping only one file per patient.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    # submitter_id is the TCGA barcode
    df = df.dropna(subset=["file_id", "submitter_id"])
    df["patient_id"] = df["submitter_id"].apply(patient_id)
    # If a patient has multiple files (duplicate aliquots), keep first only
    df = df.drop_duplicates(subset="patient_id", keep="first")
    return dict(zip(df["file_id"], df["patient_id"]))


# ──────────────────────────────────────────────────────────────────────────────
# 1. RNA-seq matrix
# ──────────────────────────────────────────────────────────────────────────────
RNA_OUT = PROCESSED / "rna_matrix_raw.parquet"

def build_rna_matrix():
    log.info("=" * 60)
    log.info("BUILDING RNA-SEQ MATRIX")
    log.info("=" * 60)

    id_map = load_manifest(METADATA_DIR / "manifest_rnaseq_meta.tsv")
    log.info(f"Manifest: {len(id_map)} unique patients mapped")

    rna_dir = RAW_DIR / "rnaseq"
    file_dirs = [d for d in rna_dir.iterdir() if d.is_dir()]
    log.info(f"Directories found: {len(file_dirs)}")

    columns = {}   # patient_id → pd.Series(gene_id → count)
    skipped = 0

    for i, file_dir in enumerate(file_dirs):
        file_id = file_dir.name
        patient = id_map.get(file_id)
        if patient is None:
            skipped += 1
            continue

        # Find the TSV file (STAR counts)
        tsv_files = list(file_dir.glob("*.rna_seq.augmented_star_gene_counts.tsv"))
        if not tsv_files:
            tsv_files = list(file_dir.glob("*.tsv"))
        if not tsv_files:
            log.warning(f"No TSV found in {file_dir}")
            skipped += 1
            continue

        tsv_path = tsv_files[0]

        try:
            # Skip the first '#' comment line, read rest
            df = pd.read_csv(
                tsv_path, sep="\t", comment="#",
                usecols=["gene_id", "unstranded"],
                dtype={"gene_id": str, "unstranded": "Int64"}
            )
            # Drop the 4 N_* summary rows at top
            df = df[~df["gene_id"].str.startswith("N_")]
            df = df.set_index("gene_id")["unstranded"]
            columns[patient] = df

        except Exception as e:
            log.warning(f"Failed to parse {tsv_path}: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0:
            log.info(f"  RNA-seq: processed {i+1}/{len(file_dirs)} files...")

    log.info(f"Loaded {len(columns)} patients, skipped {skipped}")
    log.info("Assembling matrix (genes × patients)...")

    matrix = pd.DataFrame(columns)  # genes × patients
    log.info(f"RNA-seq matrix shape: {matrix.shape}")
    log.info(f"Missing values: {matrix.isnull().sum().sum()}")

    log.info(f"Saving -> {RNA_OUT}")
    matrix.to_parquet(RNA_OUT)
    log.info("RNA-seq matrix saved.")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# 2. Methylation matrix
# ──────────────────────────────────────────────────────────────────────────────
METH_OUT = PROCESSED / "methylation_matrix_raw.parquet"

def build_methylation_matrix():
    log.info("=" * 60)
    log.info("BUILDING METHYLATION MATRIX")
    log.info("=" * 60)

    id_map = load_manifest(METADATA_DIR / "manifest_methylation_meta.tsv")
    log.info(f"Manifest: {len(id_map)} unique patients mapped")

    meth_dir = RAW_DIR / "methylation"
    file_dirs = [d for d in meth_dir.iterdir() if d.is_dir()]
    log.info(f"Directories found: {len(file_dirs)}")

    # ------------------------------------------------------------------
    # Memory-safe approach: pre-allocate a numpy float32 matrix and fill
    # column-by-column. All GDC Illumina 450K files share the same probe
    # order, so no alignment is needed — just read the values column.
    #
    # Peak RAM: ~486K × 793 × 4 bytes ≈ 1.5 GB  (vs ~6 GB with dict approach)
    # ------------------------------------------------------------------

    # Pass 1: collect valid (file_dir, patient_id) pairs and read probe IDs
    # from the first valid file.
    valid_pairs = []
    probe_ids = None
    skipped = 0

    for file_dir in file_dirs:
        file_id = file_dir.name
        patient = id_map.get(file_id)
        if patient is None:
            skipped += 1
            continue

        txt_files = list(file_dir.glob("*.methylation_array.sesame.level3betas.txt"))
        if not txt_files:
            txt_files = [f for f in file_dir.glob("*.txt")
                         if "annotation" not in f.name.lower()]
        if not txt_files:
            skipped += 1
            continue

        valid_pairs.append((file_dir, txt_files[0], patient))

        # Read probe IDs once from the first valid file
        if probe_ids is None:
            tmp = pd.read_csv(txt_files[0], sep="\t", header=None,
                              names=["probe_id", "beta_value"],
                              usecols=["probe_id"], dtype=str)
            probe_ids = tmp["probe_id"].values
            n_probes = len(probe_ids)
            log.info(f"Probe IDs loaded from first file: {n_probes:,} probes")

    n_patients = len(valid_pairs)
    log.info(f"Valid files: {n_patients} | Skipped: {skipped}")

    # Pre-allocate matrix: probes × patients, float32
    log.info(f"Pre-allocating matrix ({n_probes:,} × {n_patients}) float32 "
             f"≈ {n_probes * n_patients * 4 / 1e9:.2f} GB")
    data = np.empty((n_probes, n_patients), dtype=np.float32)
    data[:] = np.nan
    patient_ids = []

    # Pass 2: fill matrix column by column — only beta values loaded per iteration
    for i, (file_dir, txt_path, patient) in enumerate(valid_pairs):
        try:
            col = pd.read_csv(
                txt_path, sep="\t", header=None,
                names=["probe_id", "beta_value"],
                usecols=["beta_value"],
                dtype={"beta_value": np.float32}
            )["beta_value"].values

            if len(col) != n_probes:
                log.warning(f"Probe count mismatch in {txt_path}: "
                            f"expected {n_probes}, got {len(col)} — skipping")
                skipped += 1
                continue

            data[:, i] = col
            patient_ids.append(patient)

        except Exception as e:
            log.warning(f"Failed to parse {txt_path}: {e}")
            data[:, i] = np.nan
            patient_ids.append(patient)

        if (i + 1) % 100 == 0:
            log.info(f"  Methylation: processed {i+1}/{n_patients} files...")

    # Trim to successfully loaded patients
    n_loaded = len(patient_ids)
    data = data[:, :n_loaded]

    log.info(f"Loaded {n_loaded} patients")
    log.info("Building DataFrame...")

    matrix = pd.DataFrame(data, index=probe_ids, columns=patient_ids)
    log.info(f"Methylation matrix shape: {matrix.shape}")
    missing = int(np.isnan(data).sum())
    log.info(f"Missing values: {missing:,}")

    log.info(f"Saving -> {METH_OUT}")
    matrix.to_parquet(METH_OUT)
    log.info("Methylation matrix saved.")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# 3. CNV matrix
# ──────────────────────────────────────────────────────────────────────────────
CNV_OUT = PROCESSED / "cnv_matrix_raw.parquet"

def build_cnv_matrix():
    log.info("=" * 60)
    log.info("BUILDING CNV MATRIX")
    log.info("=" * 60)

    id_map = load_manifest(METADATA_DIR / "manifest_cnv_meta.tsv")
    log.info(f"Manifest: {len(id_map)} unique patients mapped")

    cnv_dir = RAW_DIR / "cnv"
    file_dirs = [d for d in cnv_dir.iterdir() if d.is_dir()]
    log.info(f"Directories found: {len(file_dirs)}")

    columns = {}
    skipped = 0
    total_imputed = 0

    for i, file_dir in enumerate(file_dirs):
        file_id = file_dir.name
        patient = id_map.get(file_id)
        if patient is None:
            skipped += 1
            continue

        tsv_files = list(file_dir.glob("*.gene_level_copy_number*.tsv"))
        if not tsv_files:
            tsv_files = list(file_dir.glob("*.tsv"))
        if not tsv_files:
            log.warning(f"No CNV TSV in {file_dir}")
            skipped += 1
            continue

        tsv_path = tsv_files[0]

        try:
            df = pd.read_csv(
                tsv_path, sep="\t",
                usecols=["gene_id", "copy_number"],
                dtype={"gene_id": str, "copy_number": np.float32}
            )
            # Impute missing copy_number with 2 (neutral diploid)
            # ~815 missing per file in centromeric/repetitive regions
            n_missing = df["copy_number"].isnull().sum()
            if n_missing > 0:
                df["copy_number"] = df["copy_number"].fillna(2.0)
                total_imputed += n_missing

            df = df.set_index("gene_id")["copy_number"]
            columns[patient] = df

        except Exception as e:
            log.warning(f"Failed to parse {tsv_path}: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0:
            log.info(f"  CNV: processed {i+1}/{len(file_dirs)} files...")

    log.info(f"Loaded {len(columns)} patients, skipped {skipped}")
    log.info(f"Total missing copy_number values imputed with 2: {total_imputed:,}")
    log.info("Assembling matrix (genes × patients)...")

    matrix = pd.DataFrame(columns)
    log.info(f"CNV matrix shape: {matrix.shape}")

    log.info(f"Saving -> {CNV_OUT}")
    matrix.to_parquet(CNV_OUT)
    log.info("CNV matrix saved.")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# 4. Clinical matrix
# ──────────────────────────────────────────────────────────────────────────────
CLINICAL_OUT = PROCESSED / "clinical_matrix.parquet"

# BCR Biotab actual column names (differ from GDC data dictionary defaults)
CLINICAL_COLS = [
    "bcr_patient_barcode",
    "vital_status",
    "death_days_to",            # actual name for days_to_death
    "last_contact_days_to",     # actual name for days_to_last_followup
    "age_at_diagnosis",         # actual name for age_at_initial_pathologic_diagnosis
    "ajcc_pathologic_tumor_stage",
    "er_status_by_ihc",
    "pr_status_by_ihc",
    "her2_status_by_ihc",
]

# Rename BCR Biotab names → clean standardised names used everywhere downstream
CLINICAL_RENAME = {
    "death_days_to":        "days_to_death",
    "last_contact_days_to": "days_to_last_followup",
    "age_at_diagnosis":     "age_at_diagnosis",
}

def build_clinical_matrix():
    log.info("=" * 60)
    log.info("BUILDING CLINICAL MATRIX")
    log.info("=" * 60)

    # The main clinical file: nationwidechildrens.org_clinical_patient_brca.txt
    # BCR Biotab format: row 0 = column names, row 1 = duplicate names, row 2 = CDE IDs
    # Data starts at row 3 → skiprows=[1, 2]
    patient_file = RAW_DIR / "clinical" / "8162d394-8b64-4da2-9f5b-d164c54b9608" / \
                   "nationwidechildrens.org_clinical_patient_brca.txt"

    if not patient_file.exists():
        # Try to find it
        found = list((RAW_DIR / "clinical").glob("**/nationwidechildrens.org_clinical_patient_brca.txt"))
        if not found:
            log.error("Clinical patient file not found!")
            sys.exit(1)
        patient_file = found[0]

    log.info(f"Reading: {patient_file}")

    df = pd.read_csv(
        patient_file, sep="\t",
        skiprows=[1, 2],   # skip duplicate header row and CDE ID row
        dtype=str,
        low_memory=False
    )

    log.info(f"Raw clinical rows: {len(df)}, columns: {len(df.columns)}")

    # Select only columns we need (keep what's available)
    available = [c for c in CLINICAL_COLS if c in df.columns]
    missing_cols = [c for c in CLINICAL_COLS if c not in df.columns]
    if missing_cols:
        log.warning(f"Columns not found (will be NaN): {missing_cols}")

    df = df[available].copy()
    df = df.rename(columns={"bcr_patient_barcode": "patient_id"})

    # Rename BCR Biotab column names to clean standardised names
    df = df.rename(columns=CLINICAL_RENAME)

    # Convert numeric columns (after renaming to clean names)
    numeric_cols = ["days_to_death", "days_to_last_followup", "age_at_diagnosis"]
    for col in numeric_cols:
        if col in df.columns:
            # BCR Biotab uses '[Not Applicable]', '[Not Available]' etc. → coerce to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no patient barcode
    df = df.dropna(subset=["patient_id"])
    df = df[df["patient_id"].str.startswith("TCGA")]
    df = df.set_index("patient_id")

    log.info(f"Clinical matrix shape: {df.shape}")
    log.info(f"Vital status distribution:\n{df['vital_status'].value_counts().to_string()}")

    # Merge PAM50 labels
    pam50 = pd.read_csv(METADATA_DIR / "pam50_labels.tsv", sep="\t")
    pam50 = pam50.set_index("patient_id")
    df = df.join(pam50, how="left")
    n_with_pam50 = df["pam50_subtype"].notna().sum()
    log.info(f"Patients with PAM50 label: {n_with_pam50}/{len(df)}")
    log.info(f"PAM50 distribution:\n{df['pam50_subtype'].value_counts().to_string()}")

    log.info(f"Saving -> {CLINICAL_OUT}")
    df.to_parquet(CLINICAL_OUT)
    log.info("Clinical matrix saved.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Sample intersection
# ──────────────────────────────────────────────────────────────────────────────
INTERSECTION_OUT = PROCESSED / "sample_intersection.txt"

def build_intersection():
    log.info("=" * 60)
    log.info("COMPUTING SAMPLE INTERSECTION")
    log.info("=" * 60)

    rna   = pd.read_parquet(RNA_OUT)
    meth  = pd.read_parquet(METH_OUT)
    cnv   = pd.read_parquet(CNV_OUT)
    clin  = pd.read_parquet(CLINICAL_OUT)

    rna_patients  = set(rna.columns)
    meth_patients = set(meth.columns)
    cnv_patients  = set(cnv.columns)
    clin_patients = set(clin.index)

    log.info(f"RNA-seq patients:    {len(rna_patients)}")
    log.info(f"Methylation patients:{len(meth_patients)}")
    log.info(f"CNV patients:        {len(cnv_patients)}")
    log.info(f"Clinical patients:   {len(clin_patients)}")

    # Intersect RNA + Methylation + CNV (clinical is less strict — many patients have it)
    common = rna_patients & meth_patients & cnv_patients
    log.info(f"\nIntersection (RNA ∩ Methylation ∩ CNV): {len(common)}")

    # Also check how many have clinical data
    with_clinical = common & clin_patients
    log.info(f"Of those, with clinical data: {len(with_clinical)}")

    # Check PAM50 coverage
    pam50 = pd.read_csv(METADATA_DIR / "pam50_labels.tsv", sep="\t")
    pam50_set = set(pam50["patient_id"])
    with_pam50 = common & pam50_set
    log.info(f"Of those, with PAM50 label:   {len(with_pam50)}")

    # Use intersection with clinical as final cohort
    final = sorted(with_clinical)
    log.info(f"\nFINAL COHORT SIZE: {len(final)} patients")

    # Save
    with open(INTERSECTION_OUT, "w") as f:
        for pid in final:
            f.write(pid + "\n")
    log.info(f"Saved -> {INTERSECTION_OUT}")

    # Summary table
    log.info("\n" + "=" * 50)
    log.info("COHORT SUMMARY")
    log.info("=" * 50)
    log.info(f"  Final patients (all modalities + clinical): {len(final)}")
    log.info(f"  With PAM50 labels:                         {len(with_pam50 & set(final))}")
    log.info(f"  Missing PAM50:                             {len(set(final) - pam50_set)}")

    return final


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║         STEP 3: BUILD MATRICES                      ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # Each step checks for checkpoint — safe to re-run after interruption
    if not checkpoint_exists(RNA_OUT):
        build_rna_matrix()

    if not checkpoint_exists(METH_OUT):
        build_methylation_matrix()

    if not checkpoint_exists(CNV_OUT):
        build_cnv_matrix()

    if not checkpoint_exists(CLINICAL_OUT):
        build_clinical_matrix()

    if not checkpoint_exists(INTERSECTION_OUT):
        build_intersection()

    log.info("\n✓ All matrices built. Next: python scripts/04_preprocess.py")
