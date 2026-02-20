#!/usr/bin/env bash
# =============================================================================
# 02_download_data.sh
# Download all TCGA-BRCA modalities using gdc-client manifests.
#
# Run from project root:
#   bash scripts/02_download_data.sh
#
# Downloads are saved to data/raw/<modality>/
# Logs are saved to logs/
# =============================================================================

set -euo pipefail

GDC_CLIENT="./gdc-client-bin/gdc-client"
RAW_DIR="data/raw"
LOG_DIR="logs"
N_CONNECTIONS=5   # parallel connections per download — safe for HDD + stable internet

# Verify gdc-client exists
if [[ ! -x "$GDC_CLIENT" ]]; then
    echo "ERROR: gdc-client not found at $GDC_CLIENT"
    echo "Run: chmod +x gdc-client-bin/gdc-client"
    exit 1
fi

echo "============================================================"
echo "TCGA-BRCA DATA DOWNLOAD"
echo "Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
download_modality() {
    local label="$1"
    local manifest="$2"
    local outdir="$3"
    local logfile="$4"

    mkdir -p "$outdir"

    # Count files in manifest (subtract header line)
    local n_files=$(( $(wc -l < "$manifest") - 1 ))
    echo ""
    echo "[$label] Files to download: $n_files"
    echo "[$label] Output: $outdir"
    echo "[$label] Log: $logfile"
    echo "[$label] Starting at $(date)..."

    "$GDC_CLIENT" download \
        --manifest "$manifest" \
        --dir "$outdir" \
        --n-processes "$N_CONNECTIONS" \
        --log-file "$logfile" \
        --retry-amount 3 \
        --wait-time 5 \
        --no-segment-md5sums \
        2>&1 | tee -a "$logfile"

    echo "[$label] Completed at $(date)"
}

# ------------------------------------------------------------
# 1. RNA-seq (STAR - Counts)
# Expected: ~1111 files, ~2-3 GB total
# ------------------------------------------------------------
download_modality \
    "RNA-seq" \
    "metadata/manifest_rnaseq.txt" \
    "$RAW_DIR/rnaseq" \
    "$LOG_DIR/download_rnaseq.log"

# ------------------------------------------------------------
# 2. DNA Methylation 450K (Beta Values)
# Expected: ~793 files, ~8-12 GB total — largest download
# ------------------------------------------------------------
download_modality \
    "Methylation" \
    "metadata/manifest_methylation.txt" \
    "$RAW_DIR/methylation" \
    "$LOG_DIR/download_methylation.log"

# ------------------------------------------------------------
# 3. Copy Number Variation (ASCAT3, Gene Level)
# Expected: ~1060 files, ~500 MB total
# ------------------------------------------------------------
download_modality \
    "CNV" \
    "metadata/manifest_cnv.txt" \
    "$RAW_DIR/cnv" \
    "$LOG_DIR/download_cnv.log"

# ------------------------------------------------------------
# 4. Clinical Supplement (BCR Biotab — contains PAM50 labels)
# Expected: 9 files, <50 MB
# ------------------------------------------------------------
download_modality \
    "Clinical" \
    "metadata/manifest_clinical.txt" \
    "$RAW_DIR/clinical" \
    "$LOG_DIR/download_clinical.log"

# ------------------------------------------------------------
# Storage report
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "DOWNLOAD COMPLETE: $(date)"
echo "============================================================"
echo ""
echo "Storage usage:"
du -sh "$RAW_DIR"/*/  2>/dev/null || echo "  (no subdirectories found)"
echo ""
du -sh "$RAW_DIR"
echo ""
echo "Next step: python scripts/03_build_matrices.py"
