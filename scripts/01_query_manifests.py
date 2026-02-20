"""
01_query_manifests.py
---------------------
Query the GDC API for TCGA-BRCA file manifests for all four modalities:
  - RNA-seq (HTSeq Counts)
  - DNA Methylation 450K (Beta Values)
  - Copy Number Variation (Gene Level)
  - Clinical data (case-level)

Outputs (saved to metadata/):
  - manifest_rnaseq.tsv
  - manifest_methylation.tsv
  - manifest_cnv.tsv
  - manifest_clinical.tsv   (case UUIDs for clinical download)

Run from project root:
  python scripts/01_query_manifests.py
"""

import requests
import json
import pandas as pd
import os
import sys

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Shared filters
# ------------------------------------------------------------------------------
BASE_FILTERS = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}


def add_filter(base, extra_filters):
    """Return a new filter dict combining base filters with additional ones."""
    return {
        "op": "and",
        "content": base["content"] + extra_filters
    }


def query_manifest(filters, fields, label, out_filename, size=2000):
    """
    Query GDC files endpoint, fetch all matching files, save as TSV manifest.
    Returns the resulting DataFrame.
    """
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": size,
    }

    print(f"\n[{label}] Querying GDC API...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]
    print(f"[{label}] Total files found: {total} (retrieved: {len(hits)})")

    if len(hits) < total:
        print(f"[{label}] WARNING: retrieved {len(hits)} of {total}. Increase size parameter.")

    rows = []
    for h in hits:
        row = {"file_id": h["file_id"], "file_name": h.get("file_name", "")}
        # Flatten nested case/sample info
        cases = h.get("cases", [])
        if cases:
            case = cases[0]
            row["case_id"] = case.get("case_id", "")
            row["submitter_id"] = case.get("submitter_id", "")
            samples = case.get("samples", [])
            if samples:
                row["sample_type"] = samples[0].get("sample_type", "")
                row["sample_id"] = samples[0].get("sample_id", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(METADATA_DIR, out_filename)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[{label}] Saved manifest -> {out_path} ({len(df)} rows)")
    return df


def query_gdc_manifest_format(filters, label, out_filename, size=2000):
    """
    Fetch the GDC-format manifest (id/filename/md5/size/state columns)
    suitable for direct use with gdc-client download.
    """
    params = {
        "filters": json.dumps(filters),
        "return_type": "manifest",
        "size": size,
    }

    print(f"\n[{label}] Fetching gdc-client manifest...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()

    out_path = os.path.join(METADATA_DIR, out_filename)
    with open(out_path, "w") as f:
        f.write(response.text)

    # Count lines (subtract header)
    lines = [l for l in response.text.strip().split("\n") if l]
    n_files = len(lines) - 1
    print(f"[{label}] gdc-client manifest saved -> {out_path} ({n_files} files)")
    return n_files


# ------------------------------------------------------------------------------
# 1. RNA-seq (HTSeq - Counts)
# ------------------------------------------------------------------------------
# NOTE: HTSeq - Counts was retired in GDC Data Release v38.
# Current harmonized workflow is STAR - Counts (1231 files for TCGA-BRCA).
rna_filters = add_filter(BASE_FILTERS, [
    {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
    {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
    {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
    {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
])

rna_fields = ["file_id", "file_name", "cases.case_id", "cases.submitter_id",
              "cases.samples.sample_type", "cases.samples.sample_id"]

query_manifest(rna_filters, rna_fields, "RNA-seq", "manifest_rnaseq_meta.tsv")
query_gdc_manifest_format(rna_filters, "RNA-seq", "manifest_rnaseq.txt")

# ------------------------------------------------------------------------------
# 2. DNA Methylation 450K (Beta Values)
# ------------------------------------------------------------------------------
meth_filters = add_filter(BASE_FILTERS, [
    {"op": "=", "content": {"field": "data_category", "value": "DNA Methylation"}},
    {"op": "=", "content": {"field": "data_type", "value": "Methylation Beta Value"}},
    {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}},
])

query_manifest(meth_filters, rna_fields, "Methylation", "manifest_methylation_meta.tsv")
query_gdc_manifest_format(meth_filters, "Methylation", "manifest_methylation.txt")

# ------------------------------------------------------------------------------
# 3. Copy Number Variation (Gene Level)
# ------------------------------------------------------------------------------
# NOTE: Gene Level Copy Number has 4 caller workflows (ASCAT2, ASCAT3, ABSOLUTE LiftOver, AscatNGS).
# We use ASCAT3: the most current GDC harmonized pipeline (~1060 primary tumor files).
cnv_filters = add_filter(BASE_FILTERS, [
    {"op": "=", "content": {"field": "data_category", "value": "Copy Number Variation"}},
    {"op": "=", "content": {"field": "data_type", "value": "Gene Level Copy Number"}},
    {"op": "=", "content": {"field": "analysis.workflow_type", "value": "ASCAT3"}},
])

query_manifest(cnv_filters, rna_fields, "CNV", "manifest_cnv_meta.tsv")
query_gdc_manifest_format(cnv_filters, "CNV", "manifest_cnv.txt")

# ------------------------------------------------------------------------------
# 4. Clinical data — fetch case UUIDs for clinical supplement download
# ------------------------------------------------------------------------------
print("\n[Clinical] Querying GDC cases endpoint...")
clinical_params = {
    "filters": json.dumps({
        "op": "=",
        "content": {"field": "project.project_id", "value": "TCGA-BRCA"}
    }),
    "fields": "case_id,submitter_id,diagnoses.vital_status,diagnoses.days_to_death,"
              "diagnoses.days_to_last_follow_up,demographic.age_at_index,"
              "diagnoses.ajcc_pathologic_stage",
    "format": "JSON",
    "size": 2000,
}

resp = requests.get(GDC_CASES_ENDPOINT, params=clinical_params, timeout=60)
resp.raise_for_status()
cases_data = resp.json()["data"]["hits"]
total_cases = resp.json()["data"]["pagination"]["total"]
print(f"[Clinical] Total TCGA-BRCA cases: {total_cases} (retrieved: {len(cases_data)})")

rows = []
for c in cases_data:
    row = {
        "case_id": c.get("case_id", ""),
        "submitter_id": c.get("submitter_id", ""),
    }
    diags = c.get("diagnoses", [{}])
    d = diags[0] if diags else {}
    row["vital_status"] = d.get("vital_status", "")
    row["days_to_death"] = d.get("days_to_death", "")
    row["days_to_last_follow_up"] = d.get("days_to_last_follow_up", "")
    row["ajcc_pathologic_stage"] = d.get("ajcc_pathologic_stage", "")
    demo = c.get("demographic", {})
    row["age_at_index"] = demo.get("age_at_index", "") if demo else ""
    rows.append(row)

clinical_df = pd.DataFrame(rows)
clinical_path = os.path.join(METADATA_DIR, "clinical_cases.tsv")
clinical_df.to_csv(clinical_path, sep="\t", index=False)
print(f"[Clinical] Saved -> {clinical_path} ({len(clinical_df)} rows)")

# Fetch clinical supplement files (BCR Biotab TSVs — includes PAM50 subtype labels).
# Filter: TCGA-BRCA, Clinical category, BCR XML or Biotab format.
# Note: do NOT include sample_type or access filters — clinical files are case-level.
clinical_file_filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Clinical"}},
        {"op": "=", "content": {"field": "data_format", "value": "BCR Biotab"}},
    ]
}
query_manifest(clinical_file_filters,
               ["file_id", "file_name", "data_type", "data_format"],
               "Clinical Supplement", "manifest_clinical_meta.tsv", size=100)
query_gdc_manifest_format(clinical_file_filters, "Clinical Supplement", "manifest_clinical.txt", size=100)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("MANIFEST QUERY COMPLETE")
print("="*60)
print(f"Output directory: {os.path.abspath(METADATA_DIR)}/")
for f in sorted(os.listdir(METADATA_DIR)):
    fpath = os.path.join(METADATA_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:45s} {size_kb:7.1f} KB")
print("\nNext step: run scripts/02_download_data.sh")
