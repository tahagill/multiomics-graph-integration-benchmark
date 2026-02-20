"""
02b_fetch_pam50.py
------------------
PAM50 subtype labels are NOT stored in GDC clinical files.
This script fetches them from cBioPortal, which hosts the official
TCGA BRCA PAM50 calls.

Source: cBioPortal API — TCGA Breast Invasive Carcinoma (TCGA, PanCancer Atlas)
Study ID: brca_tcga_pan_can_atlas_2018

Output: metadata/pam50_labels.tsv
Columns: patient_id, pam50_subtype

Run from project root:
  python scripts/02b_fetch_pam50.py
"""

import requests
import pandas as pd
import os

METADATA_DIR = "metadata"
OUT_PATH = os.path.join(METADATA_DIR, "pam50_labels.tsv")
CBIOPORTAL_API = "https://www.cbioportal.org/api"

# Study with best PAM50 coverage for TCGA BRCA
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
ATTR_ID = "SUBTYPE"

print(f"Fetching PAM50 labels from cBioPortal...")
print(f"Study: {STUDY_ID}")
print(f"Attribute: {ATTR_ID}")

# Fetch all clinical data for this attribute
url = f"{CBIOPORTAL_API}/studies/{STUDY_ID}/clinical-data"
params = {
    "clinicalDataType": "PATIENT",
    "attributeId": ATTR_ID,
    "projection": "SUMMARY",
    "pageSize": 2000,
    "pageNumber": 0,
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
data = response.json()

print(f"Records returned: {len(data)}")

if len(data) == 0:
    # Try SAMPLE level
    print("No PATIENT-level data. Trying SAMPLE level...")
    params["clinicalDataType"] = "SAMPLE"
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"Records returned (sample level): {len(data)}")

rows = []
for record in data:
    patient_id = record.get("patientId", "")
    subtype = record.get("value", "")
    rows.append({"patient_id": patient_id, "pam50_subtype": subtype})

df = pd.DataFrame(rows)

# Show value distribution
print("\nPAM50 subtype distribution:")
print(df["pam50_subtype"].value_counts().to_string())

# Save
df.to_csv(OUT_PATH, sep="\t", index=False)
print(f"\nSaved -> {OUT_PATH} ({len(df)} patients)")

# Quick sanity check
n_valid = df["pam50_subtype"].str.startswith("BRCA").sum()
print(f"Entries with BRCA subtype prefix: {n_valid}")
print("\nSample rows:")
print(df.head(5).to_string(index=False))
