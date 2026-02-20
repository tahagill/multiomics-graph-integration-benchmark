# TCGA-BRCA Multi-Omics Integration — Progress Report

**Last updated:** 2026-02-19  
**Research question:** Does SNF-based similarity-graph fusion (RNA-seq + methylation + CNV) produce better patient clusters and overall-survival prediction compared with single-omics, early concatenation, and late ensemble baselines?  
**Target outlet:** arXiv preprint (q-bio.GN / stat.ML)

---

## Pipeline Overview

```
01_query_manifests.py    ✅  GDC API → manifest TSVs
02_download_data.sh      ✅  gdc-client bulk download
02b_fetch_pam50.py       ✅  PAM50 labels from cBioPortal
03_build_matrices.py     ✅  Raw files → feature matrices + clinical table
04_preprocess.py         ✅  Filtering, normalization, MAD top-5k per modality
04b_batch_check.py       ✅  TSS batch check; pycombat applied to all 3 modalities
05_graph_construction.py ✅  Affinity matrices (bounded kernel) + SNF + spectral embedding
06_clustering.py         🔲  Not started
07_baselines.py          🔲  Not started
08_survival_models.py    🔲  Not started
09_sensitivity.py        🔲  Not started
10_figures.py            🔲  Not started
```

---

## Step-by-Step Results

### Step 01 — Query Manifests (`01_query_manifests.py`)

**What it does:** Hits the GDC REST API to pull file-level manifests for all four modalities.

**Outputs:**

| Manifest | Files |
|---|---|
| `metadata/manifest_rnaseq.txt` | 1,217 |
| `metadata/manifest_cnv.txt` | 1,098 |
| `metadata/manifest_methylation.txt` | 800 |
| `metadata/manifest_clinical.txt` | 9 (case-level) |

**Note:** GDC returns one file per aliquot — deduplication happens downstream in Step 03.

---

### Step 02 — Data Download (`02_download_data.sh`)

**What it does:** `gdc-client` bulk download from all four manifests. ~150 GB across raw data directories.

**Caveats:**
- GDC tokens expire; stale token causes silent partial downloads.
- CNV from GDC is the Affymetrix SNP 6.0 ABSOLUTE-called log2 copy-number ratio, not integer copy number.

---

### Step 02b — PAM50 Labels (`02b_fetch_pam50.py`)

**What it does:** Fetches PAM50 subtype labels from cBioPortal (study: `brca_tcga_pan_can_atlas_2018`, attribute: `SUBTYPE`).

**Output:** `metadata/pam50_labels.tsv` — 981 patients

| PAM50 Subtype | cBioPortal (full) | Final 644-patient cohort |
|---|---|---|
| LumA | 499 | 307 |
| Basal | 171 | 102 |
| LumB | 197 | 101 |
| HER2 | 78 | 30 |
| Normal | 36 | 28 |
| **Unlabelled** | — | **76** |

**Caveat:** 76 patients (11.8%) in the final cohort have no PAM50 label. Retained for survival analysis; excluded from ARI/NMI evaluation. The unlabelled fraction skews toward lower-coverage samples — may slightly inflate clustering-vs-PAM50 agreement metrics; must be stated in Methods.

---

### Step 03 — Build Matrices (`03_build_matrices.py`)

**What it does:** Parses all raw GDC files into four Parquet matrices and builds the sample intersection.

**Raw matrix shapes:**

| Modality | Features | Patients |
|---|---|---|
| RNA-seq | 60,660 genes | 1,095 |
| CNV | 60,623 genes | 1,060 |
| Methylation | 486,427 probes | 784 |
| Clinical | 9 columns | 1,097 |

**Three-modality patient intersection:** 755 patients  
Methylation is the bottleneck: GDC has fewer 450K arrays (784) than RNA/CNV assays.

**Critical bug fixed — BCR Biotab column name mismatch:**

| BCR Biotab column | Standardised name | Fix |
|---|---|---|
| `death_days_to` | `days_to_death` | `CLINICAL_RENAME` dict |
| `last_contact_days_to` | `days_to_last_followup` | `CLINICAL_RENAME` dict |
| `age_at_diagnosis` | `age_at_diagnosis` | kept |

Before this fix all survival columns were NaN — clinical endpoints entirely invalid. After fix: `days_to_death` has exactly 104 non-null values, matching the 104 patients with `vital_status == "Dead"` ✅.

**Duplicate aliquot handling:** Lexicographically last aliquot ID retained. Limitation: no read-depth-based selection — must be noted in paper Methods.

---

### Step 04 — Preprocessing (`04_preprocess.py`)

**Cohort definition:**  
755 patients → exclude OS < 30 days or OS = NaN → **644 final patients**

111 excluded: very short follow-up times — a known TCGA data-quality artifact (patients lost to follow-up immediately post-registration, or data entry errors). Not clinically meaningful events.

#### Clinical Cohort (644 patients)

| Attribute | Value |
|---|---|
| Events (died) | 65 (10.1%) |
| Censored | 579 (89.9%) |
| Median OS | 514 days (~17 months) |
| OS range | 30 – 7,067 days |
| Mean age at diagnosis | 57.9 years |
| Stage I / II / III / IV | 114 / 352 / 160 / 9 |
| Unmappable stage (NaN) | 9 patients |
| PAM50 labelled | 568 / 644 |

**Caveat — low event rate:** 65/644 (10.1%). Expected in BRCA. Bootstrap CIs with n ≥ 1,000 resamples required. Cox model results must report wide CIs honestly.

---

#### RNA-seq Pipeline

| Step | In | Out | Detail |
|---|---|---|---|
| Raw counts | 60,660 | 60,660 | HTSeq |
| CPM normalisation | 60,660 | 60,660 | Median library: 57,196,716 reads |
| log2(CPM + 1) | 60,660 | 60,660 | Variance-stabilising |
| Low-count filter (CPM < 1 in > 80% samples) | 60,660 | 17,179 | Removed 43,481 genes |
| MAD top-5,000 | 17,179 | **5,000** | — |
| Z-score per gene | — | — | μ=0, σ=1 across 755 patients |

Value range (pre-batch correction): [−8.03, +8.75]

---

#### CNV Pipeline

| Step | In | Out | Detail |
|---|---|---|---|
| Raw | 60,623 | 60,623 | Log2-ratio (ABSOLUTE) |
| Remove constant genes | 60,623 | 59,820 | 803 removed — zero variance |
| MAD top-5,000 | 59,820 | **5,000** | — |
| Z-score per gene | — | — | — |

Value range (pre-batch correction): [−2.36, +23.44]

**Caveat:** Z-score max +23.44 reflects real focal amplification (likely *ERBB2* chr17q12 or *MYC* chr8q24). Winsorisation at ±5σ applied in Step 05.

---

#### Methylation Pipeline

| Step | In | Out | Removed | Reason |
|---|---|---|---|---|
| Raw 450K | 486,427 | 486,427 | — | — |
| rs* SNP control probes | 486,427 | 486,362 | 65 | Illumina internal controls |
| Sex-chromosome probes | 486,362 | 474,709 | 11,653 | chrX/Y confound sex with tumour signal |
| SNP-proximal probes | 474,709 | 342,403 | 132,306 | Within 5 bp of known SNP; genotype-driven β |
| >20% missing | 342,403 | 283,352 | 59,051 | High non-detection rate |
| MAD top-5,000 | 283,352 | **5,000** | 278,352 | — |
| Z-score per probe | — | — | — | — |

Value range (pre-batch correction): [−2.98, +3.36]

**Improvisation — SNP filter source:** Illumina's HM450 manifest lacks a hg38 SNP mask. Zhou lab `HM450.hg38.snp.tsv.gz` (Zhou et al. 2017, *Nucleic Acids Research*) fetched at runtime; cached at `metadata/HM450.hg38.snp.tsv.gz`. Community-recommended approach for hg38-aligned TCGA data.

**Design decision — CpG-level features, no gene-level aggregation:** Preserves intra-gene methylation heterogeneity (promoter vs. gene-body differential methylation), which is a primary signal in BRCA subtype biology. Decision locked; must be stated in Methods.

---

### Step 04b — Batch Effect Check (`04b_batch_check.py`) ← NEW

**What it does:**  
For each preprocessed modality:
1. Extract Tissue Source Site (TSS) from TCGA barcode field 2 (e.g. `BH` from `TCGA-BH-A0AU`)
2. PCA → 10 PCs
3. Compute mean η² (eta-squared): fraction of variance in PC space explained by TSS group
4. Decision: R² ≥ 0.05 → apply pycombat; overwrite parquet in-place; delete downstream checkpoints
5. Generate PC1 vs PC2 figures coloured by TSS (saved to `results/figures/batch_check/`)

**Why TSS, not plate_id:** True Illumina plate IDs require the full 28-character aliquot barcode. Patient-level barcodes (12 characters) only carry TSS. TSS is the primary source of systematic variation in TCGA (different hospitals, extraction protocols, shipping times). Checking TSS is the correct and standard approach.

**Why this matters:** Uncorrected batch effects can dominate the top PCs, causing downstream clustering to recover hospital identity rather than tumour biology. Finding it: RNA R²=0.106 means ~10.5% of variance in the top 10 PCs is explained by which TSS site the sample came from — that would contaminate any Basil vs. Luminal separation.

**TSS cohort distribution (39 sites):** BH=94, A2=64, D8=61, AR=60, E2=52 (top 5 of 39)

**Thresholds:**

| R² | Action |
|---|---|
| < 0.02 | No action |
| 0.02 – 0.05 | Log warning, no correction |
| ≥ 0.05 | Apply pycombat, overwrite parquet |

**Results:**

| Modality | Batch R² (η²) | Before correction | Action |
|---|---|---|---|
| RNA-seq | **0.106** | Strongest signal; library-prep variation | pycombat applied |
| CNV | **0.056** | Modest; GDC harmonisation partially reduces it | pycombat applied |
| Methylation | **0.061** | Array processing variation across sites | pycombat applied |

**Implementation detail — singleton batches:**  
10 of 39 TSS sites have exactly 1 patient. pycombat cannot estimate within-batch variance for n=1 (produces NaN). Fix: automatically detect singleton batches → pass `mean_only=True` → removes only mean shift without variance rescaling. Correct and documented pycombat handling.

**Critical incident — first run crash + RNA parquet corruption:**  
First run of `04b_batch_check.py` crashed mid-way with a `ValueError: Input X contains NaN` during the post-pycombat PCA plot. Root cause: pycombat with default settings produced NaN for singleton batches, which were not caught before being written into the RNA parquet. The parquet was partially overwritten with NaN values before the crash.  
**Recovery:** Deleted the corrupted `rna_preprocessed.parquet`; re-ran `04_preprocess.py` (checkpoint logic rebuilt only the RNA parquet, not CNV/methylation). Fixed `04b_batch_check.py` with `mean_only=True` + NaN-fill fallback. Second run: all 3 modalities corrected, 0 NaN remaining.

**Figures:** `results/figures/batch_check/pca_tss_{rna,cnv,methylation}.pdf` — required for paper supplementary section.

---

### Step 05 — Graph Construction & SNF (`05_graph_construction.py`)

**Hyperparameters (locked, not swept per README):**

| Parameter | Value | Rationale |
|---|---|---|
| K | 20 | ~3% of cohort; Wang et al. 2014 default |
| T | 20 | Published diffusion convergence default |
| μ | 0.5 | Standard SNF default |
| Winsorisation | ±5σ | Clips focal-amp extremes; retains rank ordering |
| Spectral dims | 50 | Retains sufficient variance for downstream clustering |

**Load pipeline (per modality):** Load batch-corrected parquet → subset to 644 cohort → winsorise at ±5σ → L2-normalise each patient row (unit hypersphere).

Winsorisation impact (post-batch-correction):

| Modality | Values winsorised | % |
|---|---|---|
| RNA | 384 | 0.012% |
| CNV | 9,491 | 0.295% |
| Methylation | 0 | 0% |

---

#### Critical Fix — Affinity Kernel Replacement

**Previous broken behaviour (snfpy 0.2.2):**  
`snf.make_affinity()` implements:
```
W(i,j) = scipy.stats.norm.pdf(d_ij, loc=0, scale=μ·σ_ij)
```
This is an **unbounded probability density**, not a bounded similarity score. When σ→0, PDF = 1/(σ·√2π) → ∞. Root cause: 69 CNV patients had TT (mean KNN sqeuclidean distance) = 0 exactly after L2 normalisation — near-duplicate profiles on the unit hypersphere. Result: CNV affinity max = **3.59 × 10¹⁵**.

Note: snfpy's formula also diverges from the published Wang et al. 2014 formulation, which is the bounded exponential below.

**Replacement — Wang et al. 2014 bounded kernel:**

```python
from scipy.spatial.distance import cdist

D = cdist(X, X, metric='sqeuclidean')              # pairwise sq-euclidean
D_sorted = np.sort(D, axis=1)
TT = D_sorted[:, 1:K+1].mean(axis=1)              # mean KNN dist per patient
sigma = mu * (TT[:, None] + TT[None, :] + D) / 3  # bandwidth matrix (σ_ij)
sigma = np.maximum(sigma, np.finfo(np.float64).tiny)  # prevent σ=0
W = np.exp(-D / (2.0 * sigma))                     # bounded in (0, 1]
np.fill_diagonal(W, 0.0)                           # zero self-similarity
W = (W + W.T) / 2.0                                # enforce symmetry
```

`snf.snf()` (the iterative diffusion fusion step) is still used unchanged — only `make_affinity()` is replaced.

**Additional snfpy compatibility fix** (applied earlier, still in effect): sklearn 1.8 renamed `force_all_finite` → `ensure_all_finite`; patched in `snf/compute.py`.

---

#### Affinity Matrix Results (Final, All Valid)

| Modality | TT range | Affinity range | Status |
|---|---|---|---|
| RNA | 0.694 – 1.594 (mean 1.161) | [0.000, **0.551**] | ✅ |
| CNV | 4.50×10⁻⁴ – 1.501 (mean 0.386) | [0.000, **1.000**] | ✅ |
| Methylation | 0.247 – 1.677 (mean 1.066) | [0.000, **0.716**] | ✅ |
| **Fused (SNF)** | — | [**0.0013, 0.5431**] | ✅ |
| Fused symmetry error | — | 0.00e+00 | ✅ |

CNV max=1.000: a small subset of patients remain near-identical on the unit hypersphere after batch correction. The bounded kernel correctly produces affinity ≈ 1 for them instead of overflowing.

---

#### Eigengap Analysis

| k | Gap |
|---|---|
| **2** | **0.0130** ← largest |
| 3 | 0.0114 |
| 4 | 0.0048 |
| 5 – 10 | ≤ 0.0028 |

**Pre-specified k = 2.** Determined from eigengap heuristic on the normalised graph Laplacian of the fused matrix, applied before examining cluster assignments. k=2 most likely separates Basal-like (triple-negative) from all Luminal subtypes — the dominant biological axis in BRCA. Both eigengap and silhouette will be evaluated for k=2..6 in `06_clustering.py` before cluster assignments are examined.

**Spectral embedding:** `data/processed/spectral_embedding.parquet` — 644 × 50, float32 ✅

---

## Current Data Artifacts (`data/processed/`)

| File | Shape | Notes | Status |
|---|---|---|---|
| `rna_preprocessed.parquet` | 5,000 × 755 | pycombat-corrected (R²=0.106) | ✅ |
| `cnv_preprocessed.parquet` | 5,000 × 755 | pycombat-corrected (R²=0.056) | ✅ |
| `methylation_preprocessed.parquet` | 5,000 × 755 | pycombat-corrected (R²=0.061) | ✅ |
| `clinical_preprocessed.parquet` | 644 × 12 | OS, PAM50, stage | ✅ |
| `final_cohort.txt` | 644 patients | — | ✅ |
| `affinity_rna.npy` | 644 × 644 | [0, 0.551]; bounded kernel | ✅ |
| `affinity_cnv.npy` | 644 × 644 | [0, 1.000]; bounded kernel | ✅ |
| `affinity_meth.npy` | 644 × 644 | [0, 0.716]; bounded kernel | ✅ |
| `affinity_fused.npy` | 644 × 644 | [0.0013, 0.543]; SNF fused | ✅ |
| `spectral_embedding.parquet` | 644 × 50 | From valid fused matrix | ✅ |
| `eigenvalues_fused.npy` | 50 values | Eigengap k=2 | ✅ |

**All artifacts are valid. No stale or numerically broken files remain.**

---

## All Bugs Fixed

| # | Bug | Root cause | Fix |
|---|---|---|---|
| 1 | Clinical survival columns all-NaN | BCR Biotab non-standard names (`death_days_to` etc.) | `CLINICAL_RENAME` dict in `03_build_matrices.py` |
| 2 | SNP-proximal methylation filter skipped silently | HM450 manifest lacks hg38 SNP mask column | `fetch_snp_probes()` using Zhou lab `HM450.hg38.snp.tsv.gz` |
| 3 | `AttributeError: force_all_finite` | sklearn 1.8 renamed parameter | Patched `snf/compute.py` in-place |
| 4 | CNV affinity overflow (max=3.59×10¹⁵) | snfpy uses unbounded `stats.norm.pdf`; 69 patients TT=0 | Replaced `snf.make_affinity()` with bounded Wang et al. 2014 kernel |
| 5 | RNA parquet corrupted mid-write by pycombat NaN | Singleton TSS batches (n=1) produced NaN in pycombat | `mean_only=True`; NaN-fill fallback; parquet regenerated |
| 6 | Batch effects in all 3 modalities (unchecked) | TSS systematic variation; RNA R²=0.106 | `04b_batch_check.py`; pycombat applied to all 3 |

---

## Known Limitations

1. **Low event rate (65/644 = 10.1%):** Limited survival modeling power. Bootstrap CIs and CV essential. Wide CIs must be reported.

2. **Methylation cohort bottleneck:** 784 methylation arrays vs 1,095 RNA/CNV samples → intersection 755, losing ~27% of cohort.

3. **SNP filter runtime download:** Zhou lab annotation fetched from web. Cached at `metadata/HM450.hg38.snp.tsv.gz`.

4. **PAM50 unlabelled (76/644 = 11.8%):** ARI/NMI on 568-patient subset only.

5. **pycombat mean_only=True for 10 singleton TSS batches:** Mean shift estimates for n=1 batches are noisy. Correct fallback but a limitation vs. balanced batch sizes.

6. **Single PAM50 label source (cBioPortal pan-cancer atlas):** Differs slightly from TCGA marker-paper and METABRIC labels.

7. **No tumour purity correction:** CNV and methylation confounded by cellularity. TCGA ABSOLUTE estimates available but not integrated. Likely reviewer question.

8. **Duplicate aliquot strategy:** Lexicographically last aliquot retained; no read-depth-based selection.

---

## Remaining Work

| Script | Purpose | Depends on |
|---|---|---|
| `06_clustering.py` | Spectral clustering on fused embedding; eigengap + silhouette k=2..6; ARI/NMI vs PAM50; KM + log-rank; 10-seed stability (mean ARI ≥ 0.80 criterion) | Step 05 ✅ |
| `07_baselines.py` | Single-omics (RNA/CNV/Meth), early concat PCA, late C1 (risk avg), late C2 (Ridge Cox concat) | Step 05 ✅ |
| `08_survival_models.py` | Cox PH on fused embedding; covariate-adjusted Cox (age + stage); 5-fold CV; bootstrap pairwise C-index (delta ≥ 0.03 criterion); optional RSF | Steps 06 + 07 |
| `09_sensitivity.py` | K ∈ {10,15,20,25}; features ∈ {2k,5k,10k}; metric ∈ {Pearson, cosine} | Step 05 ✅ |
| `10_figures.py` | UMAP, KM curves, eigengap plot, sensitivity heatmap, batch PCA (saved) | Steps 06–09 |

**Pre-specified success criteria (locked before results examined):**
- Fused graph: log-rank p < 0.05 vs all baselines
- Delta C-index ≥ 0.03 over best single-omics baseline, 95% bootstrap CI excluding zero
- C-index holds after covariate adjustment (age + stage)
- ARI/NMI vs PAM50 competitive or superior to early integration
- Clustering stability: mean cross-seed ARI ≥ 0.80
