# Graph-Based Multi-Omics Integration Improves Subtype Recovery and Survival Prediction Over Classical Integration Strategies in TCGA-BRCA

**Taha Ahmad** — Middle East Technical University

> Full PDF: [`snf-multiOmics-benchmark.pdf`](snf-multiOmics-benchmark.pdf)
> In-Depth explanation of the papers biology and logic : [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
---

## Overview

This repository contains the complete, reproducible analysis pipeline for a benchmarking study of Similarity Network Fusion (SNF) against classical multi-omics integration strategies in the TCGA-BRCA breast cancer cohort.

**Research questions:**
1. Does graph-based fusion recover PAM50 molecular subtypes better than single-omic or concatenation-based clustering?
2. Does fusing three modalities improve survival prediction over the best single modality, and is any advantage statistically distinguishable from zero?
3. How sensitive are the results to SNF hyperparameters?

**Key findings:**

| Method | NMI vs PAM50 | OOF C-index |
|--------|-------------|-------------|
| **SNF (proposed)** | **0.495** | **0.681** |
| RNA-only | 0.428 | 0.632 |
| Methylation-only | 0.260 | 0.592 |
| Early concatenation (PCA) | 0.175 | 0.618 |
| CNV-only | 0.037 | 0.559 |

SNF significantly outperforms CNV-only in survival prediction (ΔC = +0.122, 95% CI 0.020–0.211). The advantage over RNA-only (ΔC = +0.049) is positive but the CI crosses zero — reported transparently. IHC validation (ER, PR, triple-negative) confirmed cluster biology independently (all p < 10⁻¹⁰⁰).

---

## Data

**Source:** TCGA-BRCA via NCI Genomic Data Commons (GDC), downloaded with `gdc-client`.  
**Cohort:** 644 patients with matched RNA-seq, 450k DNA methylation, and GISTIC2 copy number profiles, plus overall survival data.

| Modality | Processing | Features |
|----------|-----------|----------|
| RNA-seq | HTSeq counts → CPM → log₂(CPM+1) → z-score | 5,000 genes (top MAD) |
| Methylation | β-values → probe QC → ComBat → z-score | 5,000 CpG probes (top MAD) |
| CNV | GISTIC2 segment values → z-score | 5,000 segments (top MAD) |

PAM50 labels (568/644 patients) from TCGA PanCanAtlas. Survival endpoint: overall survival (OS time + event indicator).

---

## Setup

```bash
git clone https://github.com/tahagill/multiomics-graph-integration-benchmark
cd multiomics-graph-integration-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Python 3.13.** Key dependencies: `snfpy`, `scikit-learn`, `scikit-survival`, `lifelines`, `umap-learn`, `statsmodels`, `neurocombat-sklearn`.

---

## Pipeline

Run scripts in order. Each step is checkpointed — interruptions do not require a full re-run.

| Script | Description |
|--------|-------------|
| `01_query_manifests.py` | Query GDC API, write download manifests to `metadata/` |
| `02_download_data.sh` | Download raw files via `gdc-client` |
| `02b_fetch_pam50.py` | Extract PAM50 labels from clinical download |
| `03_build_matrices.py` | Aggregate per-sample files into cohort-wide matrices; intersect to 644-patient cohort |
| `04_preprocess.py` | Normalise, filter, batch-correct, z-score each modality |
| `04b_batch_check.py` | PCA visualisation of methylation batch structure pre/post ComBat |
| `05_graph_construction.py` | Build per-modality affinity matrices; fuse with SNF (K=20, T=20, μ=0.5) |
| `06_clustering.py` | Spectral clustering k=2–6; eigengap/silhouette k-selection; stability ARI; PAM50 NMI/ARI |
| `07_baselines.py` | Single-omics and early-concatenation baselines |
| `08_survival_models.py` | Ridge Cox OOF C-index; bootstrap CIs; pairwise delta comparisons; covariate-adjusted models |
| `09_sensitivity.py` | One-at-a-time sweep: K ∈ {10,15,20,25}, features ∈ {2000,5000}, metric ∈ {euclidean, cosine} |
| `10_figures.py` | Generate all main paper figures (fig1–fig4) |
| `11_clinical_validation.py` | IHC validation, differential expression, differential methylation/CNV |
| `12_missing_figures.py` | Supplementary figure generation |

---

## Repository Structure

```
├── scripts/              # Pipeline scripts 01–12
├── manuscript/
│   ├── main.tex          # LaTeX source
│   └── references.bib    # Bibliography
├── metadata/             # GDC manifests, PAM50 labels, clinical TSVs
├── data/
│   ├── raw/              # GDC downloads (not tracked — re-downloadable)
│   └── processed/        # Affinity matrices, fused network, cohort list
├── results/
│   ├── figures/          # All paper figures (PDF)
│   └── tables/           # All result tables (TSV)
├── requirements.txt
└── snf-multiOmics-benchmark.pdf   # Final manuscript PDF
```

---

## Results Summary

### Clustering

The SNF k=2 partition was pre-specified on eigengap and silhouette criteria before any outcome data were examined. It is perfectly stable (stability ARI = 1.00 across 100 sub-sampled replicates) and separates 503 Luminal-enriched patients from 141 Basal-enriched patients.

### IHC Validation

| Marker | Cluster 0 (Luminal) | Cluster 1 (Basal) | p |
|--------|--------------------|--------------------|---|
| ER-positive | 92.8% | 15.6% | < 10⁻¹⁰⁰ |
| PR-positive | 82.3% | 7.1% | < 10⁻¹⁰⁰ |
| Triple-negative | 1.0% | 45.4% | < 10⁻¹⁰⁰ |
| HER2-positive | 11.1% | 5.7% | 0.291 (ns) |

### Survival

SNF OOF C-index: **0.681** (95% CI 0.610–0.760).  
Significantly better than CNV-only (ΔC = +0.122, CI 0.020–0.211) and Late-C2 (ΔC = +0.091, CI 0.011–0.170).  
Advantage over RNA-only (ΔC = +0.049, CI −0.036–0.144) is positive but not conclusive at this cohort size and event count (65 events / 644 patients).

### Sensitivity

NMI varies by ≤ 0.013 across K = 10–25. Results are not inflated by parameter selection — the primary configuration is neither the maximum nor minimum on any metric.

---

## Citation

If you use this code or findings, please cite the preprint:

```
Ahmad, T. (2026). Graph-Based Multi-Omics Integration Improves Subtype Recovery and
Survival Prediction Over Classical Integration Strategies in TCGA-BRCA.
bioRxiv. https://doi.org/10.1101/BIORXIV/2026/708481
```

---

## License

Code: [MIT](LICENSE)  
Data: TCGA data are subject to the [NIH GDC Data Access Policy](https://gdc.cancer.gov/access-data/data-access-policies).
