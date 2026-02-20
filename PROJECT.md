

---

# PROJECT BLUEPRINT

## Title

**Graph-Based Multi-Omics Integration for Robust Subtype Discovery and Survival Stratification in TCGA-BRCA**

---

# 1. PROJECT OVERVIEW

## Objective

To evaluate whether graph-based integration of multi-omics data improves cancer subtype discovery and survival prediction compared to classical early and late integration strategies.

## Core Research Question

Does similarity-graph fusion of RNA-seq, DNA methylation, and CNV data produce:

1. More biologically meaningful patient clusters?
2. Stronger survival separation?
3. Better survival prediction performance?

Compared to:

* Single-omics models
* Early feature concatenation
* Late model ensembling

---

# 2. DATA SOURCE

Primary dataset:

* TCGA-BRCA (Breast Invasive Carcinoma)
* Source: Genomic Data Commons (GDC)
* Data type: Harmonized processed data (NOT raw FASTQ)

Data modalities:

1. RNA-seq (HTSeq raw counts)
2. DNA methylation (Illumina 450K beta values)
3. Copy number variation (gene-level copy number scores)
4. Clinical metadata (survival)

We do NOT:

* Download FASTQ
* Perform alignment
* Use STAR, FastQC, cutadapt
* Rebuild a sequencing pipeline

This is Layer 3 modeling, not raw processing.

---

# 3. PROJECT SCOPE

## Included

* Multi-omics preprocessing
* Feature filtering
* Graph construction per modality
* Similarity Network Fusion (or equivalent graph integration)
* Spectral clustering
* Survival analysis (Kaplan-Meier + log-rank)
* Cox proportional hazards modeling
* Random survival forest (optional)
* Cross-validation
* Statistical benchmarking
* Reproducible pipeline
* arXiv-ready manuscript

## Excluded

* Deep learning (no GNNs)
* Autoencoders
* FASTQ processing
* Pan-cancer extension (single cancer only)
* Wet lab validation

---

# 4. TECHNICAL ENVIRONMENT

OS: Linux
Execution style: bash-driven + Python scripts
Project structure:

```
brca_multiomics_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── metadata/
│
├── scripts/
│
├── notebooks/
│
├── results/
│
├── logs/
│
├── environment.yml
├── README.md
```

All steps reproducible via bash.

---

# 5. DATA ACQUISITION PIPELINE

Tool:

* gdc-client CLI
* GDC API JSON queries

Steps:

1. Query RNA-seq (HTSeq - Counts)
2. Query Gene-level CNV
3. Query DNA Methylation 450K
4. Query clinical data

Filtering rules:

* Keep only TCGA-BRCA
* Keep only Primary Tumor samples (sample code "01")
* Use harmonized data only
* Remove duplicate aliquots if present

Output:

* Gene expression matrix (genes × samples)
* CNV matrix (genes × samples)
* Methylation matrix (CpGs × samples)
* Clinical table (samples × survival variables)

---

# 6. DATA PREPROCESSING

## RNA-seq

* Remove low-count genes
* Filter by variance (top 5–10k genes)
* Log2(count + 1)
* Z-score normalization

## CNV

* Gene-level scores
* Variance filtering
* Standardization

## Methylation

* Remove CpGs with high missingness
* Variance filtering
* Optional gene-level aggregation
* Standardization

## Sample Matching

Intersect sample IDs across all modalities and clinical data.

Final cohort expected:
~600–800 patients.

---

# 7. BASELINE METHODS

These must be implemented for benchmarking.

## A. Single-Omics Models

For each modality:

* Spectral clustering
* Survival analysis

## B. Early Integration

* Concatenate standardized features
* PCA (dimensionality reduction)
* Clustering
* Survival evaluation

## C. Late Integration

* Build survival model per modality
* Ensemble predictions (averaging risk scores)

---

# 8. PROPOSED METHOD

## Graph Construction (Per Modality)

For each modality:

1. Compute similarity matrix (Pearson or cosine)
2. Build k-nearest neighbor graph
3. Construct adjacency matrix

## Graph Integration

Use Similarity Network Fusion (SNF-style diffusion)

Iteratively update similarity matrices until convergence.

Output:
Fused similarity matrix.

## Embedding

Apply spectral embedding to fused matrix.

Low-dimensional representation (e.g., 50 dims).

---

# 9. SUBTYPE DISCOVERY

From fused embedding:

* Spectral clustering
* Choose k via eigengap or silhouette

Validation:

* Kaplan-Meier survival curves
* Log-rank test
* Clinical feature enrichment

Goal:
Clusters show stronger survival separation than baselines.

---

# 10. SURVIVAL MODELING

Using fused embeddings:

* Cox proportional hazards model
* Evaluate Concordance Index (C-index)
* 5-fold cross-validation
* Confidence intervals

Optional:
Random survival forest.

Compare against:

* RNA-only
* Early integration
* Late integration

---

# 11. STATISTICAL RIGOR

Required:

* Multiple random seeds
* Cross-validation
* Confidence intervals
* Statistical comparison between models
* Report effect sizes

No single-run claims.

---

# 12. DELIVERABLES

## 1. Codebase

* Modular Python scripts
* Reproducible pipeline
* Config-driven parameters

## 2. Results

* Survival plots
* Benchmark comparison tables
* Cluster visualizations (UMAP)
* Performance metrics

## 3. Documentation

* Detailed README
* Methods description
* Reproducibility instructions

## 4. Preprint

Target:

* arXiv q-bio / stat.ML

Paper sections:

1. Introduction
2. Related Work
3. Methods
4. Experimental Setup
5. Results
6. Discussion
7. Limitations
8. Conclusion

---

# 13. SUCCESS CRITERIA

Project is successful if:

* Fused graph improves survival separation vs baseline
* C-index improves meaningfully
* Pipeline is reproducible
* Results statistically validated
* Manuscript arXiv-ready

---

# 14. RISKS

* Low modality overlap
* Overfitting
* High dimensionality noise
* Poor preprocessing decisions

Mitigation:
Variance filtering + cross-validation + clean sample intersection.

---

# 15. STRATEGIC VALUE

Signals:

* Graph modeling competence
* Statistical maturity
* Systems biology understanding
* Reproducible research practices

Compensates for low GPA through demonstrated research depth.

---

# 16. CURRENT STAGE

We are currently at:

DATA ACQUISITION PHASE

Using:

* Linux
* Bash
* gdc-client
* Python-based matrix building

Goal:
Download and build unified matrices for RNA, CNV, methylation, and clinical data.

---

This is the full A–Z blueprint.

If you paste this into another AI session, it will:

* Know the scope
* Avoid suggesting FASTQ processing
* Avoid deep learning
* Understand benchmarking requirements
* Continue from data acquisition

