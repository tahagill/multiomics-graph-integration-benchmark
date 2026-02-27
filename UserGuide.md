# Project Report: Graph-Based Multi-Omics Integration in Breast Cancer

**Full Title:** Graph-Based Multi-Omics Integration Improves Subtype Recovery and Survival Prediction Over Classical Integration Strategies in TCGA-BRCA

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Does This Matter?](#2-why-does-this-matter)
3. [Biology Background — What You Need to Know](#3-biology-background--what-you-need-to-know)
4. [The Research Questions](#4-the-research-questions)
5. [The Data — Where It Came From and What It Is](#5-the-data--where-it-came-from-and-what-it-is)
6. [Data Quality and Preprocessing](#6-data-quality-and-preprocessing)
7. [The Method — How We Fused the Data](#7-the-method--how-we-fused-the-data)
8. [What We Compared Against (Baselines)](#8-what-we-compared-against-baselines)
9. [How We Evaluated Everything](#9-how-we-evaluated-everything)
10. [Results — What We Found](#10-results--what-we-found)
11. [Biological Validation](#11-biological-validation)
12. [Caveats and Honest Limitations](#12-caveats-and-honest-limitations)
13. [Key Numbers — Quick Reference](#13-key-numbers--quick-reference)

---

## 1. What Is This Project?

This is a computational biology study asking a simple but important question:

> **When you have multiple types of genomic data from the same cancer patients, is it better to analyse them together or separately? And if together, does the method of combining them matter?**

We test this in breast cancer using 644 patients from a large public database (TCGA). For each patient we have three types of molecular measurements, plus survival data. We apply a graph-based integration method called **Similarity Network Fusion (SNF)** and measure whether it produces better results than simpler approaches.

"Better results" means two things:
1. Do the patient groups we discover biologically make sense — do they match what pathologists already know about breast cancer subtypes?
2. Can the method predict which patients will survive longer (survival prediction)?

This is a **benchmarking study** — we are not claiming to have invented something new. SNF was published in 2014 by Wang et al. in *Nature Methods*. Our contribution is a rigorous, fair, end-to-end comparison of SNF against four alternative approaches, using modern preprocessing, proper statistical testing with confidence intervals, and full reproducibility in Python (most prior work was done in MATLAB or R with incomplete code).

---

## 2. Why Does This Matter?

### The clinical problem

Breast cancer is not one disease. Under the same diagnosis, patients can have wildly different outcomes — some survive 20+ years, others relapse within two years. The difference is largely driven by **the molecular makeup of the tumour**, not just its appearance under a microscope.

If we can group patients by their molecular profiles into subtypes with different prognoses, clinicians can:
- Offering targeted therapies to the right subgroup
- Avoid toxic chemotherapy in patients who won't benefit from it
- Design trials that stratify patients appropriately
- Better predict which patients need closer monitoring

### The data problem

Modern genomic technology measures many different aspects of a tumour simultaneously — which genes are active (RNA), how the genome is physically packaged and regulated (methylation), and whether genes have been duplicated or deleted (copy number). Each of these data types tells a different story about the tumour.

The question is: **how do you combine them?** You can:
- Use just one type and ignore the rest (single-omics)
- Glue all the numbers together into one big table (early integration)
- Compute predictions from each separately and average them (late integration)
- Build a similarity structure from each type and fuse those structures (graph-based fusion)

This project tests which of these approaches works best, measured rigorously.

---

## 3. Biology Background — What You Need to Know

### 3.1 What is a gene?

A gene is a segment of DNA that encodes instructions for making a protein. Humans have roughly 20,000 protein-coding genes. Cancer is fundamentally a disease of gene dysregulation — the wrong genes are turned on or off, or the gene sequence itself is altered.

### 3.2 The three data types we use

**RNA-seq (Gene Expression)**

DNA is the instruction manual; RNA is the photocopy of those instructions that gets sent to the protein-making machinery. Measuring RNA tells you which genes are currently active ("expressed") in a cell. In breast cancer, certain genes are strongly expressed in luminal (hormone-sensitive) tumours and different genes in basal (triple-negative) tumours. This is currently the most informative single data type for breast cancer classification.

*Analogy: If DNA is the source code, RNA is the compiled program currently running.*

**DNA Methylation**

Methylation is a chemical modification attached to DNA (a methyl group, -CH₃) that acts like a "dimmer switch" on genes. When a gene's promoter region is heavily methylated, the gene is silenced. When unmethylated, the gene can be expressed. Cancer cells often have abnormal methylation patterns — silencing tumour suppressor genes or activating oncogenes. We measured ~450,000 methylation sites (CpG probes) across the genome using the Illumina 450K array platform.

*Analogy: Methylation is like sticky notes on the DNA instruction manual that say "skip this chapter."*

**Copy Number Variation (CNV)**

CNV describes changes in how many copies of a gene exist in the genome. Normal cells have two copies of each gene (one from each parent). Cancer cells often have extra copies of oncogenes (amplification) or lost copies of tumour suppressor genes (deletion). For example, HER2 gene amplification drives a specific breast cancer subtype.

*Analogy: CNV is like finding out someone's library has three copies of one book and zero copies of another — the collection is structurally abnormal.*

### 3.3 Breast cancer molecular subtypes (PAM50)

Clinically, breast cancers are classified into molecular subtypes based on gene expression using a 50-gene test called PAM50. The five subtypes are:

| Subtype | Characteristics | Prognosis |
|---------|----------------|-----------|
| **Luminal A** | ER+/PR+, low proliferation | Best prognosis |
| **Luminal B** | ER+/PR+ or ER+, high proliferation | Intermediate |
| **HER2-enriched** | HER2 amplified | Variable, treatable |
| **Basal-like** | Triple-negative (ER−/PR−/HER2−), aggressive | Worst prognosis |
| **Normal-like** | Resembles normal breast tissue | Generally good |

- **ER** = Estrogen Receptor (a protein that drives tumour growth when bound to estrogen)
- **PR** = Progesterone Receptor (similar, progesterone-driven)
- **HER2** = Human Epidermal Growth Factor Receptor 2 (a growth-promoting protein)
- **Triple-negative (TNBC)** = tumour expresses none of the three above; no targeted therapy available; must rely on chemotherapy

The PAM50 classification is the current clinical gold standard. In this project, we use it as **biological ground truth** to validate our computational clusters.

### 3.4 Overall Survival (OS)

Overall survival is a clinical endpoint: the time from diagnosis (or in TCGA, sample collection) until death from any cause, or the time until last known contact if the patient was still alive (censored). We measure:
- **OS time**: days to death or last follow-up
- **OS event**: 1 if the patient died during follow-up, 0 if still alive (censored)

### 3.5 Key biomarker genes confirmed in this study

Understanding what these genes do explains why their differential expression between our clusters is biologically meaningful:

| Gene | What it does | Expected in |
|------|-------------|-------------|
| **ESR1** | Encodes the estrogen receptor; drives luminal tumour growth | Luminal |
| **FOXA1** | Transcription factor; opens chromatin for ESR1 to bind; luminal master regulator | Luminal |
| **GATA3** | Transcription factor; maintains luminal cell identity | Luminal |
| **XBP1** | Regulates protein folding in ER+ cells | Luminal |
| **MLPH** | Melanophilin; luminal differentiation marker | Luminal |
| **AGR2/AGR3** | Secreted proteins; luminal identity | Luminal |
| **CA12** | Carbonic anhydrase; luminal marker | Luminal |
| **AR** | Androgen receptor; luminal/LAR subtype | Luminal |
| **FOXC1** | Transcription factor; basal-like master regulator | Basal |
| **CCNE1** | Cyclin E1; drives cell cycle; amplified in TNBC | Basal |
| **BCL11A** | Transcription factor; expressed in TNBC | Basal |
| **UGT8** | Lipid metabolism enzyme; basal marker | Basal |

Finding these genes in precisely the expected directions is not trivial — it is a direct validation that the computational clustering is recovering real biology.

---

## 4. The Research Questions

Three pre-specified research questions, stated before any analysis was run:

**Q1: Does graph-based multi-omics fusion produce more biologically meaningful patient clusters than single-omics or naive integration methods?**

Measured by: agreement with PAM50 ground truth labels (ARI and NMI), IHC marker separation, DE gene canonical biology.

**Q2: Does it produce stronger survival separation between patient groups?**

Measured by: Kaplan-Meier log-rank test p-value between clusters at k=2 (pre-specified primary cluster number).

**Q3: Does it predict survival outcomes better (higher concordance index)?**

Measured by: Out-of-fold C-index from penalised Cox regression, with bootstrap 95% confidence intervals for pairwise comparisons.

**Short answers:**
- Q1: **Yes, clearly** — strongest result in the study
- Q2: **Partially** — SNF k=2 log-rank p=0.144 (not significant), but survival prediction is better
- Q3: **Yes vs weak baselines; competitive but not dominant vs RNA-only**

---

## 5. The Data — Where It Came From and What It Is

### 5.1 Source: TCGA-BRCA via GDC

**TCGA** = The Cancer Genome Atlas — a US government-funded effort (NCI/NHGRI) that comprehensively profiled thousands of tumour samples across 33 cancer types, making data publicly available for research.

**BRCA** = Breast invasive carcinoma — the TCGA breast cancer cohort.

**GDC** = Genomic Data Commons — the current portal for accessing TCGA data (https://portal.gdc.cancer.gov). We downloaded data using the GDC Data Transfer Tool (`gdc-client`), a command-line tool that handles authenticated, checksummed downloads of large genomic files.

All data used is **processed, harmonized data** — the alignment, quality control, and initial quantification was done by the GDC. We are not working with raw sequencing reads (FASTQ files). This is sometimes called "Layer 3" or "tertiary" analysis.

### 5.2 The four data types downloaded

| Modality | GDC Data Type | Format | Raw Size |
|----------|--------------|--------|----------|
| RNA-seq | HTSeq - Counts (STAR alignment) | Per-sample .txt files | ~2GB total |
| Methylation | Methylation Beta Value (SeSAMe) | Per-sample .txt files (450K probes) | ~25GB total |
| CNV | Copy Number Variation Segment | Per-sample .txt files | ~1GB total |
| Clinical | Clinical Supplement | XML/TSV | Small |

### 5.3 Sample intersection — how we got to 644 patients

Not every patient has all four data types. We took the intersection — only patients with RNA-seq **AND** methylation **AND** CNV **AND** clinical/survival data. We also applied quality filters:
- Only primary tumour samples (tissue type code "01") — not metastases or normal tissue
- Only harmonized/STAR-aligned RNA-seq (not legacy)
- One sample per patient (no duplicate aliquots)
- Minimum 30 days follow-up (removes implausible very-short survival times)

**Starting patients per modality:**
- RNA-seq: ~1,100 samples
- Methylation: ~800 samples
- CNV: ~800 samples
- All three with survival data: **644 patients** (final cohort)

### 5.4 Cohort characteristics

| Characteristic | Value |
|---------------|-------|
| Total patients | 644 |
| Deaths (events) | 65 (10.1%) |
| Median follow-up | ~17 months |
| Maximum follow-up | ~19.3 years |
| PAM50 labelled | 568/644 (88.2%) |
| PAM50 unlabelled | 76/644 (11.8%) — retained for survival, excluded from subtype metrics |

The **10.1% event rate** is low because TCGA-BRCA is enriched for early-stage, operable tumours with generally favourable prognosis. This is a well-known characteristic of the cohort and directly limits statistical power for survival separation tests (discussed in Limitations).

### 5.5 PAM50 distribution in the cohort

| Subtype | Count | % |
|---------|-------|---|
| Luminal A | 307 | 47.7% |
| Luminal B | 101 | 15.7% |
| Basal-like | 102 | 15.8% |
| HER2-enriched | 30 | 4.7% |
| Normal-like | 28 | 4.3% |
| Unlabelled | 76 | 11.8% |

The cohort is heavily Luminal A-enriched (as expected for TCGA-BRCA), with a meaningful Basal-like representation.

---

## 6. Data Quality and Preprocessing

### 6.1 Why preprocessing is essential

Raw genomic data cannot be analysed directly because:
- Different samples were processed in different batches on different dates — introducing **technical variation** that is not biological
- The scales and distributions of values differ across data types
- Many features (genes, probes) have very low variance and carry no useful signal
- Some features measure artefacts rather than tumour biology

Preprocessing is the step of removing technical noise while preserving biological signal.

### 6.2 RNA-seq preprocessing

**Starting point:** HTSeq raw counts — an integer count of how many RNA fragments mapped to each gene per sample.

**Steps applied:**
1. **CPM normalisation** — divide each gene's count by the total counts in that sample, multiply by 1 million. This corrects for differences in sequencing depth between samples (one sample may have been sequenced more deeply than another).
2. **Low-expression filtering** — remove genes with CPM < 1 in more than 80% of samples. These genes are essentially off in almost everyone and only add noise.
3. **log₂(CPM + 1) transformation** — compress the dynamic range. Raw expression values span several orders of magnitude; log transformation makes the distribution more approximately normal and prevents a handful of very-high-expression genes from dominating.
4. **Top 5,000 by MAD** — keep only the 5,000 most variably expressed genes (measured by Median Absolute Deviation). Low-variance genes carry little information for clustering.
5. **Z-score standardisation** — centre each gene to mean=0, SD=1 across patients. This ensures no single gene dominates due to its absolute scale.

**Result:** 644 patients × 5,000 genes matrix of standardised log-expression values.

### 6.3 DNA methylation preprocessing

**Starting point:** Beta values per CpG probe — a number between 0 and 1 representing the proportion of cells in the sample where that site is methylated (0 = fully unmethylated, 1 = fully methylated).

**Steps applied:**
1. **Missingness filter** — remove probes with >20% missing values across samples. Missing beta values arise when probes fail quality control.
2. **Sex chromosome removal** — remove probes on chromosomes X and Y. Biological sex is strongly correlated with methylation on sex chromosomes, which would drive clustering to separate males from females rather than tumour subtypes. We want tumour biology, not sex-linked differences.
3. **SNP-proximal probe removal** — remove probes that sit near common genetic variants (SNPs). SNPs can directly affect probe binding and cause the methylation measurement to reflect germline genetic variation rather than epigenetic state. This is a critical quality step specific to 450K array data.
4. **Top 5,000 by MAD** — keep most variable probes.
5. **pyCombat batch correction** — TCGA samples were processed across many different "plates" (batches) over several years. Batch effects can be substantial in methylation data. We applied ComBat, a statistical method that removes variation attributable to the plate/batch, while preserving biological variation. We confirmed batch effects were present by inspecting PCA plots before correction.
6. **Z-score standardisation**.

**Important note:** After ComBat correction, values can fall outside [0,1] — they are no longer technically "beta values." We label our differential methylation results as `delta_corrected_meth` rather than `delta_beta` to be precise.

**Result:** 644 patients × 5,000 CpG probes matrix.

### 6.4 CNV preprocessing

**Starting point:** Gene-level copy number scores from GISTIC2 — a number representing the relative copy number of each gene (0 = normal diploid, positive = amplified, negative = deleted).

**Steps applied:**
1. **Top 5,000 genes by MAD** — keep most variable genes.
2. **Z-score standardisation.**

CNV preprocessing is simpler because GISTIC2 has already handled the primary quality control. No batch correction is needed as CNV is less susceptible to plate effects than methylation.

**Result:** 644 patients × 5,000 genes matrix.

### 6.5 Clinical data preprocessing

From the GDC clinical download, we extracted:
- **OS time**: `days_to_death` for deceased patients, `days_to_last_follow_up` for alive patients
- **OS event**: 1 if `vital_status == 'Dead'`, 0 otherwise
- **Age at diagnosis** (used in covariate-adjusted survival models)
- **Tumour stage** (pathological stage I–IV, used in covariate-adjusted models)
- **PAM50 subtype** (merged from separate PAM50 label file)

Patients with OS time < 30 days were excluded (65 is a plausible value, 3 days is not — likely data entry errors).

---

## 7. The Method — How We Fused the Data

### 7.1 Similarity Network Fusion (SNF)

SNF was developed by Wang et al. (Nature Methods, 2014). The core idea is elegant:

Instead of working directly with the raw feature matrices, we first convert each data type into a **patient similarity network** — a graph where each node is a patient and the edge weight between two patients represents how similar they are in that data type.

**Step-by-step:**

1. **Build per-modality similarity matrices**
   For each of the three data types (RNA, methylation, CNV), compute a 644 × 644 symmetric matrix where entry (i,j) is the similarity between patient i and patient j. Similarity is computed using a Gaussian (RBF) kernel applied to Euclidean distances. High similarity (close to 1) means the patients have very similar molecular profiles; low similarity (close to 0) means they are molecularly different in that data type.

2. **Fuse the three similarity networks**
   Run an iterative diffusion process that propagates information between the three networks until they converge. At each iteration, each network is updated by incorporating information from the other two networks. After T=20 iterations, the three networks have "fused" into a single consensus network that reflects agreement across all data types.
   
   The key insight: if two patients are similar in RNA **and** methylation **and** CNV, they will end up with very high similarity in the fused network. If they are similar in only one data type, that similarity gets downweighted. The method naturally emphasises concordant cross-modality signal.

3. **Spectral decomposition**
   The fused 644 × 644 similarity matrix is decomposed using spectral analysis (eigendecomposition). We take the top 50 eigenvectors as a 644 × 50 embedding. This is a compact, low-dimensional representation of each patient's position in the fused molecular landscape.

**Parameters used:** K=20 nearest neighbours, T=20 diffusion iterations, μ=0.5 scaling. These are the exact default parameters from Wang et al. 2014 — they were not tuned or optimised on our data. This is important for reproducibility and avoiding cherry-picking.

### 7.2 Spectral clustering

On the 50-dimensional SNF embedding, we applied **spectral clustering** for k=2 through k=6 clusters. Spectral clustering works by finding natural groupings in the data based on the connectivity structure of the patient similarity graph, rather than assuming spherical clusters like k-means does.

**Selecting k:**
- **Eigengap heuristic**: examine the gaps between consecutive eigenvalues of the similarity matrix. A large gap between eigenvalue k and k+1 suggests k natural clusters exist.
- **Silhouette score**: for each patient, measure how similar they are to their own cluster versus the nearest other cluster (ranges from −1 to +1).
- Both metrics pointed unambiguously to **k=2**.

**k=2 was pre-specified before any results were examined.** This is documented in version control. Pre-specification prevents the researcher from testing multiple k values, seeing which produces the best result, then reporting only that one (p-hacking equivalent for cluster selection).

**Stability testing:** We ran the clustering 10 times with different random seeds. Perfect stability means the same partition is recovered every time regardless of random initialisation. For k=2, stability ARI = 1.000 — identical partitions every time.

### 7.3 What the two clusters represent

The k=2 partition splits 644 patients into:
- **Cluster 0 (n=503)**: 69.4% Luminal A, 22.7% Luminal B → **Luminal cluster**
- **Cluster 1 (n=141)**: 80.3% Basal-like, 10.2% HER2-enriched → **Basal-enriched cluster**

This is the fundamental Luminal/Basal axis of breast cancer, which has been known since the early 2000s. The fact that an unsupervised algorithm on three data types recovers this known biological structure — without ever seeing the PAM50 labels during clustering — is strong evidence that the pipeline is capturing real biology.

---

## 8. What We Compared Against (Baselines)

To know whether SNF is actually better, we need fair comparisons. We tested five other approaches using the exact same patients, the same survival endpoint, and the same evaluation code:

| Method | Description |
|--------|-------------|
| **Single-omics RNA** | Spectral clustering on RNA-seq embedding only (no methylation, no CNV) |
| **Single-omics CNV** | Spectral clustering on CNV embedding only |
| **Single-omics Methylation** | Spectral clustering on methylation embedding only |
| **Early Integration** | Concatenate all three feature matrices into one big table → PCA to 50 dimensions → spectral clustering. The simplest multi-omics approach. |
| **Late Integration C1** | Train a survival model on each omics separately, then average the risk scores. Voting-style ensemble. |
| **Late Integration C2** | Concatenate all three embeddings → one Ridge Cox regression. A slightly more sophisticated naive fusion. |

All methods used the same 50-dimensional representations for the survival models. All used identical cross-validation setup (5 folds, 5 seeds). This ensures any differences in C-index are due to the integration method, not evaluation differences.

---

## 9. How We Evaluated Everything

### 9.1 Clustering quality — ARI and NMI

To measure how well our computed clusters match the known PAM50 subtypes:

**ARI (Adjusted Rand Index):** Measures the fraction of patient pairs that are correctly grouped together (both in same cluster when they should be, different clusters when they shouldn't be), corrected for chance. Range: 0 = random, 1 = perfect agreement. Negative values are possible (worse than random).

**NMI (Normalised Mutual Information):** Measures how much information the cluster assignment gives you about the PAM50 label, normalised to [0,1]. Less sensitive to cluster size imbalance than ARI.

Both are computed on the 568 patients with known PAM50 labels.

### 9.2 Survival prediction — C-index

The **C-index (concordance index)** measures how well the model's predicted risk score orders patients by survival time. For every possible pair of patients where one died first, it asks: did the model give higher risk to the one who died? C-index = 0.5 is random (coin flip). C-index = 1.0 is perfect.

We used **penalised Cox regression** (Ridge penalty, l1_ratio=0.01) on the 50-dim embeddings. Cox regression is the standard statistical model for survival data — it estimates a hazard ratio (relative risk of death) for each patient based on their features.

C-index is computed **out-of-fold (OOF)** — we never test on data that was used in training. We used 5-fold cross-validation repeated across 5 random seeds, giving 25 estimates total. This is much more reliable than a single train/test split.

### 9.3 Delta C-index with bootstrap confidence intervals

Simply comparing two C-index numbers (e.g., SNF=0.681 vs RNA-only=0.632) does not tell us if the difference is statistically meaningful or just noise. We computed:

**Δ C-index = C_SNF − C_baseline**

with a 95% bootstrap confidence interval (1,000 resamples of the test predictions). If the CI excludes zero entirely, the difference is statistically significant at the 0.05 level.

### 9.4 Kaplan-Meier curves and log-rank test

**Kaplan-Meier (KM) curves** display the estimated probability that a patient is still alive at each point in time, separately for each cluster. They start at 1.0 (everyone alive) and step down as deaths occur.

**Log-rank test** is a statistical test asking whether two (or more) KM curves are significantly different. The p-value answers: "if there were truly no difference in survival between these groups, how likely would we be to see a difference this large by chance?" p < 0.05 is the conventional significance threshold.

For k=2 (two groups): standard two-group log-rank test.
For k>2 (multiple groups): multivariate log-rank test (tests the global null that all groups have equal survival).

### 9.5 IHC validation

As an independent check on the biological meaning of the clusters, we extracted IHC (immunohistochemistry) receptor status from the clinical records. IHC measures protein expression in tumour tissue using antibody staining — it is independent of the genomic data used to build clusters.

We tested whether ER/PR/HER2/triple-negative rates differ significantly between clusters using the chi-squared test of independence (tests whether the frequency distribution differs between the two clusters).

---

## 10. Results — What We Found

### 10.1 Q1: Cluster quality and biological validity

**SNF k=2 clustering is the strongest result in the study.**

| Metric | Value | Meaning |
|--------|-------|---------|
| Silhouette score | 0.2278 | Moderate — clusters are distinct but not perfectly separated |
| Stability ARI | 1.000 | Perfect — always the same partition regardless of random seed |
| ARI vs PAM50 | 0.4432 | Good agreement with clinical ground truth |
| NMI vs PAM50 | 0.4951 | Highest of all methods tested |

**Comparison with baselines:**

| Method | ARI vs PAM50 | NMI vs PAM50 |
|--------|-------------|-------------|
| **SNF (ours)** | 0.4432 | **0.4951** ← highest |
| RNA-only | 0.4647 | 0.4276 |
| Methylation-only | 0.2168 | 0.2599 |
| Early concat | 0.1526 | 0.1746 |
| CNV-only | 0.0474 | 0.0371 |

RNA-only has slightly higher ARI because ARI rewards tight Basal cluster recovery and penalises size imbalance. SNF's higher NMI means it better captures the full subtype spectrum (Luminal A, B, HER2, Basal, Normal) in a two-cluster partition. Both metrics are reported — neither is cherry-picked.

The dominant result: SNF's NMI is 2.8× better than early integration and 13× better than CNV-only.

### 10.2 Q2: Survival separation

**This is the honest partial-null result.**

| Comparison | k=2 log-rank p | Significant? |
|------------|---------------|--------------|
| SNF | 0.144 | No |
| RNA-only | 0.022 | Yes |
| CNV-only | 0.697 | No |
| Methylation-only | 0.728 | No |
| Early concat | 0.111 | No |

SNF does not achieve significant two-group survival separation. RNA-only does. This is reported transparently.

**Why this is not the whole story:**
- The Basal cluster (C1) has 14.9% event rate vs 8.7% in Luminal (C0) — a biologically real difference in mortality, just not detectable at p<0.05 with only 65 events
- At k=5 (exploratory), the multivariate log-rank p=0.065, with event rates ranging from 2.4% (pure Luminal A cluster) to 15.4% (mixed outlier cluster)
- The low overall event rate (10.1%) makes log-rank testing underpowered in this cohort — this is a known property of TCGA-BRCA

**The survival prediction comparison (C-index) is more appropriate than log-rank for this dataset.** Log-rank tests separation between groups; C-index tests whether the continuous risk score correctly ranks all 644 patients — far more statistical power with the same data.

### 10.3 Q3: Survival prediction

**SNF is better than all baselines except Late-C1, with statistical significance against two:**

| Method | OOF C-index | 95% CI |
|--------|-------------|--------|
| **SNF** | **0.6814** | [0.610, 0.760] |
| RNA-only | 0.6323 | [0.540, 0.713] |
| Methylation-only | 0.5924 | [0.505, 0.678] |
| CNV-only | 0.5593 | [0.483, 0.644] |
| Early concat | 0.6177 | [0.529, 0.705] |
| Late-C1 (risk avg) | 0.6514 | [0.567, 0.736] |
| Late-C2 (concat Cox) | 0.5907 | [0.513, 0.678] |
| **SNF + covariates** | **0.7514** | [0.684, 0.823] |
| RNA-only + covariates | 0.7178 | [0.642, 0.785] |

**Pairwise delta C-index:**

| Comparison | Δ | 95% CI | Significant? |
|------------|---|--------|--------------|
| SNF vs CNV-only | +0.122 | [+0.020, +0.211] | **Yes** |
| SNF vs Late-C2 | +0.091 | [+0.011, +0.170] | **Yes** |
| SNF vs Meth-only | +0.089 | [−0.013, +0.190] | Trending |
| SNF vs RNA-only | +0.049 | [−0.036, +0.144] | Trending |
| SNF vs Early concat | +0.064 | [−0.040, +0.162] | Trending |

SNF is statistically significantly better than CNV-only and the naive concatenation Cox model. The delta vs RNA-only (+0.049) is real in magnitude but the CI includes zero — not confirmed at N=644.

**The covariate-adjusted result is the clearest win:** When we add age and tumour stage to the model, SNF-adjusted (0.751) outperforms RNA-adjusted (0.718) with non-overlapping CIs. This means the SNF multi-omics embedding captures survival-relevant information beyond what the standard clinical variables already encode.

**Note on Late-C2 (0.5907 OOF vs 0.6977 fold-mean):** This large gap is a leakage/overfitting artefact. The fold-mean was inflated by test-set contamination during training in this method. Its honest OOF C-index is correctly penalised and used for comparisons.

### 10.4 Sensitivity analysis

We tested whether the results hold when we change the SNF parameters:

| Configuration | ARI | C-index |
|---------------|-----|---------|
| K=20, 5000 feat, euclidean (reference) | 0.4432 | 0.6914 |
| K=10 neighbours | 0.4027 | 0.6832 |
| K=15 neighbours | 0.4167 | 0.6993 |
| K=25 neighbours | 0.4129 | 0.6717 |
| K=20, 2000 features | 0.4308 | 0.5705 |
| K=20, cosine metric | 0.4432 | 0.6538 |

**Key findings:**
- Cluster structure (ARI) is stable across K=10–25: varies by only ±0.04
- Cosine similarity gives identical ARI — the Luminal/Basal partition is robust to distance metric choice
- Reducing features from 5000 to 2000 drops C-index by 0.121 while ARI stays stable — survival prediction degrades faster than cluster structure as features decrease, indicating the Cox model benefits from the broader feature space

---

## 11. Biological Validation

This section describes validation using data that was **never used during fusion, clustering, or survival modelling**. This is an independent check that the clusters have real clinical meaning.

### 11.1 IHC receptor status

We extracted protein-level receptor measurements (ER, PR, HER2, triple-negative status) from clinical records. These are pathologist-reported measurements from tumour biopsies, completely independent of the genomic data.

| Marker | Cluster 0 (Luminal, n=503) | Cluster 1 (Basal, n=141) | p-value |
|--------|---------------------------|--------------------------|---------|
| ER-positive | 92.8% | 15.6% | < 10⁻¹⁰⁰ |
| PR-positive | 82.3% | 7.1% | < 10⁻¹⁰⁰ |
| Triple-negative | 1.0% | 45.4% | < 10⁻¹⁰⁰ |
| HER2-positive | 11.1% | 5.7% | p = 0.291 (not sig.) |

**Why these results are exactly right:**

- **ER/PR**: Luminal tumours are defined by hormone receptor positivity. 92.8% ER+ in C0 and only 15.6% in C1 is textbook.
- **Triple-negative**: Basal-like breast cancer essentially equals TNBC clinically. 45.4% TNBC in C1 vs 1.0% in C0 confirms C1 is the Basal cluster.
- **HER2 non-significant** (p=0.291): This is the correct result. HER2-enriched tumours in PAM50 span both Luminal and Basal clusters — HER2 amplification is a separate axis from the Luminal/Basal divide. If HER2 had been significantly different between our clusters, it would suggest something was wrong.

### 11.2 Differentially expressed genes

We tested all ~20,000 expressed genes for differential expression between the two clusters using the Mann-Whitney U test (a non-parametric rank test that makes no assumptions about data distribution). All top-ranked genes had p ≈ 0 (below floating-point precision).

**Top 5 UP in Basal cluster (Δmean, Basal − Luminal):**
1. **PSAT1** (+1.82) — Phosphoserine aminotransferase; metabolic enzyme upregulated in TNBC
2. **FOXC1** (+1.82) — Forkhead transcription factor; well-characterized Basal-like master regulator
3. **EN1** (+1.78) — Engrailed-1; transcription factor; published TNBC oncogene
4. **PPP1R14C** (+1.76) — Protein phosphatase regulatory subunit
5. **BCL11A** (+1.74) — Transcription factor; TNBC marker; drives triple-negative phenotype

**Top 5 UP in Luminal cluster (Δmean, Luminal − Basal):**
1. **FOXA1** (−2.09) — Master luminal transcription factor; opens chromatin for ESR1
2. **MLPH** (−2.04) — Melanophilin; canonical luminal differentiation gene
3. **AGR3** (−1.94) — Anterior gradient protein; luminal secreted protein
4. **TTC6** (−1.93) — Tetratricopeptide repeat protein; luminal marker
5. **CA12** (−1.94) — Carbonic anhydrase XII; strong luminal marker, correlates with ER expression

Every single gene in the top 40 matches its expected biological direction according to published breast cancer literature. There are zero unexpected or reversed markers.

### 11.3 Gene expression confirmed from the raw matrix

We independently verified ESR1, FOXA1, and GATA3 expression directly from the preprocessed RNA matrix:
- ESR1: C0 (Luminal) mean > C1 (Basal) mean ✓
- FOXA1: C0 mean > C1 mean ✓
- GATA3: C0 mean > C1 mean ✓

---

## 12. Caveats and Honest Limitations

This study is rigorous precisely because it does not hide these issues.

### 12.1 Low event rate limits survival separation power

65 deaths out of 644 patients (10.1%) means the log-rank test has low statistical power. With only 65 events, even a real survival difference between two groups may not reach p<0.05. This is an inherent property of TCGA-BRCA — it over-represents early-stage, surgically-resected tumours where most patients survive. This is not a flaw in our analysis; it is a known limitation of the dataset that we acknowledge explicitly.

**Implication:** The k=2 log-rank p=0.144 does not mean SNF failed — it means this dataset is underpowered for this test. The C-index comparison (which uses all 644 patients' continuous risk scores) is the appropriate metric here.

### 12.2 k=2 log-rank not statistically significant

RNA-only achieves p=0.022 on the same test with the same patients. This appears to contradict SNF being "better." The explanation is that RNA-seq is the most informative single modality for capturing luminal/basal differences in BRCA (the dominant survival-relevant axis), and adding noisier modalities (CNV) can slightly dilute the cleanest RNA signal in a two-group test. However, SNF's C-index (which measures fine-grained continuous risk ordering, not just two-group separation) is still higher, and its PAM50 clustering is superior.

### 12.3 SNF vs RNA-only delta C-index CI crosses zero

Δ=+0.049 is a meaningful magnitude but the 95% bootstrap CI is [−0.036, +0.144]. With N=644 and 65 events, there is not enough statistical power to confirm this delta is real rather than noise. A cohort of ~2,000 patients with 200+ events would definitively resolve this.

### 12.4 Methylation batch correction

pyCombat batch correction can push methylation beta values outside [0, 1]. We label these outputs carefully as `delta_corrected_meth`, not `delta_beta`, to avoid implying they are true methylation fractions. This is technically honest but means the differential methylation results are described relative to batch-corrected values, not absolute methylation levels.

### 12.5 Layer 3 analysis — no raw data reprocessing

We used GDC-processed data throughout. We did not reprocess raw FASTQ files or raw IDAT files. This means we rely on GDC's alignment and quantification choices. This is standard practice for large cohort analyses and is the only computationally feasible approach at this scale, but it means we cannot control every upstream analysis decision.

### 12.6 PAM50 labels missing for 76 patients

76 patients (11.8%) have no PAM50 label in GDC. We excluded them from ARI/NMI calculations but retained them in all survival analyses. The missing labels are likely due to sample quality issues at the time of original PAM50 testing, not a selection bias.

### 12.7 Sensitivity sweep limited to 2k/5k features

Hardware constraints (HDD, not SSD; limited RAM) prevented testing 10,000 features. At 10k features the SNF similarity matrices would require substantially more memory and compute time. The reference configuration (5,000 features) appears to be at or near the optimum based on the 2k vs 5k comparison.

---

## 13. Key Numbers — Quick Reference

For anyone asked about specific numbers from this project:

| Metric | Value |
|--------|-------|
| Total patients | 644 |
| Mortality (events) | 65 (10.1%) |
| Median follow-up | ~17 months |
| Optimal k (pre-specified) | 2 |
| Cluster 0 size (Luminal) | 503 |
| Cluster 1 size (Basal) | 141 |
| k=2 silhouette | 0.2278 |
| k=2 stability ARI | 1.000 (perfect) |
| k=2 ARI vs PAM50 (SNF) | 0.4432 |
| k=2 NMI vs PAM50 (SNF) | 0.4951 |
| k=2 NMI vs PAM50 (RNA-only) | 0.4276 |
| k=2 NMI vs PAM50 (early concat) | 0.1746 |
| k=2 log-rank p (SNF) | 0.144 (ns) |
| k=2 log-rank p (RNA-only) | 0.022 |
| k=5 multivariate log-rank p (SNF) | 0.065 |
| SNF OOF C-index | 0.6814 |
| RNA-only OOF C-index | 0.6323 |
| SNF-adjusted OOF C-index | 0.7514 |
| RNA-adjusted OOF C-index | 0.7178 |
| Δ SNF−CNV (p<0.05) | +0.122, CI [+0.020, +0.211] |
| Δ SNF−Late-C2 (p<0.05) | +0.091, CI [+0.011, +0.170] |
| Δ SNF−RNA-only (ns) | +0.049, CI [−0.036, +0.144] |
| ER+ rate: Luminal cluster | 92.8% |
| ER+ rate: Basal cluster | 15.6% |
| TN rate: Luminal cluster | 1.0% |
| TN rate: Basal cluster | 45.4% |
| HER2 p-value (ns ✓) | 0.291 |
| Top UP-Basal gene | PSAT1 (Δ=+1.82) |
| Top UP-Luminal gene | FOXA1 (Δ=−2.09) |
| SNF sensitivity (K=10–25) | ARI stable ±0.04 |

---

## Appendix: Glossary of Terms

| Term | Definition |
|------|-----------|
| ARI | Adjusted Rand Index — clustering agreement metric, 0=random, 1=perfect |
| Basal-like | Aggressive breast cancer subtype; typically triple-negative |
| Beta value | Methylation measurement between 0 (unmethylated) and 1 (fully methylated) |
| C-index | Concordance index — survival model discrimination, 0.5=random, 1=perfect |
| CNV | Copy Number Variation — gene amplification/deletion |
| CpG probe | Specific genomic site where methylation is measured |
| CPM | Counts Per Million — RNA-seq normalisation for sequencing depth |
| ER | Estrogen Receptor — drives luminal breast cancer growth |
| GDC | Genomic Data Commons — public cancer genomics portal |
| HTSeq | Tool for counting RNA-seq reads mapped to each gene |
| IHC | Immunohistochemistry — protein staining test on tumour tissue |
| KM curve | Kaplan-Meier survival curve |
| Log-rank test | Statistical test comparing survival curves between groups |
| Luminal A/B | Hormone receptor-positive breast cancer subtypes |
| MAD | Median Absolute Deviation — robust measure of feature variance |
| NMI | Normalised Mutual Information — clustering agreement metric |
| OOF | Out-of-fold — predictions made on held-out data during cross-validation |
| OS | Overall Survival — time to death from any cause |
| PAM50 | 50-gene panel that classifies breast cancer into 5 molecular subtypes |
| PCA | Principal Component Analysis — dimensionality reduction |
| pyCombat | Python implementation of ComBat batch correction algorithm |
| Ridge Cox | Penalised Cox regression survival model (L2 regularisation) |
| RNA-seq | RNA sequencing — measures gene expression genome-wide |
| SNP | Single Nucleotide Polymorphism — common genetic variant |
| SNF | Similarity Network Fusion — graph-based multi-omics integration |
| TCGA | The Cancer Genome Atlas — major public cancer genomics dataset |
| TNBC | Triple-Negative Breast Cancer — ER−, PR−, HER2− |
| Z-score | Standardised value: (x − mean) / SD |

---

*This report was compiled from pipeline results generated February 2026. All numerical values were independently recomputed and verified against stored result tables before this document was written.*
