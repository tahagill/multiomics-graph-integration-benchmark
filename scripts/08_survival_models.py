"""
08_survival_models.py
---------------------
Survival prediction evaluation via Ridge-penalised Cox PH models.
Compares SNF fused spectral embedding against all baseline embeddings.

Methods evaluated
─────────────────
  SNF          : 644×50 fused spectral embedding from Step 05
  RNA-only     : 644×50 RNA spectral embedding from Step 07
  CNV-only     : 644×50 CNV spectral embedding from Step 07
  Meth-only    : 644×50 Methylation spectral embedding from Step 07
  Early-concat : 644×50 PCA embedding of concatenated 15k features (Step 07)
  Late-C1      : Risk-score averaging — independent Cox per modality,
                 average predicted risk, evaluate ensemble C-index
  Late-C2      : Ridge Cox on concatenated 150-dim (rna+cnv+meth embeddings)
  SNF-adj      : SNF embedding + age_at_diagnosis + tumor_stage (covariate-adjusted)
  RNA-adj      : RNA embedding + age_at_diagnosis + tumor_stage (covariate-adjusted)

All models use Ridge-penalised Cox PH (l1_ratio=0.0, CoxnetSurvivalAnalysis).
Alpha is selected per outer fold via inner 3-fold CV.

Evaluation protocol
───────────────────
  • 5-fold cross-validation × 5 random seeds → fold-level C-index → mean ± SD
  • Primary OOF (seed=42): out-of-fold predictions for all 644 patients
  • Bootstrap CI: 1000 resamples of OOF (patient-level) → 95% CI of C-index
  • Delta C-index (SNF vs each baseline): paired bootstrap, 95% CI,
    criterion: 95% CI excludes zero (supports the δ ≥ 0.03 success criterion)

Clinical covariate adjustment
──────────────────────────────
  Covariates: age_at_diagnosis (years, standardised within fold)
              tumor_stage_numeric (1–4; median-imputed within fold for NaN)
  Comparison: unadjusted vs adjusted C-index for SNF and RNA-only.
  Purpose: show SNF captures signal beyond age/stage alone (closes reviewer
           objection "are you just recovering tumour-stage information?").

Outputs
───────
  results/tables/
    survival_metrics.tsv         per-method C-index (mean±SD, OOF, 95% CI)
    delta_cindex.tsv             pairwise delta C-index vs each baseline
    covariate_adjusted_cindex.tsv  unadjusted vs adjusted C-index table

  results/figures/survival/
    cindex_comparison.pdf        forest plot of C-indices with 95% CI
    delta_cindex_vs_snf.pdf      delta C-index (SNF − baseline) forest plot
    adjusted_cindex.pdf          unadjusted vs adjusted bar chart

Run from project root:
  python scripts/08_survival_models.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
N_SPLITS    = 5
N_SEEDS     = 5
PRIMARY_SEED = 42   # seed for OOF patient-ordered predictions
N_BOOT      = 1000
DELTA_TARGET = 0.03  # README success criterion

ALPHAS_GRID  = [0.01, 0.1, 1.0, 10.0, 100.0]   # inner CV alpha options
L1_RATIO     = 0.01   # near-pure Ridge (l1_ratio must be > 0 for sksurv)

PROCESSED   = Path("data/processed")
FIG_OUT     = Path("results/figures/survival")
TABLE_OUT   = Path("results/tables")
FIG_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR     = Path("logs")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "08_survival_models.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Survival structured array helper
# ──────────────────────────────────────────────────────────────────────────────

def make_surv_array(events: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Build sksurv structured array from event/time arrays."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ridge Cox per fold (with inner alpha selection)
# ──────────────────────────────────────────────────────────────────────────────

def fit_ridge_cox_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    alphas_grid: list = ALPHAS_GRID,
    inner_seed: int = 99,
) -> tuple[np.ndarray, float]:
    """
    Fit Ridge Cox on (X_tr, y_tr) with alpha chosen by inner 3-fold CV.
    Returns (OOF risk scores on X_te, best_alpha).
    Uses StandardScaler fitted on X_tr (no leakage).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    inner_cv  = KFold(n_splits=3, shuffle=True, random_state=inner_seed)
    best_alpha, best_ci = ALPHAS_GRID[2], -np.inf   # default alpha=1.0

    for alpha in alphas_grid:
        ci_inner = []
        for ii_tr, ii_val in inner_cv.split(X_tr_s):
            if y_tr[ii_val]["event"].sum() < 2:
                continue
            try:
                m = CoxnetSurvivalAnalysis(
                    l1_ratio=L1_RATIO, alphas=[alpha], max_iter=300,
                    fit_baseline_model=False,
                )
                m.fit(X_tr_s[ii_tr], y_tr[ii_tr])
                pred = m.predict(X_tr_s[ii_val]).ravel()
                c = concordance_index_censored(
                    y_tr[ii_val]["event"], y_tr[ii_val]["time"], pred
                )[0]
                ci_inner.append(c)
            except Exception:
                pass
        if ci_inner and float(np.mean(ci_inner)) > best_ci:
            best_ci    = float(np.mean(ci_inner))
            best_alpha = alpha

    # Final fit with best alpha on full outer train fold
    model = CoxnetSurvivalAnalysis(
        l1_ratio=L1_RATIO, alphas=[best_alpha], max_iter=300,
        fit_baseline_model=False,
    )
    model.fit(X_tr_s, y_tr)
    risks_te = model.predict(X_te_s).ravel()
    return risks_te, best_alpha


# ──────────────────────────────────────────────────────────────────────────────
# Full CV evaluation for a single feature matrix
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_evaluation(
    X: np.ndarray, y: np.ndarray, method_name: str
) -> dict:
    """
    5-fold CV × N_SEEDS → fold-level C-index + OOF predictions (seed=42 % N_SEEDS).
    Returns dict with fold_cidxs, oof_risks, oof_events, oof_times.
    """
    fold_cidxs = []
    # Allocate OOF arrays (primary seed)
    oof_risks  = np.full(len(X), np.nan)
    oof_events = np.full(len(X), np.nan)
    oof_times  = np.full(len(X), np.nan)
    primary_seed = PRIMARY_SEED % N_SEEDS   # 42 % 5 = 2

    log.info(f"  [{method_name}] CV ({N_SPLITS}-fold × {N_SEEDS} seeds)...")

    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
            y_te = y[te_idx]
            if y_te["event"].sum() < 2:
                log.warning(f"    seed={seed} fold={fold_idx}: "
                            f"<2 events in test, skipping")
                continue
            risks_te, alpha = fit_ridge_cox_fold(
                X[tr_idx], y[tr_idx], X[te_idx]
            )
            c = concordance_index_censored(
                y_te["event"], y_te["time"], risks_te
            )[0]
            fold_cidxs.append(c)

            # Store OOF predictions from primary seed
            if seed == primary_seed:
                oof_risks[te_idx]  = risks_te
                oof_events[te_idx] = y_te["event"].astype(float)
                oof_times[te_idx]  = y_te["time"]

    mean_ci = float(np.mean(fold_cidxs))
    std_ci  = float(np.std(fold_cidxs))
    log.info(f"  [{method_name}] fold mean C-index = {mean_ci:.4f} ± {std_ci:.4f}")

    return {
        "method":      method_name,
        "fold_cidxs":  fold_cidxs,
        "mean_ci":     mean_ci,
        "std_ci":      std_ci,
        "oof_risks":   oof_risks,
        "oof_events":  oof_events,
        "oof_times":   oof_times,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Late C1: risk-score averaging (per-fold, fold-consistent)
# ──────────────────────────────────────────────────────────────────────────────

def run_late_c1(emb_dict: dict, y: np.ndarray) -> dict:
    """
    Late C1: within each fold, fit 3 independent Cox models (RNA, CNV, Meth),
    average predicted risk scores on test fold, compute C-index.
    5-fold × N_SEEDS; OOF from primary seed.
    """
    modalities  = ["rna", "cnv", "meth"]
    fold_cidxs  = []
    oof_risks   = np.full(len(y), np.nan)
    oof_events  = np.full(len(y), np.nan)
    oof_times   = np.full(len(y), np.nan)
    primary_seed = PRIMARY_SEED % N_SEEDS

    log.info("  [Late-C1] CV (risk-score averaging)...")

    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(y)):
            y_te = y[te_idx]
            if y_te["event"].sum() < 2:
                continue

            # Fit one Cox per modality, collect test risks
            te_risks_list = []
            for mod in modalities:
                X_mod = emb_dict[mod]
                risks_te_mod, _ = fit_ridge_cox_fold(
                    X_mod[tr_idx], y[tr_idx], X_mod[te_idx]
                )
                te_risks_list.append(risks_te_mod)

            avg_risks = np.nanmean(te_risks_list, axis=0)
            c = concordance_index_censored(
                y_te["event"], y_te["time"], avg_risks
            )[0]
            fold_cidxs.append(c)

            if seed == primary_seed:
                oof_risks[te_idx]  = avg_risks
                oof_events[te_idx] = y_te["event"].astype(float)
                oof_times[te_idx]  = y_te["time"]

    mean_ci = float(np.mean(fold_cidxs))
    std_ci  = float(np.std(fold_cidxs))
    log.info(f"  [Late-C1] fold mean C-index = {mean_ci:.4f} ± {std_ci:.4f}")

    return {
        "method":     "Late-C1 (risk avg)",
        "fold_cidxs": fold_cidxs,
        "mean_ci":    mean_ci,
        "std_ci":     std_ci,
        "oof_risks":  oof_risks,
        "oof_events": oof_events,
        "oof_times":  oof_times,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_cindex(
    events: np.ndarray, times: np.ndarray, risks: np.ndarray,
    n_boot: int = N_BOOT, rng_seed: int = 0,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for C-index via patient-level resampling.
    Returns (oof_cindex, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(rng_seed)
    # OOF C-index (no bootstrap)
    valid_mask = ~np.isnan(risks)
    ev_v  = events[valid_mask].astype(bool)
    t_v   = times[valid_mask]
    r_v   = risks[valid_mask]
    oof_c = concordance_index_censored(ev_v, t_v, r_v)[0]

    boot_cidxs = []
    n = len(ev_v)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ev_b, t_b, r_b = ev_v[idx], t_v[idx], r_v[idx]
        if ev_b.sum() < 2:
            continue
        try:
            c = concordance_index_censored(ev_b, t_b, r_b)[0]
            boot_cidxs.append(c)
        except Exception:
            pass

    lower, upper = np.percentile(boot_cidxs, [2.5, 97.5])
    return float(oof_c), float(lower), float(upper)


def bootstrap_delta(
    events: np.ndarray, times: np.ndarray,
    risks_a: np.ndarray, risks_b: np.ndarray,
    n_boot: int = N_BOOT, rng_seed: int = 0,
) -> tuple[float, float, float]:
    """
    Paired bootstrap for delta C-index (risks_a − risks_b).
    Both risk arrays must be patient-ordered OOF from the same fold scheme.
    Returns (delta_point, lower_95, upper_95).
    """
    rng  = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(risks_a) | np.isnan(risks_b))
    ev_v  = events[valid].astype(bool)
    t_v   = times[valid]
    ra_v  = risks_a[valid]
    rb_v  = risks_b[valid]

    ca_pt = concordance_index_censored(ev_v, t_v, ra_v)[0]
    cb_pt = concordance_index_censored(ev_v, t_v, rb_v)[0]
    delta_pt = ca_pt - cb_pt

    deltas = []
    n = len(ev_v)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ev_b, t_b, ra_b, rb_b = ev_v[idx], t_v[idx], ra_v[idx], rb_v[idx]
        if ev_b.sum() < 2:
            continue
        try:
            ca = concordance_index_censored(ev_b, t_b, ra_b)[0]
            cb = concordance_index_censored(ev_b, t_b, rb_b)[0]
            deltas.append(ca - cb)
        except Exception:
            pass

    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return float(delta_pt), float(lower), float(upper)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def forest_plot_cindex(rows: list[dict], out_path: Path,
                        title: str = "Survival C-index (5-fold CV)") -> None:
    """
    Forest plot: C-index point estimate (OOF) ± 95% bootstrap CI.
    Vertical dashed line at 0.5 (random), shaded region = SNF result.
    """
    methods  = [r["method"] for r in rows]
    oof_ci   = [r["oof_cindex"] for r in rows]
    lo       = [r["ci_lower"]  for r in rows]
    hi       = [r["ci_upper"]  for r in rows]
    n        = len(rows)

    fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.5))
    y_pos = np.arange(n)[::-1]   # top to bottom

    for i, (y, m, c, l, h) in enumerate(zip(y_pos, methods, oof_ci, lo, hi)):
        colour = "goldenrod" if "SNF" in m and "adj" not in m.lower() else \
                 "steelblue" if "SNF" in m else "dimgrey"
        ax.hlines(y, l, h, lw=2.5, color=colour)
        ax.plot(c, y, "o", color=colour, markersize=7, zorder=5)

    ax.axvline(0.5, ls="--", lw=1, color="grey", alpha=0.7,
               label="Random (C=0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("C-index (OOF, 95% bootstrap CI)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.40, 0.80)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Forest plot → {out_path}")


def delta_forest_plot(rows: list[dict], out_path: Path) -> None:
    """
    Forest plot: delta C-index (SNF − baseline) with 95% bootstrap CI.
    Vertical dashed line at 0 (no difference) and at 0.03 (success criterion).
    """
    comparisons = [r["comparison"] for r in rows]
    deltas  = [r["delta"]   for r in rows]
    lo      = [r["ci_lower"] for r in rows]
    hi      = [r["ci_upper"] for r in rows]
    n       = len(rows)

    fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.5))
    y_pos = np.arange(n)[::-1]

    for y, comp, d, l, h in zip(y_pos, comparisons, deltas, lo, hi):
        excludes_zero = l > 0 or h < 0
        colour = "mediumseagreen" if excludes_zero else "dimgrey"
        ax.hlines(y, l, h, lw=2.5, color=colour)
        ax.plot(d, y, "o", color=colour, markersize=7, zorder=5)

    ax.axvline(0, ls="--", lw=1, color="grey", alpha=0.7, label="No difference")
    ax.axvline(DELTA_TARGET, ls=":", lw=1.5, color="steelblue", alpha=0.8,
               label=f"Success criterion (δ={DELTA_TARGET})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons, fontsize=9)
    ax.set_xlabel("ΔSNF C-index (95% bootstrap CI)", fontsize=10)
    ax.set_title("SNF vs baselines: delta C-index\n"
                 "(green = 95% CI excludes zero)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Delta forest plot → {out_path}")


def adjusted_cindex_chart(rows: list[dict], out_path: Path) -> None:
    """Bar chart comparing unadjusted vs adjusted C-index for SNF and best baseline."""
    methods = [r["method"] for r in rows]
    unadj   = [r["unadj_cindex"] for r in rows]
    adj     = [r["adj_cindex"]   for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - width/2, unadj, width, label="Unadjusted",
           color="steelblue", edgecolor="white")
    ax.bar(x + width/2, adj,   width, label="+ age + stage",
           color="darkorange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("C-index (OOF)")
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Covariate-adjusted Cox\n(embedding vs embedding + age + stage)",
                 fontsize=11)
    ax.axhline(0.5, ls="--", lw=0.8, color="grey", alpha=0.6)
    ax.legend(fontsize=9)
    for xi, (u, a) in zip(x, zip(unadj, adj)):
        ax.text(xi - width/2, u + 0.005, f"{u:.3f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(xi + width/2, a + 0.005, f"{a:.3f}", ha="center",
                va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Adjusted C-index chart → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Covariate-adjusted Cox helper
# ──────────────────────────────────────────────────────────────────────────────

def run_adjusted_cv(
    X_embed: np.ndarray, cov_age: np.ndarray,
    cov_stage: np.ndarray, y: np.ndarray,
    method_name: str,
) -> dict:
    """
    5-fold × N_SEEDS Cox with (embedding + age + stage) covariates.
    Returns OOF predictions for primary seed as well as fold C-indices.
    Age is standardised per fold; stage NaN imputed with fold train median.
    """
    fold_cidxs  = []
    oof_risks   = np.full(len(y), np.nan)
    oof_events  = np.full(len(y), np.nan)
    oof_times   = np.full(len(y), np.nan)
    primary_seed = PRIMARY_SEED % N_SEEDS

    log.info(f"  [{method_name}-adj] Covariate-adjusted CV...")

    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_embed)):
            y_te = y[te_idx]
            if y_te["event"].sum() < 2:
                continue

            # Impute stage within fold (train median)
            stage_tr = cov_stage[tr_idx].copy().astype(float)
            stage_te = cov_stage[te_idx].copy().astype(float)
            tr_med   = np.nanmedian(stage_tr)
            stage_tr = np.where(np.isnan(stage_tr), tr_med, stage_tr)
            stage_te = np.where(np.isnan(stage_te), tr_med, stage_te)

            age_tr = cov_age[tr_idx].copy().astype(float)
            age_te = cov_age[te_idx].copy().astype(float)
            age_tr = np.where(np.isnan(age_tr), np.nanmean(age_tr), age_tr)
            age_te = np.where(np.isnan(age_te), np.nanmean(age_tr), age_te)

            # Build feature matrix: embedding + standardised age + stage
            Xtr_cov = np.column_stack([X_embed[tr_idx],
                                        age_tr[:, None],
                                        stage_tr[:, None]])
            Xte_cov = np.column_stack([X_embed[te_idx],
                                        age_te[:, None],
                                        stage_te[:, None]])

            risks_te, _ = fit_ridge_cox_fold(
                Xtr_cov, y[tr_idx], Xte_cov
            )
            c = concordance_index_censored(
                y_te["event"], y_te["time"], risks_te
            )[0]
            fold_cidxs.append(c)

            if seed == primary_seed:
                oof_risks[te_idx]  = risks_te
                oof_events[te_idx] = y_te["event"].astype(float)
                oof_times[te_idx]  = y_te["time"]

    mean_ci = float(np.mean(fold_cidxs))
    std_ci  = float(np.std(fold_cidxs))
    log.info(f"  [{method_name}-adj] C-index = {mean_ci:.4f} ± {std_ci:.4f}")

    return {
        "method":     f"{method_name}-adj",
        "fold_cidxs": fold_cidxs,
        "mean_ci":    mean_ci,
        "std_ci":     std_ci,
        "oof_risks":  oof_risks,
        "oof_events": oof_events,
        "oof_times":  oof_times,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         STEP 8: SURVIVAL MODELS & C-INDEX COMPARISON    ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"CV scheme:    {N_SPLITS}-fold × {N_SEEDS} seeds")
    log.info(f"Bootstrap N:  {N_BOOT}")
    log.info(f"Delta target: δ ≥ {DELTA_TARGET} with 95% CI excluding zero")

    # ── Load clinical data ─────────────────────────────────────────────────────
    log.info("\nLoading clinical data...")
    emb_snf_df = pd.read_parquet(PROCESSED / "spectral_embedding.parquet")
    patient_ids = emb_snf_df.index.tolist()
    n = len(patient_ids)

    clin = pd.read_parquet(PROCESSED / "clinical_preprocessed.parquet")
    clin = clin.loc[patient_ids]

    events = clin["os_event"].values.astype(bool)
    times  = clin["os_time"].values.astype(float)
    y_surv = make_surv_array(events, times)

    cov_age   = clin["age_at_diagnosis"].values.astype(float)
    cov_stage = clin["tumor_stage_numeric"].values.astype(float)

    log.info(f"  Patients: {n}  Events: {events.sum()}  "
             f"Event rate: {100*events.mean():.1f}%")
    log.info(f"  Median follow-up: {np.median(times):.0f} days")

    # ── Load embeddings ────────────────────────────────────────────────────────
    log.info("\nLoading embeddings...")
    emb_files = {
        "SNF":          "spectral_embedding.parquet",
        "RNA-only":     "embedding_rna.parquet",
        "CNV-only":     "embedding_cnv.parquet",
        "Meth-only":    "embedding_meth.parquet",
        "Early-concat": "embedding_early.parquet",
    }
    embeddings = {}
    for name, fname in emb_files.items():
        path = PROCESSED / fname
        df   = pd.read_parquet(path).loc[patient_ids]
        embeddings[name] = df.values.astype(np.float64)
        log.info(f"  {name}: {embeddings[name].shape}")

    # C2: concat(rna + cnv + meth) = 150-dim
    rna_e  = embeddings["RNA-only"]
    cnv_e  = embeddings["CNV-only"]
    meth_e = embeddings["Meth-only"]
    emb_c2 = np.concatenate([rna_e, cnv_e, meth_e], axis=1)   # 644 × 150
    embeddings["Late-C2 (concat)"] = emb_c2
    log.info(f"  Late-C2: {emb_c2.shape}")

    # dict for Late-C1 per-modality fits
    emb_singleomics = {
        "rna":  rna_e,
        "cnv":  cnv_e,
        "meth": meth_e,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Run CV for all standard methods
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("CROSS-VALIDATED COX MODELS")
    log.info("═"*60)

    results = {}
    for name, X in embeddings.items():
        res = run_cv_evaluation(X, y_surv, name)
        results[name] = res

    # Late C1
    res_c1 = run_late_c1(emb_singleomics, y_surv)
    results["Late-C1 (risk avg)"] = res_c1

    # ══════════════════════════════════════════════════════════════════════════
    # Covariate-adjusted Cox (SNF and best single-omics)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("COVARIATE-ADJUSTED COX")
    log.info("═"*60)

    res_snf_adj = run_adjusted_cv(
        embeddings["SNF"], cov_age, cov_stage, y_surv, "SNF"
    )
    res_rna_adj = run_adjusted_cv(
        embeddings["RNA-only"], cov_age, cov_stage, y_surv, "RNA-only"
    )
    results["SNF-adj"]      = res_snf_adj
    results["RNA-only-adj"] = res_rna_adj

    # ══════════════════════════════════════════════════════════════════════════
    # Bootstrap C-index + CI for all methods
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("BOOTSTRAP C-INDEX (1000 resamples)")
    log.info("═"*60)

    metrics_rows = []
    for name, res in results.items():
        oof_c, lo, hi = bootstrap_cindex(
            res["oof_events"], res["oof_times"], res["oof_risks"]
        )
        res["oof_cindex"] = oof_c
        res["ci_lower"]   = lo
        res["ci_upper"]   = hi
        log.info(f"  {name:<30s} C={oof_c:.4f}  "
                 f"95%CI=[{lo:.4f}, {hi:.4f}]  "
                 f"fold_mean={res['mean_ci']:.4f}±{res['std_ci']:.4f}")
        metrics_rows.append({
            "method":          name,
            "fold_mean_cindex": res["mean_ci"],
            "fold_std_cindex":  res["std_ci"],
            "oof_cindex":       oof_c,
            "ci_lower_95":      lo,
            "ci_upper_95":      hi,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Delta C-index: SNF vs each baseline (paired bootstrap)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("DELTA C-INDEX (SNF vs baselines, paired bootstrap)")
    log.info("═"*60)

    snf_res      = results["SNF"]
    baselines    = ["RNA-only", "CNV-only", "Meth-only",
                    "Early-concat", "Late-C1 (risk avg)", "Late-C2 (concat)"]
    delta_rows   = []

    for baseline in baselines:
        b_res = results[baseline]
        d, lo, hi = bootstrap_delta(
            snf_res["oof_events"], snf_res["oof_times"],
            snf_res["oof_risks"], b_res["oof_risks"],
        )
        excludes_zero = (lo > 0) or (hi < 0)
        meets_criterion = excludes_zero and (d >= DELTA_TARGET)  # delta ≥ 0.03 AND CI excludes zero (pre-specified criterion)
        flag = "✓" if excludes_zero else "✗"
        log.info(f"  SNF − {baseline:<28s}  "
                 f"Δ={d:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  {flag}")
        delta_rows.append({
            "comparison":       f"SNF − {baseline}",
            "delta":            d,
            "ci_lower":         lo,
            "ci_upper":         hi,
            "ci_excludes_zero": excludes_zero,
            "meets_delta_criterion": meets_criterion,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Save tables
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SAVING TABLES")
    log.info("═"*60)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = TABLE_OUT / "survival_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"  Survival metrics → {metrics_path}")

    delta_df   = pd.DataFrame(delta_rows)
    delta_path = TABLE_OUT / "delta_cindex.tsv"
    delta_df.to_csv(delta_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"  Delta C-index    → {delta_path}")

    # Covariate-adjusted table
    adj_table_rows = []
    for name, adj_name in [("SNF", "SNF-adj"), ("RNA-only", "RNA-only-adj")]:
        adj_table_rows.append({
            "method":          name,
            "unadj_oof_cindex": results[name]["oof_cindex"],
            "adj_oof_cindex":   results[adj_name]["oof_cindex"],
            "unadj_fold_mean":  results[name]["mean_ci"],
            "adj_fold_mean":    results[adj_name]["mean_ci"],
        })
    adj_df = pd.DataFrame(adj_table_rows)
    adj_path = TABLE_OUT / "covariate_adjusted_cindex.tsv"
    adj_df.to_csv(adj_path, sep="\t", index=False, float_format="%.4f")
    log.info(f"  Adjusted Cox     → {adj_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═"*60)
    log.info("SAVING FIGURES")
    log.info("═"*60)

    # Ordered display (SNF first, then baselines, then adjusted)
    display_order = [
        "SNF", "RNA-only", "CNV-only", "Meth-only",
        "Early-concat", "Late-C1 (risk avg)", "Late-C2 (concat)",
        "SNF-adj", "RNA-only-adj",
    ]
    forest_rows = []
    for name in display_order:
        res = results[name]
        forest_rows.append({
            "method":     name,
            "oof_cindex": res["oof_cindex"],
            "ci_lower":   res["ci_lower"],
            "ci_upper":   res["ci_upper"],
        })

    forest_plot_cindex(
        forest_rows,
        FIG_OUT / "cindex_comparison.pdf",
        title="Survival C-index comparison\n(OOF ± 95% bootstrap CI, N=1000)",
    )
    delta_forest_plot(
        delta_rows,
        FIG_OUT / "delta_cindex_vs_snf.pdf",
    )
    adjusted_cindex_chart(
        [{"method": name,
          "unadj_cindex": results[name]["oof_cindex"],
          "adj_cindex":   results[f"{name}-adj"]["oof_cindex"]}
         for name in ["SNF", "RNA-only"]],
        FIG_OUT / "adjusted_cindex.pdf",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║         SURVIVAL MODELS COMPLETE — SUMMARY              ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  {'Method':<30} {'OOF C-idx':>9} {'[95% CI]':>16} "
             f"{'Mean/fold':>9}")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for name in display_order:
        res = results[name]
        flag = "★" if name == "SNF" else " "
        log.info(
            f"║ {flag} {name:<29} "
            f"{res['oof_cindex']:>9.4f} "
            f"[{res['ci_lower']:.4f},{res['ci_upper']:.4f}] "
            f"{res['mean_ci']:>9.4f}"
        )
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Delta C-index (SNF vs best baseline):")
    snf_vs_rna = next(r for r in delta_rows if "RNA-only" in r["comparison"])
    d, lo, hi = snf_vs_rna["delta"], snf_vs_rna["ci_lower"], snf_vs_rna["ci_upper"]
    log.info(f"║    SNF − RNA-only: Δ={d:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
             + ("✓ CI>0" if lo > 0 else "✗ CI crosses 0"))
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Covariate-adjusted Cox:")
    for name, adj_name in [("SNF", "SNF-adj"), ("RNA-only", "RNA-only-adj")]:
        u = results[name]["oof_cindex"]
        a = results[adj_name]["oof_cindex"]
        log.info(f"║    {name}: unadj={u:.4f}  adj={a:.4f}  "
                 f"Δ(adj-unadj)={a-u:+.4f}")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("\n✓ Next: python scripts/09_sensitivity.py")
