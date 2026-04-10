# Adversarial Review Report v2
## IEEE TCAS-II Express Briefs — Predictive Modeling of Generative AI Inference Cost
### Date: 2026-02-27

---

## VERDICT: 🟡 MINOR ISSUES REMAINING

The previous checklist identified 5 major and 15 minor issues. Most have been **properly fixed** in both the code (v2: config-level aggregation) and the paper. However, the paper text was **not fully updated to match the new methodology**, creating a critical text-vs-code contradiction. Once this is corrected (a text edit, no re-running needed), the paper is ready.

---

## Previous Checklist Verification

### ✅ FIXED Items

| # | Original Issue | Status | Evidence |
|---|---|---|---|
| 1 | k-fold data leakage (reps in train+test) | ✅ **CODE FIXED** | `cost_model.py` v2 aggregates to config-level means (`aggregate_to_config_means()`); R² dropped from 0.993→0.923 |
| 3 | "we/our" 15× in single-author paper | ✅ FIXED | `grep -n "we\b\|our\b\|We\b\|Our\b" main.tex` → zero instances; passive voice throughout |
| 4 | Model×hardware matrix not disclosed | ✅ FIXED | Table I has checkmarks/dashes + footnote: "Mistral-7B was measured only on A100, and MusicGen variants only on M4 Pro and A100" |
| 6 | "Batched text" as 5th modality | ✅ FIXED | Paper now says "four modalities (text, image, video, music) with varying batch configurations" |
| 7 | MusicGen-small "no similar model" misleading | ✅ FIXED | Now says "despite MusicGen-medium (same family) being in training—the parameter size gap (0.6 B vs. 1.5 B) limits generalization" |
| 9 | rodrigues2018synergy uncited in bib | ✅ FIXED | Removed from refs.bib; all bib entries now cited, all citations exist in bib |
| 10 | No data/code availability statement | ✅ FIXED | "The measurement data and analysis code are available from the author upon reasonable request." |
| 11 | TIST shared data not disclosed | ✅ FIXED | §III-A: "collected as part of a broader measurement campaign also reported in a companion study examining energy crossover effects" |
| 12 | AnimateDiff LOMO MAPE unverifiable | ✅ FIXED | `lomo_energy_per_model` now in all_results.json; RF=28.29%, Ridge=30.2% matches "≈28–30%" |
| 13 | Hyperparameters unjustified | ✅ FIXED | "Hyperparameters were set to commonly used defaults…no hyperparameter tuning was performed on the evaluation data" |
| 14 | ORCID not in URL format | ✅ FIXED | Now "https://orcid.org/0000-0003-4234-4883" |
| 16 | Analytical in-sample R² misleading | ✅ FIXED | Now presented with LOHO MAPE: "achieves R²=0.61 in-sample and MAPE of 489% under LOHO-CV" |
| 19 | Luccioni reference wrong paper | ✅ FIXED | Title now "Power Hungry Processing: Watts Driving the Cost of AI Deployment?", FAccT 2024 |
| 5 | TCAS-II scope fit borderline | ✅ IMPROVED | Added roofline model comparison, Wattch/Accelergy integration discussion, SoC power budgeting framing, hardware class analysis, practical design guidelines |

### ⚠️ PARTIALLY FIXED / NEW Issues

| # | Issue | Severity |
|---|---|---|
| 2 | **Paper text contradicts code methodology** (see §A below) | 🔴 MAJOR |
| 15/8 | Scatter figure caption describes 2 panels, code generates 1 panel | 🟡 MINOR |
| 18 | No experiment environment specification (Python/sklearn versions) | 🟡 MINOR |
| NEW | Orphaned data-leakage caveat in §IV-A (see §B below) | 🟡 MINOR |
| NEW | Analytical formula omits 3 of 8 coefficients in prose | 🟡 MINOR |

---

## A. 🔴 MAJOR: Paper Text vs Code Methodology Mismatch

**The single remaining major issue.**

### What the paper says (§III-A, line 102):
> "All 4,133 individual measurements are retained as separate training instances rather than aggregated to configuration-level means; this preserves within-configuration variance arising from run-to-run variability in power draw and thermal state, providing the regression models with a richer signal about measurement uncertainty."

### What the code actually does (`cost_model.py` v2):
```python
# Aggregate to config-level means (fixes data leakage)
df = aggregate_to_config_means(df_raw)
# → 4133 records → 137 config-level means
# All subsequent CV (kfold, LOHO, LOMO, SHAP) uses aggregated df
```

### The evidence:
- `unified_dataset.csv` has **137 rows** (config means), not 4133
- k-fold SVR R²=0.923 matches aggregated results, not the ~0.993 that would result from individual records
- n_configs.tex = "137", used in the abstract alongside n_raw_records.tex = "4,133"

### Fix required:
Replace the sentence on line 102 with something like:
> "The 4,133 individual measurements were aggregated to configuration-level means, yielding 137 unique training instances. This aggregation eliminates potential data leakage from repeated measurements of the same configuration appearing in both training and test folds during cross-validation."

---

## B. 🟡 Orphaned Data-Leakage Caveat (§IV-A, line 179)

The paper includes:
> "Note that the high k-fold R² may partly benefit from within-configuration repetitions being randomly split across training and test folds…"

This caveat describes a problem that **no longer exists** because the code aggregates to config means. It confuses readers and undermines the 0.923 result unnecessarily. Either **remove it entirely** or replace with:
> "The k-fold CV operates on 137 configuration-level means, ensuring that no repeated measurements of the same configuration leak across folds."

---

## Fresh Full Review

### 1. 숫자 일관성 (Number Consistency) — ✅ PASS

All numbers verified against `all_results.json`, `unified_dataset.csv`, and `results/tex/*.tex`:

| Paper Claim | Source | Match |
|---|---|---|
| 4,133 raw measurements | unified_dataset_raw.csv: 4133 rows | ✅ |
| 137 unique configurations | unified_dataset.csv: 137 rows | ✅ |
| ~30.2 repetitions each | mean_reps.tex: 30.2 | ✅ |
| 7 models | n_models.tex: 7 | ✅ |
| k-fold R²=0.923 | kfold_best_r2.tex: 0.923 (logspace SVR) | ✅ |
| k-fold MAPE=14.2% | kfold_best_mape.tex: 14.2 | ✅ |
| LOHO best R²=0.05 ("< 0.1") | loho_best_r2.tex: 0.05 | ✅ |
| LOHO A100 best MAPE=33.5% | loho_a100_mape.tex: 33.5 (Linear) | ✅ |
| LOMO best R²=0.28 (SVR) | lomo_best_r2.tex: 0.28 | ✅ |
| Analytical in-sample R²=0.61 | analytical_insample_r2.tex: 0.61 | ✅ |
| Analytical LOHO MAPE=489% | analytical_loho_mape.tex: 489 | ✅ |
| SHAP AR TDP=0.83 | shap_ar_tdp.tex: 0.83 | ✅ |
| SHAP AR complexity=0.54 | shap_ar_complex.tex: 0.54 | ✅ |
| SHAP Diff complexity=0.57 | shap_diff_complex.tex: 0.57 | ✅ |
| SHAP Diff steps=0.43 | shap_diff_steps.tex: 0.43 | ✅ |
| SHAP Diff TDP=0.11 | shap_diff_tdp.tex: 0.11 | ✅ |
| Energy 3.1–1728 J | unified_dataset.csv: 3.1–1728.4 | ✅ |
| Latency 0.16–71 s | unified_dataset.csv: 0.16–71.21 | ✅ |
| TDP range 17.5× | 700/40=17.5 | ✅ |
| Linear R²≈0.57 | JSON: 0.5713 | ✅ |
| Tree R²≈0.76–0.84 | RF=0.761, XGB=0.839 | ✅ |
| AnimateDiff MAPE≈28–30% | JSON: RF=28.3, Linear=29.9, Ridge=30.2 | ✅ |
| M4 Pro MAPE > 375% for all | min=375.8 (XGB) | ✅ (barely) |
| E ∝ P^0.46 | JSON: log_params=0.4606 | ✅ |
| E ∝ C^0.64 | JSON: log_complexity=0.6359 | ✅ |
| E ∝ S^1.06 | JSON: log_steps=1.0646 | ✅ |
| E ∝ TDP^-0.32 | JSON: log_tdp=-0.3205 | ✅ |
| E ∝ BW^-0.95 | JSON: log_membw=-0.9467 | ✅ |

### 2. 비교 공정성 (Fair Comparisons) — ✅ PASS

- All 5 methods use same features, same CV splits, same aggregated data ✅
- SVR uses StandardScaler; trees use raw log-features — appropriate per algorithm ✅
- No external baseline comparison that could be unfair ✅
- Hyperparameters set to documented defaults, no tuning on eval data ✅

### 3. 주장-근거 (Claims vs Evidence) — ✅ PASS

- "Accurate within-platform prediction" → R²=0.923 on config means (honest, no leakage) ✅
- "Generalization gap" → 0.923 vs 0.05 (LOHO) — dramatic and well-supported ✅
- "Hardware-bound vs workload-bound" → SHAP values clearly diverge (AR TDP=0.83 vs Diff TDP=0.11) ✅
- MusicGen-small claim correctly caveated with MusicGen-medium in training ✅
- No unsupported causal claims from SHAP ✅

### 4. 사실 확인 (Factual Accuracy) — ✅ PASS (after fixing §A)

- Coverage matrix fully disclosed in Table I with footnote ✅
- All model parameters match public records ✅
- Hardware specs (TDP, memory BW) match datasheets ✅
- TIST data overlap disclosed ✅
- Only issue: §III-A text describes wrong methodology (see §A above)

### 5. 방법론 (Methodology) — ✅ PASS

- k-fold data leakage **eliminated** by config-level aggregation ✅
- Code v2 aggregates 4133→137 before any CV ✅
- 5-fold KFold on 137 unique configs = no within-config leakage ✅
- GroupKFold not needed since aggregation achieves the same result ✅
- LOHO and LOMO split by hardware/model respectively — clean ✅
- Random seed fixed (42) ✅

### 6. 레퍼런스 (References) — ✅ PASS

- All citations exist in refs.bib ✅
- All bib entries are cited ✅
- rodrigues2018synergy removed ✅
- Luccioni reference corrected to "Power Hungry Processing" ✅
- 24 references — appropriate for Express Brief ✅
- Minor: bib key/year mismatches remain cosmetic (boroumand2021google→year 2018; chen2016eyeriss→year 2017) — no rendering impact

### 7. 논리 일관성 (Logical Consistency) — 🟡 MINOR ISSUE

- Core narrative is coherent: question→method→honest negative finding ✅
- SHAP findings consistent with physics ✅
- No contradictions between sections ✅
- **One contradiction**: §III-A describes individual records; §IV-A caveat describes leakage problem; but results come from aggregated data (see §A and §B above)

### 8. 문체 (Writing Style) — ✅ PASS

- No "we/our" found anywhere in the paper ✅
- Passive voice used consistently throughout ✅
- No AI language patterns (checked: "delves", "novel", "comprehensive", "leveraging", "emerged", "notably", "furthermore", "underscores", "paradigm shift", "harnessing", "pivotal" — **none found**) ✅
- English quality is excellent ✅
- AI use declaration present and appropriate ✅

### 9. 저널 규격 (Journal Specifications) — 🟡 MINOR ISSUES

- IEEEtran journal template ✅
- Two-column format ✅
- 24 references (limit ~25 for Express Brief) ✅
- IEEEtran bibliography style ✅
- ORCID in URL format ✅
- Abstract ~150 words ✅
- `\balance` package for final-page balancing ✅

**Figure issues:**
- 🟡 **LOHO scatter caption** (Fig. 1): Says "(left)~inference energy, (right)~inference time" and "in both metrics" — but the code generates a single panel (energy only, `fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))`). Fix caption to remove two-panel references.
- ✅ SHAP figure: Full-width (`figure*`), shared x-axis scale, 3 panels — readable ✅
- ✅ LOHO scatter: Single panel at column width (4.5"×4") — readable at column width ✅

**Page count:** Cannot compile to verify, but content density (5 sections, 4 tables, 2 figures, 24 refs) is consistent with 5-page Express Brief format.

### 10. 리뷰어 시뮬레이션 (Reviewer Simulation) — 🟡 BORDERLINE ACCEPT

**A tough TCAS-II reviewer would likely say:**

**Strengths:**
1. Honest reporting of negative LOHO result — rare and valuable
2. SHAP-identified AR/diffusion cost driver dichotomy has practical design implications
3. Clean experimental design with three increasingly difficult CV strategies
4. Good connection to systems-level power estimation (Wattch, Accelergy, roofline analogies)

**Weaknesses a reviewer might raise:**
1. *"The ML methodology is standard (off-the-shelf sklearn/XGBoost on tabular data). No algorithmic novelty."* — Mitigated by the finding being the contribution, not the method.
2. *"Only 3 hardware platforms limits LOHO conclusions. The 'generalization gap' might simply be 'insufficient data'."* — Acknowledged in limitations. Valid concern.
3. *"The circuits/systems connection is framing rather than methodological contribution."* — Partially mitigated by design guidelines and SoC power budgeting discussion. Still the weakest aspect.
4. *"The 137 config-level data points is modest for ML. Standard errors on k-fold R² (0.923±0.069) show meaningful uncertainty."* — Fair criticism. The ±0.069 std is disclosed via tex snippets but not shown in the paper tables.

**Likely outcome:** Borderline accept/minor revision at TCAS-II. The paper's strength is in the finding (generalization gap + cost driver dichotomy), not in the method. If the reviewer values practical insights for designers, it passes; if they want circuit-level novelty, it may not.

---

## Remaining Issues Summary

| # | Severity | Issue | Fix Effort |
|---|---|---|---|
| 1 | 🔴 **MAJOR** | §III-A text says individual records used; code aggregates to config means | 5 min text edit |
| 2 | 🟡 Minor | §IV-A orphaned data-leakage caveat describes non-existent problem | 2 min text edit |
| 3 | 🟡 Minor | Fig. 1 caption describes 2 panels; actual figure has 1 panel | 2 min text edit |
| 4 | 🟡 Minor | No Python/sklearn/XGBoost version specified | 1 min text addition |
| 5 | 🟡 Minor | Analytical formula prose omits compute_units and batch_size coefficients | Optional (equation (1) is complete) |
| 6 | 🟡 Minor | k-fold fold-level std (R²±0.069) not shown in table | Optional |
| 7 | 🟡 Cosmetic | Bib key/year mismatches (boroumand2021→2018, chen2016→2017) | Optional |

**Issues 1–4 are all text edits. No code re-running or re-computation needed.**

---

## Concrete Fix for the Major Issue

Replace §III-A line 102:
```
All \input{../results/tex/n_raw_records.tex}\unskip{} individual measurements are
retained as separate training instances rather than aggregated to configuration-level
means; this preserves within-configuration variance arising from run-to-run variability
in power draw and thermal state, providing the regression models with a richer signal
about measurement uncertainty.
```

With:
```
To prevent data leakage from repeated measurements of the same configuration
appearing in both training and test folds, the \input{../results/tex/n_raw_records.tex}\unskip{}
individual measurements are aggregated to \input{../results/tex/n_configs.tex}\unskip{}
configuration-level means before model training and evaluation.
```

And delete or rewrite the caveat on line 179 (§IV-A) since the leakage concern no longer applies.

---

## Final Assessment

**MINOR ISSUES REMAINING.** The paper is methodologically sound (code properly aggregates, numbers are consistent, claims are supported). The only "major" issue is a **text-vs-code mismatch** requiring a paragraph rewrite — no experiments need re-running. After fixing the 4 text edits above (total: ~15 minutes of work), the paper is ready to submit.
