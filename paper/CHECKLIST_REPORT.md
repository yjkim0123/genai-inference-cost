# Checklist & Adversarial Review Report
## IEEE TCAS-II Express Briefs — Predictive Modeling of Generative AI Inference Cost
### Date: 2026-02-26

---

## Summary Verdict: 🔴 MAJOR REVISION

The paper has a well-structured argument and an interesting negative finding (cross-platform prediction fails), but suffers from **three critical methodological issues** that must be resolved before submission: (1) data leakage in k-fold CV inflating headline R²=0.993, (2) false claim that means were used for modeling when individual repetitions are used, and (3) undisclosed incomplete model×hardware coverage matrix. Additionally, the TCAS-II scope fit is borderline.

---

## 1. 숫자 일관성 검증 🔢 — 🟡 Minor

**Numbers checked against `all_results.json` and `results/tex/*.tex`:**

| Paper claim | Source value | Match? |
|---|---|---|
| 4,133 records | CSV: 4133 rows, JSON: 4133 | ✅ |
| 7 models | JSON: 7, CSV: 7 unique | ✅ |
| k-fold R²=0.993 | kfold_best_r2.tex: 0.993 | ✅ |
| k-fold MAPE=7.1% | kfold_best_mape.tex: 7.1 | ✅ |
| LOHO XGB R²=0.08 | loho_xgb_r2.tex: 0.08 | ✅ |
| LOHO A100 best MAPE=35.1% | loho_a100_mape.tex: 35.1 (Ridge) | ✅ |
| LOMO best R²=0.53 | lomo_best_r2.tex: 0.53 (SVR) | ✅ |
| Analytical LOHO MAPE=558% | analytical_loho_mape.tex: 558 | ✅ |
| SHAP AR TDP=0.83 | shap_ar_tdp.tex: 0.83 | ✅ |
| SHAP Diff complexity=0.56 | shap_diff_complex.tex: 0.56 | ✅ |
| SHAP Diff steps=0.43 | shap_diff_steps.tex: 0.43 | ✅ |
| Energy range 1.3–1779 J | CSV: 1.3–1779.4 J | ✅ |
| Time range 0.05–100 s | CSV: 0.05–100.41 s | ✅ |
| 17.5× TDP range | 700/40 = 17.5 | ✅ |
| Linear R² ≈ 0.58 | JSON: 0.5766 | ✅ |

🟡 **Minor issue:** The "R² = 0.61 on the full dataset" for the analytical formula is not saved in `all_results.json` (it's the in-sample R², computed in the code but not persisted). This number is not independently verifiable from the saved artifacts.

🟡 **Minor issue:** "AnimateDiff MAPE ≈ 20–30%" (LOMO per-model) — the per-model LOMO results are not saved in `all_results.json`. The code prints them but doesn't persist them. **Unverifiable claim.**

---

## 2. 비교 공정성 검증 ⚖️ — 🟢 OK

- All five regression methods use the same features and the same CV splits ✅
- Comparison is method-vs-method on identical data, not against external baselines ✅
- SVR uses standardized features; tree models use raw — appropriate for each ✅
- No unfair advantage to any method ✅

---

## 3. 주장-근거 검증 📋 — 🟡 Minor

🟢 Core claims are supported by evidence:
- "Accurate within-platform prediction" → k-fold R²=0.993 (but see §5 data leakage)
- "Generalization gap" → LOHO R²=0.08 vs k-fold R²=0.993
- "Hardware-bound vs workload-bound" → SHAP values diverge

🟡 **"MusicGen-small proves challenging due to its unique audio generation profile with no similar model in the training set"** — This is misleading. MusicGen-medium (same architecture family, same modality) IS in the training set during LOMO-CV for MusicGen-small. The claim should be: "despite MusicGen-medium providing some training signal, the parameter size difference (0.59B vs 1.5B) limits generalization."

🟡 **"MAPE > 450% for most models" for M4 Pro LOHO** — Technically "all models" not "most": Linear 982%, Ridge 984%, RF 453%, XGBoost 453%, SVR 1301%. All > 450%.

---

## 4. 사실 확인 ✅ — 🔴 MAJOR

🔴 **FALSE CLAIM — "means were used for modeling"** (line 101): The paper states "Each configuration was repeated 30 times to capture run-to-run variance, and means were used for modeling." **This is false.** The code (`cost_model.py`, `load_all_data()`) loads every individual measurement as a separate row. The unified dataset has 4,133 records across only 135 unique configurations (~30.6 reps per config). The model trains on individual repetitions, not means.

🟡 **"five modalities"** — The paper lists "(text, image, video, music, batched text)." Batched text is the same modality as text with `batch_size > 1`. Batch size is already a separate feature. Calling this a 5th modality inflates the claim. Should say "four modalities with varying batch configurations."

🟡 **"seven generative models...across three hardware platforms"** — This implies full 7×3 coverage, but the actual coverage matrix is sparse:

| Model | M4 Pro | A100 | H100 |
|---|---|---|---|
| Phi-3-mini-4k | ✅ | ✅ | ✅ |
| Mistral-7B | ❌ | ✅ | ❌ |
| MusicGen-small | ✅ | ✅ | ❌ |
| MusicGen-medium | ✅ | ✅ | ❌ |
| SD-v1-5 | ✅ | ✅ | ✅ |
| SDXL-base-1.0 | ✅ | ✅ | ✅ |
| AnimateDiff-v1.5 | ✅ | ✅ | ✅ |

Only 4 models are on all 3 platforms. Mistral only exists on A100. **This must be disclosed** as it affects LOHO and LOMO results. When H100 is held out, predictions are only for 4 models. When Mistral is held out in LOMO, only A100 test data exists.

---

## 5. 방법론 검증 🔬 — 🔴 MAJOR

🔴 **DATA LEAKAGE in k-fold CV:** With 4,133 records from 135 unique configs (~30 reps each), random 5-fold splitting puts ~24 repetitions of each config in train and ~6 in test. The model memorizes config→energy mappings from training repetitions and "predicts" near-identical test repetitions. **The R²=0.993 headline number is inflated by this leakage.**

**Evidence:** The k-fold R²=0.993 is suspiciously close to perfect, while LOHO (R²=0.08) and LOMO (R²=0.53) — which split by orthogonal grouping variables — show dramatically worse performance. This gap is partially genuine (cross-platform/model generalization IS harder), but the k-fold number is also artificially high.

**Fix options:**
1. Aggregate to config-level means (as the paper claims), giving ~135 rows. Train/test on means.
2. Use `GroupKFold` where each unique config is a group, ensuring all repetitions of one config go to the same fold.
3. Both approaches will yield a lower but more honest R².

🟡 **No confidence intervals or error bars:** 5-fold CV gives a single R² number. With 135 effective configs split into 5 folds, variance can be substantial. Report mean ± std across folds.

🟡 **Hyperparameter selection not justified:** RF (200 trees, depth 12), XGBoost (200 boosters, depth 8, LR 0.1), SVR (C=100) — how were these chosen? No hyperparameter search is described. Were these tuned on the full dataset (another leakage source)?

---

## 6. 레퍼런스 검증 📚 — 🟡 Minor

🟢 All 24 cited references exist in `refs.bib` ✅
🟢 All bib entries are cited except one ✅
🟢 Boroumand et al. verified via Semantic Scholar: "Google Workloads for Consumer Devices: Mitigating Data Movement Bottlenecks", ASPLOS 2018 ✅

🟡 **Uncited bib entry:** `rodrigues2018synergy` (Rodrigues et al., "Synergy of Software and Hardware Approaches for GPU Energy Estimation") is in `refs.bib` but never cited. Remove it.

🟡 **Bib key/year mismatches (cosmetic, no rendering impact):**
- `boroumand2021google` → year={2018} (key says 2021)
- `chen2016eyeriss` → year={2017} (key says 2016; paper was ISCA 2016, JSSC 2017)
- `luccioni2024power` → year={2023} (key says 2024)

🟡 **Luccioni reference:** The bib entry title is "Estimating the Carbon Footprint of BLOOM" but the paper cites it in the context of "energy efficiency." The well-known Luccioni energy paper is actually "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" (2024). The BLOOM carbon footprint paper (2023) is a different work. Verify this is the intended reference.

---

## 7. 논리 일관성 검증 🧠 — 🟢 OK

🟢 Introduction poses the question → Results answer it ✅
🟢 The negative finding (LOHO fails) is honestly reported ✅
🟢 Limitations section acknowledges "three-platform scope" ✅
🟢 SHAP findings are consistent with physical intuition (AR = hardware-bound, diffusion = workload-bound) ✅
🟢 No contradictions between sections ✅

---

## 8. 문체 검증 ✍️ — 🔴 MAJOR

🔴 **Single-author "we" problem:** The paper uses "we" / "our" **15 times** (e.g., "We present," "Our contributions," "We collected," "Our key findings"). For a single-author paper, this should be "The author presents" or passive voice ("A systematic framework is presented"). While IEEE technically permits editorial "we," TCAS-II reviewers for a single-author Express Brief will flag this.

**Instances requiring change:**
- Abstract: "We present" → "This brief presents" / "A systematic framework is presented"
- Abstract: "we train and evaluate" → passive
- §I: "Our contributions" → "The contributions of this work"
- §I: "Our work fills" → "This work fills"
- §II-C: "We leverage SHAP" → "SHAP is leveraged"
- §III-A: "We collected" → "Inference energy measurements were collected"
- §III-C: "We engineer" → "Eight input features are engineered"
- §III-D: "We evaluate" → "Five regression approaches are evaluated"
- §III-E: "We evaluate" → same pattern
- §IV-E: "We also fit" → "A log-linear analytical model is also fit"
- §IV-F: "Based on our analysis, we offer" → "Based on the analysis, the following guidelines..."
- §V: "We presented" → "This brief presented"
- §V: "Our key findings" → "The key findings"
- §V: "our cost prediction framework" → "the cost prediction framework"

🟢 **No AI language patterns detected.** Checked for: "delves into," "it is worth noting," "novel framework," "comprehensive," "leveraging," "emerged as," "notably," "furthermore," "underscores," "paradigm shift," "harnessing," "pivotal." None found. ✅

---

## 9. 제출 전 최종 확인 — Deferred (depends on fixing 🔴 issues)

---

## 10. 실험 재현 검증 🔁 — 🟡 Minor

🟢 Code (`cost_model.py`) is complete, readable, and produces `all_results.json` ✅
🟢 `generate_tex.py` correctly extracts numbers from JSON to tex snippets ✅
🟢 Random seeds fixed (`random_state=42`) ✅

🟡 **No code/data availability statement in the paper.** For IEEE journals, this is increasingly expected. Add a statement: "Code and data are available at [URL]" or "upon reasonable request."

🟡 **Experiment environment not specified in paper:** No mention of Python version, sklearn version, XGBoost version, hardware used for training the ML models (distinct from the profiled hardware). Add to methodology.

---

## 11. 논문이 대답하는 질문 검증 🎯 — 🟡 Minor

🟢 **"So what?" test:** The paper answers a practical question (can you predict inference cost without running the workload?) and honestly reports that cross-platform prediction fails. This negative finding IS valuable. ✅

🟡 **Contribution clarity:** The paper's main finding is negative ("you can't generalize across hardware with only 3 platforms"). Positive contribution is thin: the SHAP analysis and practical guidelines are useful but may feel insufficient for a journal publication. A reviewer might ask: "What's the actionable takeaway beyond 'you need to profile each hardware'?"

🟡 **Topic timeliness:** GenAI inference cost is a hot topic in 2024-2026. No risk of being outdated. ✅

---

## 12. 흔한 AI 실수 패턴 🤖 — 🟢 OK

🟢 No fabricated numbers (all trace to code/data) ✅
🟢 No "approximately" or "generally" without basis ✅
🟢 No causal claims from SHAP (correctly says "cost drivers" not "causes") ✅
🟢 No fixed-institution errors ✅

---

## 13. 저널 규격 검증 📏 — 🟡 Minor

🟢 **Template:** IEEEtran journal class, two-column, correct for TCAS-II ✅
🟢 **Page count:** ~5 pages (estimated from content density) — matches Express Brief limit ✅
🟢 **References:** 24 — appropriate for Express Brief (typical: 15–25) ✅
🟢 **Keywords:** 6 (Energy modeling, generative AI, inference cost prediction, hardware-aware design, cross-platform estimation, SHAP analysis) ✅
🟢 **Table style:** booktabs, no vertical lines ✅
🟢 **Abstract:** ~150 words — within IEEE limit ✅

🟡 **Figure issues:**
- `loho_scatter.pdf`: Two-panel figure (energy + time) set at `\columnwidth` (single column). At ~3.5" width, each panel is ~1.75" wide — fonts will be illegibly small. Either:
  - Use `\begin{figure*}` for full-width, or
  - Show only the energy panel (time is not discussed in text), or
  - Stack panels vertically
- `shap_importance.pdf`: Full-width (`figure*`) with 3 panels — OK, but x-axis scales differ across panels (0–0.85 vs 0–0.56), making visual cross-panel comparison misleading. Standardize x-axis range.
- Both figures: A100 and H100 markers in the scatter plot use similar cyan shades — hard to distinguish in grayscale print. Use more distinct colors.

🟡 **The scatter figure caption mentions only energy but the figure shows both energy and time panels.** Either update the caption to describe both panels, or remove the time panel.

---

## 14. 저자 정보 정확성 검증 👤 — 🟡 Minor

🟢 Name: "Yongjun Kim" (no comma) ✅
🟢 Department: "Department of Software, Ajou University" ✅
🟢 Email: yjkim0123@ajou.ac.kr ✅
🟢 City: "Suwon 16499, South Korea" ✅

🟡 **ORCID format:** Currently "ORCID: 0000-0003-4234-4883". IEEE standard is to use the URL format: `https://orcid.org/0000-0003-4234-4883`.

🟡 **Author footnote:** "Y. Kim is with..." — standard IEEE uses "Y. Kim" for the short form, but should match the author block format. Acceptable. ✅

---

## 15. 리뷰 방식 & Blinding 검증 🕶️ — 🟢 OK

🟢 TCAS-II is **single-blind** → author info should be present ✅
🟢 Author name, affiliation, email, ORCID all included ✅
🟢 No need for anonymization ✅

---

## 16. 제출 포맷 검증 📄 — 🟡 Minor

🟢 IEEEtran template used ✅
🟢 `\bibliographystyle{IEEEtran}` ✅
🟢 Two-column format ✅
🟢 `\balance` package included for final-page balancing ✅

🟡 **No cover letter mentioned.** Prepare one addressing: why TCAS-II is appropriate, key contributions, circuits/systems relevance.

🟡 **No data availability statement.** Add before references.

🟡 **"Manuscript received XX; revised XX"** — placeholder, normal for submission but verify it's filled before final submission.

🟡 **Figure files:** PDF format ✅ (IEEE accepts PDF). DPI 300 ✅.

---

## 17. 리뷰어 시뮬레이션 🧑‍🔬

### A. 동기 & 노벨티 — 🟡

🟢 **Gap is clearly defined:** existing work reports costs but doesn't predict them ✅
🟡 **Novelty concern:** The core method is straightforward ML regression (RF, XGBoost, Ridge, SVR) on tabular data. No new algorithm or model architecture. The contribution is primarily the dataset and the negative finding. A TCAS-II reviewer may judge this as "incremental."
🟡 **Similar work exists:** LLM energy prediction papers are proliferating (Samsi 2023, Luccioni 2023/2024, Desislavov 2023). The multi-modality angle (text + image + video + music) is the differentiator but may not feel sufficient.

### B. 실험 설계 공격 — 🔴

🔴 **k-fold data leakage (repeated above):** 4133 records from 135 configs → repetitions leak across train/test in random k-fold. This is the biggest methodological vulnerability. A reviewer who checks the dataset size vs. config count will catch this immediately.

🔴 **Incomplete model×hardware matrix not disclosed:** A reviewer checking Table I against the data would find that Mistral-7B only exists on A100, MusicGen-small/medium only on M4 Pro + A100. This asymmetry biases LOHO results and is not discussed.

🟡 **Hyperparameter justification:** RF depth=12, XGBoost depth=8, SVR C=100 — no justification. Were these tuned? On what data? If on the full dataset, that's another leakage source.

🟡 **No temporal split consideration:** The 30 repetitions per config were presumably collected sequentially. If hardware temperature drifts during the session, later repetitions might systematically differ from earlier ones. This is a minor threat to validity.

### C. 결과 해석 공격 — 🟡

🟢 **Honest negative reporting:** The LOHO failure (R²=0.08) is presented prominently and discussed in depth. The paper doesn't hide negative results. ✅

🟡 **k-fold R²=0.993 framing:** The abstract leads with this impressive number, burying the negative LOHO result. A reader skimming the abstract gets an overly optimistic impression. Consider leading with the gap finding instead.

🟡 **"Near-perfect prediction"** — the text says "highly accurate cost prediction is feasible when the target hardware is known." This is technically about k-fold, but as argued above, the k-fold number is inflated. After fixing the leakage (using GroupKFold or means), the true within-platform R² will be lower.

🟡 **Analytical formula R²=0.61 "on the full dataset":** This is in-sample training R², which is a biased metric. The LOHO analytical R² is -7.61. Presenting the in-sample number could be seen as cherry-picking.

### D. 작성 품질 — 🟡

🟢 Abstract accurately reflects results ✅
🟢 Related work is comprehensive and fair ✅
🟢 Limitations honestly discussed ✅
🟢 English quality is excellent, no grammatical errors ✅
🟢 No AI language patterns ✅

🟡 **"we" vs single author** (see §8) — will be noticed immediately.

🟡 **Figure caption mismatch:** Fig. 1 (loho_scatter) caption describes only energy but figure has both energy and time panels.

### E. 재현성 & 윤리 — 🟡

🟡 **No code/data availability:** Neither public repository nor "upon request" statement.
🟡 **No experiment environment specification:** Python/library versions not mentioned.
🟡 **Shared data with TIST paper not disclosed:** The measurement data is shared with the TIST "Energy crossover effect" paper. IEEE requires disclosure of prior/concurrent publications using the same data. Add a note: "The energy measurements in this study were collected as part of a broader measurement campaign also used in [self-cite] for a different research question (observing crossover effects rather than building predictive models)."

🟢 **No IRB needed** (hardware measurements, not human subjects) ✅
🟢 **No AI usage declaration needed** for IEEE currently ✅

---

## 🔴 Critical: TCAS-II Scope Fit

**This paper's circuits/systems connection is THIN.** The actual methodology is:
1. Profile hardware energy with `nvidia-smi` / `powermetrics`
2. Engineer 8 tabular features
3. Train sklearn models (RF, XGBoost, SVR, Ridge, Linear)
4. Evaluate with k-fold, LOHO, LOMO
5. Run SHAP

There is **no circuit-level contribution**: no RTL analysis, no architecture simulation (like Wattch/Eyeriss), no power integrity modeling, no hardware design proposal, no accelerator design. The Wattch/Eyeriss/DRAMPower references in §II are framing citations only — the paper doesn't build on or contribute to that methodology.

**Mitigation suggestions:**
- Strengthen the "practical guidelines for circuit/system designers" section with concrete design examples (e.g., "a system designer choosing between A100 and H100 for a diffusion workload can use...").
- Add discussion of how the SHAP-identified cost drivers relate to hardware microarchitecture (e.g., why unified memory architecture makes M4 Pro unpredictable from GPU training data).
- Consider positioning as a "systems" paper with hardware-aware AI workload characterization rather than a "circuits" paper.
- Alternatively, target a different venue (e.g., IEEE TPDS, IEEE TCAD, ACM TACO, or a systems venue like ISPASS) where the ML-on-energy-data methodology is a better fit.

---

## Issue Summary

| # | Severity | Issue | Section |
|---|---|---|---|
| 1 | 🔴 Major | k-fold CV data leakage (repetitions in train+test) | §5 |
| 2 | 🔴 Major | "Means were used" is false — individual reps used | §4 |
| 3 | 🔴 Major | "we/our" used 15× in single-author paper | §8 |
| 4 | 🔴 Major | Incomplete model×hardware matrix not disclosed | §4 |
| 5 | 🔴 Major | TCAS-II circuits/systems scope fit is borderline | Scope |
| 6 | 🟡 Minor | "Batched text" counted as 5th modality | §4 |
| 7 | 🟡 Minor | MusicGen-small "no similar model" claim misleading | §3 |
| 8 | 🟡 Minor | Figure readability issues (scatter plot, SHAP scales) | §13 |
| 9 | 🟡 Minor | rodrigues2018synergy in bib but uncited | §6 |
| 10 | 🟡 Minor | No data/code availability statement | §16 |
| 11 | 🟡 Minor | TIST shared data not disclosed | §17E |
| 12 | 🟡 Minor | AnimateDiff LOMO MAPE claim unverifiable (not in JSON) | §1 |
| 13 | 🟡 Minor | Hyperparameters unjustified | §17B |
| 14 | 🟡 Minor | ORCID not in URL format | §14 |
| 15 | 🟡 Minor | Scatter figure caption/content mismatch (2 panels, 1 described) | §13 |
| 16 | 🟡 Minor | Analytical in-sample R²=0.61 potentially misleading | §17C |
| 17 | 🟡 Minor | No cover letter prepared | §16 |
| 18 | 🟡 Minor | No experiment environment specification | §10 |
| 19 | 🟡 Minor | Luccioni reference might be wrong paper | §6 |
| 20 | 🟡 Minor | Abstract leads with k-fold R² rather than gap finding | §17C |

---

## Recommended Action Plan

### Must fix before submission (🔴):
1. **Fix data leakage:** Either (a) aggregate to config-level means before modeling, or (b) use `GroupKFold` with config as group. Re-run all experiments. Update all numbers.
2. **Correct "means were used" statement:** After fixing #1, this becomes true. Or if using GroupKFold, rewrite to describe the actual methodology.
3. **Replace all "we/our" with passive voice or "the author."** 15 instances identified above.
4. **Disclose incomplete model×hardware matrix.** Add a note in §III-B stating which combinations exist and noting that Mistral-7B is only available on A100 and MusicGen variants are not measured on H100.
5. **Strengthen TCAS-II scope framing** or consider an alternative venue.

### Should fix (🟡):
6. Save per-model LOMO results to JSON for reproducibility.
7. Fix "five modalities" → "four modalities with batch-size variations."
8. Fix MusicGen-small LOMO description.
9. Fix figure issues (scatter: single panel or full-width; SHAP: standardize x-axes).
10. Remove rodrigues2018synergy from bib.
11. Add data availability statement.
12. Disclose TIST data overlap.
13. Add hyperparameter justification.
14. Format ORCID as URL.
15. Fix scatter figure caption.

---

## Final Verdict

**As a TCAS-II Express Briefs reviewer, I would recommend: MAJOR REVISION.**

The data leakage issue (#1) is the most critical — it invalidates the headline result. Once fixed, the honest k-fold R² (using GroupKFold or means) will be lower, possibly substantially, which changes the narrative. The paper's strength lies in the honest negative LOHO finding and the SHAP analysis, but the inflated k-fold number undermines credibility.

The single-author "we" (#3) would be immediately flagged and signals carelessness. The undisclosed model×hardware gaps (#4) would erode trust if a reviewer checks. The TCAS-II scope (#5) is borderline — the paper might have a better home at a systems or sustainable AI venue.

If all 🔴 issues are resolved and the revised k-fold numbers remain strong (e.g., R² > 0.95 with proper CV), the paper has a reasonable chance at TCAS-II. The cross-platform generalization gap is a genuinely useful finding for the community.
