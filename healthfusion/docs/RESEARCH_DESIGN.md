# Research design (Phase 2)

This document fixes the research design *before* reading model results. Every
decision is tied to something observed in the data during Phase 1
(`results/reports/01_phase1_feasibility_report.md`).

## 1. Two separate studies, not one dataset

MMASH and QoL_Stress differ in population, devices, protocol and target. They
are analysed as two studies with their own pipelines, validation and models.
They meet only in Phase 9, where comparable indicators are *described* side by
side. No records are pooled and no model is moved from one dataset to the other.

| | QoL_Stress (main contribution) | MMASH (secondary, exploratory) |
|---|---|---|
| Unit of analysis | participant x phase (2 rows per person) | participant (1 row per person) |
| n | 66 participants, 45 with all three modalities passing QC in both phases | 21 participants with a sleep record |
| Task | binary classification: stress vs relax | regression of one night's sleep efficiency |
| Validation | 5-fold participant-grouped CV x 5 repeats, nested tuning | leave-one-subject-out (LOSO) |
| Significance | corrected resampled t-test; within-participant label permutation | label permutation (1000) |

## 2. QoL_Stress design

**Label.** `phase` (relax = 0, stress = 1), as given in `ecg.csv`, `eda.csv` and `stai.csv`.

**Inputs (15 features, 3 modalities).** ECG (5): ultra-short HRV from the
30-s strip. EDA (5): skin-conductance statistics and response rate. HR (5):
Fitbit heart-rate statistics in an 8-min window. See `docs/FEATURE_DICTIONARY.md`.

**Phase alignment for Fitbit HR.** The HR files have no phase column and only a
time of day. Phase 1 showed that the Fitbit stream and the questionnaires share
one clock (the stream starts ~20 min before and ends at the stress-STAI
submission). Each phase window is therefore the 8 min that end at that phase's
STAI submission. Windows of equal length avoid duration confounds; 8 min fits
inside the shortest observed stress phase (9.1 min).

**Excluded inputs.**
* STAI is completed at the end of each phase: it is a self-report of the
  label, so using it would be leakage. It is used only as *external validation*
  (does the model's stress probability rise when self-reported anxiety rises?).
* PSS-14 has one value per participant; it is identical for both phases and
  carries no information about the phase.
* Absolute clock time: relax always precedes stress, so time encodes the label.
* Device `heart_rate` and `result_classification`: QC only.

**Sample definition.** The primary analysis uses complete-case participants
(all three modalities passing QC in both phases, n = 45) so that every modality
comparison is made on identical people. A sensitivity analysis uses all 66
participants (missing modalities imputed in early fusion, skipped in late
fusion).

**Models.** Logistic regression, RBF-SVM (Platt-calibrated), random forest,
XGBoost and gradient boosting; plus three baselines (chance/prior, mean Fitbit
HR alone, ECG HR alone). Deep learning is not used: 114 observations are far
too few.

**Fusion.**
* Early: concatenate scaled features of the selected modalities.
* Late: one model per modality, pooled in log-odds with equal weights.
* Stacked (weighted): non-negative logistic meta-model fitted on inner-CV
  out-of-fold log-odds inside each training fold. The weights are learned, never
  chosen by hand and never fitted on test data.
All 7 modality subsets are evaluated: ECG, EDA, HR, ECG+EDA, ECG+HR, EDA+HR, ECG+EDA+HR.

**Validation.** Outer: `GroupKFold(5, shuffle)` by participant, repeated with
seeds 42-46. Inner: `GroupKFold(3)` on training participants only, for
hyper-parameters and stacking weights. Imputation and scaling are pipeline steps
fitted on training folds. Every split is asserted to share no participant.

**Metrics.** Accuracy, precision, recall, F1, macro-F1, balanced accuracy,
ROC-AUC, confusion matrix (threshold fixed at 0.5), and *pairwise accuracy*:
for each held-out participant, is the stress observation scored higher than the
relax observation? Mean, SD and 95 % CI over the 25 outer folds are reported;
no single best fold is reported.

**Inference.** Model differences: Nadeau-Bengio corrected resampled t-test
(fold AUCs are not independent because training sets overlap). Better than
chance: permutation test that swaps labels *within* each participant
(200 permutations for linear models, 50 for heavier ones).

**Validation criterion for serving a model (fixed in advance).** Mean AUC > 0.5
and corrected t-test against 0.5 with p < 0.05.

## 3. MMASH design

**Target selection.** Candidate targets were compared on scientific grounds:

| Candidate | Verdict | Reason |
|---|---|---|
| Sleep efficiency (SE) | **primary** | objective (actigraphy), normalised for time in bed, available for 21 |
| WASO | secondary | objective but strongly tied to time in bed |
| Total sleep time | secondary | dominated by bedtime choices on one night |
| Sleep fragmentation index | secondary | derived index, noisy for one night |
| Pittsburgh (PSQI) | secondary, exploratory | retrospective (past month), not about the recorded night; self-report like several inputs |

**Prospective window.** Predictors come only from the pre-sleep period
(recording start to first In-Bed time). Nothing measured during or after the
night is an input, so the actigraphy that scores sleep cannot leak into the
features.

**Excluded inputs.** All `sleep.csv` columns (targets); night-time actigraphy
and RR; PSQI (contains a self-reported habitual sleep-efficiency component);
PANAS 9+1 and the wake-up saliva sample (measured after the night).

**Modalities.** Physiological (5, from RR), Behavioural (7, actigraphy +
diary), Psychological (7, questionnaires).

**Why participant-level and LOSO.** With 21 observations, a held-out test set
would hold 4-5 people. LOSO uses every participant once as a test case.
Overlapping windows are never used to inflate n.

**Models.** Predict-the-mean baseline, Ridge (alpha chosen by efficient LOO
inside each training fold), shallow random forest (fixed, untuned). Every Ridge
model of the primary target gets 1000 label permutations; every random-forest
model whose LOSO error is below the baseline gets 200 (RF is ~50x slower). Early and
equal-weight late fusion are run as exploratory comparisons; **stacked
fusion is not run**, because learning modality weights from 20 people per fold
is not meaningful.

**Validation criterion.** LOSO R^2 > 0 and permutation p < 0.05 after Holm
adjustment across all permutation-tested configurations of the primary target
(they all test the same question, so an unadjusted minimum would be optimistic).

## 4. Explainability

Out-of-fold SHAP: each fold's model explains only its held-out participants.
Reported: global mean |SHAP| (with fold-to-fold SD), beeswarm summaries, modality
share of total |SHAP|, rank stability across folds, agreement between model
families, and individual waterfall explanations. SHAP describes model
behaviour; it does not establish physiological causation.

## 5. Cross-dataset analysis

Only indicators computed the same way in both datasets are compared: HR,
RMSSD and SDNN from 30-s inter-beat windows. Three descriptive analyses:
distributions (Mann-Whitney U, Cliff's delta), within-dataset state effects
(MMASH asleep vs awake; QoL stress vs relax) and the sign of associations with
self-reported anxiety/stress. No transfer experiment is run and no
generalisation is claimed.

## 6. Reproducibility

Seeds: global 42; CV repeats 42-46. Library versions: `requirements.txt` and
`models/registry.json`. Each model card in `models/` stores its features,
hyper-parameters, seed, validation scheme and verdict.
