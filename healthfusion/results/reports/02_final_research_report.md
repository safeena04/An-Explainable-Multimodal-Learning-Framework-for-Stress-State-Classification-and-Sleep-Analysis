# HealthFusion: final research report

**Explainable multimodal learning for stress-state classification (QoL_Stress)
and participant-level sleep analysis (MMASH)**

Every number below comes from a file in `results/metrics/`; the file is named
in brackets. Figures are in `results/figures/`. The design was fixed before
modelling (`docs/RESEARCH_DESIGN.md`); one correction made during the analysis
is reported openly in Section 8.

---

## Abstract

Two separate studies were run. **(1) QoL_Stress (main contribution).** We
classified relax vs stress phases of 45 participants (90 observations; all
three modalities passing QC in both phases) from a 30-s wrist ECG, a ~2.7-min
EDA scan and 8 min of optical heart rate. We used participant-grouped nested
cross-validation (5 folds x 5 repeats). The best configuration, early fusion of
all 15 features with logistic regression, reached ROC-AUC 0.737 (95 % CI
0.697-0.776), balanced accuracy 0.642 and within-participant pairwise accuracy
0.778. It beat chance (within-participant permutation p = 0.005) and a
heart-rate-only baseline (corrected resampled t-test p = 0.0004). It was **not**
significantly better than the best single modality (EDA, AUC 0.682, p = 0.47)
or than heart rate alone (AUC 0.680, p = 0.31). Late and stacked fusion never
beat early fusion. The modality ranking is EDA ≈ HR > ECG, with all three
contributing similar SHAP shares (EDA 39 %, HR 33 %, ECG 28 %). Model stress
probabilities did not track self-reported anxiety change (Spearman rho = 0.14,
p = 0.35). **(2) MMASH.** With 21 participants and one night each, no model
predicted sleep efficiency better than the participant mean under
leave-one-subject-out validation after correcting for the number of
configurations tested, and no feature-outcome correlation survived FDR
correction. **(3) Cross-dataset.** Comparable 30-s HR/HRV indicators show the
expected sleep-wake contrast in MMASH, but almost no ECG change between relax
and stress in QoL_Stress.

---

## 1. Data actually used

| | QoL_Stress | MMASH |
|---|---|---|
| Participants in release | 66 | 22 (all men) |
| Analysed (primary) | 45 complete-case (90 rows) | 21 with a sleep record |
| Sensitivity sample | all 66 (132 rows) | - |
| Exclusions | logged in `data/processed/qol_qc_log.csv` | logged in `data/processed/mmash_qc_log.csv` |

Integrity problems found and handled are in the Phase 1 report
(`01_phase1_feasibility_report.md`, Section 5).

## 2. Manipulation check (QoL_Stress)

STAI state anxiety rose from 33.5 to 36.7 (mean +3.2; Cohen's dz = 0.51;
Wilcoxon p = 2.6e-5). Only 43 of 66 participants reported an increase (11 no
change, 12 a decrease) [`qol_manipulation_check.json`]. The induction worked
on average but was mild. This limits how well *any* physiological model can
separate the phases.

## 3. RQ2 - Can ECG, EDA and heart rate distinguish stress from relaxation?

### 3.1 Feature-level evidence (paired Wilcoxon, BH-FDR) [`qol_paired_tests.csv`, `qol_paired_effects.png`]

| Feature | n pairs | Median relax | Median stress | r (rank-biserial) | FDR p |
|---|---|---|---|---|---|
| eda_scl_range | 60 | 1.70 | 2.62 | +0.70 | <0.001 |
| eda_scl_std | 60 | 0.78 | 1.07 | +0.58 | <0.001 |
| fb_hr_p95 (bpm) | 64 | 80.9 | 83.9 | +0.46 | 0.008 |
| fb_hr_std (bpm) | 64 | 4.0 | 5.3 | +0.43 | 0.009 |
| fb_hr_slope (bpm/min) | 64 | -0.20 | -0.69 | -0.43 | 0.009 |
| ecg_hr_range (bpm) | 51 | 16.5 | 20.1 | +0.39 | 0.031 |
| fb_hr_mean (bpm) | 64 | 72.1 | 73.7 | +0.38 | 0.019 |
| ecg_sdnn (ms) | 51 | 53.0 | 63.2 | +0.37 | 0.039 |
| ecg_hr, ecg_rmssd, ecg_pnn50, eda_scl_mean, eda_resp_rate, eda_scl_slope, fb_hr_p05 | | | | -0.28 to +0.19 | > 0.05 |

Under stress, skin conductance became more variable and heart rate became
higher and more variable. Mean ECG heart rate and RMSSD did not change.
Short-strip SDNN and HR range *increased*, which is more consistent with
respiratory or movement changes (for example speaking during a task) than with
classic vagal withdrawal. The release does not describe the stressor, so this
cannot be resolved.

### 3.2 Classification [`qol_cv_summary.csv`, `qol_best_per_modality_set.csv`, `qol_modality_comparison.png`]

Best configuration per modality set (complete case, n = 45; mean over 25
participant-grouped outer folds, 95 % CI of the mean):

| Modality set | Best configuration | ROC-AUC (95 % CI) | Balanced acc. | Macro-F1 | Pairwise acc. |
|---|---|---|---|---|---|
| ECG + EDA + HR | early, logistic regression | **0.737** (0.697-0.776) | 0.642 | 0.632 | 0.778 |
| EDA + HR | early, random forest | 0.734 (0.695-0.774) | 0.640 | 0.630 | 0.791 |
| ECG + HR | early, logistic regression | 0.696 (0.655-0.738) | 0.631 | 0.611 | 0.707 |
| ECG + EDA | late, XGBoost | 0.691 (0.638-0.744) | 0.633 | 0.627 | 0.676 |
| EDA | random forest | 0.682 (0.630-0.735) | 0.656 | 0.651 | 0.724 |
| HR | logistic regression | 0.680 (0.644-0.715) | 0.647 | 0.632 | 0.676 |
| ECG | logistic regression | 0.621 (0.574-0.667) | 0.640 | 0.628 | 0.596 |
| Baseline: mean Fitbit HR only | logistic regression | 0.558 (0.535-0.580) | 0.542 | 0.535 | 0.689 |
| Baseline: ECG HR only | logistic regression | 0.532 (0.513-0.551) | 0.547 | 0.532 | 0.540 |
| Baseline: chance (prior) | - | 0.500 | 0.500 | 0.333 | 0.500 |

The best model's mean confusion matrix per CV repeat (90 observations) is
TN 31.6, FP 13.4, FN 18.8, TP 26.2 (fold-mean precision 0.68, recall 0.58)
[`qol_confusion_matrix_best.png`]. Its probabilities are compressed around 0.5
(strong regularisation was selected), so ranking quality (AUC, pairwise
accuracy) is clearly better than thresholded accuracy.

**Significance.** Within-participant label-permutation tests (single 5-fold
repeat) [`qol_permutation_tests.csv`]: all-modality logistic regression p = 0.005
(200 permutations); EDA logistic p = 0.005; HR logistic p = 0.005; EDA random
forest p = 0.039 (50); ECG logistic p = 0.015. The pre-specified serving
criterion (corrected resampled t-test of AUC vs 0.5) is stricter: ECG alone
gives p = 0.071 and is **not** served by the API. Every other modality set passes.

**Answer to RQ2.** Yes, but modestly. Wearable signals separate the phases
clearly above chance (AUC ~0.74 at best). This is far from reliable
per-observation classification, which is consistent with the mild stress
response in Section 2.

## 4. RQ3 and RQ4 - Which modality contributes most, and does fusion help?

**Modality ranking.** By the best single-modality model, EDA (0.682) ≈ HR (0.680)
> ECG (0.621). By SHAP share in the best all-modality model, EDA 39 %, HR 33 %,
ECG 28 % [`shap_modality_share.csv`]. Stacking gave the largest mean weight to
HR (0.68), then EDA (0.49) and ECG (0.45), with fold-to-fold SDs of 0.19-0.31
[`qol_stacking_weights.csv`]. The three views agree that no single
modality dominates and that ECG is the weakest alone. At this sample size the
EDA-vs-HR order is not resolved.

**Fusion vs single modality** (Nadeau-Bengio corrected resampled t-test on the
25 fold AUCs) [`qol_model_comparisons.csv`]:

| Comparison | ΔAUC | p |
|---|---|---|
| All-modality early LR vs EDA random forest (best single) | +0.054 | 0.47 |
| All-modality early LR vs HR logistic regression | +0.057 | 0.31 |
| All-modality early LR vs ECG logistic regression | +0.116 | 0.037 |
| All-modality early LR vs mean-HR baseline | +0.179 | 0.0004 |
| All-modality early LR vs EDA + HR early RF | +0.003 | 0.97 |

**Fusion strategy** [`qol_fusion_strategies.png`]: for every modality pair and
the triple, late (equal log-odds pooling) and stacked fusion were never
significantly different from the best early-fusion model (all 16 comparisons
p > 0.34). Late fusion's practical advantage is robustness to a missing sensor,
not accuracy.

**Answer to RQ4.** Fusion consistently ranks at the top numerically
(+0.05 AUC over the best single modality). With 45 participants that gain is
not statistically distinguishable from zero, and it should not be claimed as
an improvement. What can be claimed is that fusion is never worse and
significantly beats ECG alone and the HR-level baseline.

**Answer to RQ1 (stress side).** Combining physiological modalities improves
on the weakest ones (ECG, HR level) but not significantly on the best one.

## 5. Sensitivity analyses [`qol_sensitivity.csv`]

| Analysis | n | Early LR | Late LR | Stacked LR |
|---|---|---|---|---|
| Primary (complete case) | 45 | 0.737 | 0.722 | 0.698 |
| All 66 participants (missing modalities imputed / skipped) | 66 | 0.685 | 0.657 | 0.666 |
| Excluding participants with inconsistent ECG timestamps | 34 | 0.700 | 0.696 | 0.655 |

Performance drops but stays well above chance when lower-quality and
incomplete participants are added. It does not depend on participants whose
ECG clock looked inconsistent. The best configuration was chosen as the maximum
of 54 cross-validated configurations, so its own estimate is optimistic
(winner's curse). The all-66 figure (0.685) is a more conservative expectation.

## 6. RQ5 - Explainability (SHAP) [`shap_summary.json`, `shap_global_*.csv`, SHAP figures]

Out-of-fold SHAP: each fold's model explains only held-out participants.

* **Best model (all-modality logistic regression, log-odds):** top features by
  mean |SHAP| were fb_hr_slope (0.053), ecg_hr_range (0.052), eda_scl_range
  (0.048), eda_scl_std (0.044) and ecg_sdnn (0.040). Fold-to-fold rank stability
  (mean pairwise Spearman) was 0.74.
* **All-modality XGBoost:** eda_scl_range, fb_hr_slope, ecg_hr_range and
  fb_hr_std lead; stability 0.64; ranking agreement with logistic regression
  rho = 0.67.
* **EDA random forest:** eda_scl_range dominates (stability 0.50).
* The features the models rely on are the ones that changed most within
  participants (Section 3.1). Variability features (range, SD, slope) matter;
  absolute levels (mean SCL, mean HR) matter little, because absolute levels
  differ more *between* people than between phases.
* Individual explanations: waterfall plots for a confidently classified stress
  observation (Q13, P = 0.73), a confidently classified relax observation (Q05,
  P = 0.36) and the most confidently wrong one (Q33 relax, P = 0.63)
  [`shap_waterfall_*.png`]. The confident stress prediction for Q13 is driven mostly
  by ECG variability (HR range 46 bpm, SDNN 175 ms within 30 s). Those values passed
  artefact QC but are at the edge of plausibility for a seated adult. ECG
  variability features may therefore partly capture movement- or speech-related
  irregularity rather than autonomic regulation alone.

**Caveats.** SHAP explains model behaviour, not physiology, and does not show
that any feature *causes* or *indicates* stress. Correlated features (e.g.
fb_hr_mean, fb_hr_p95, fb_hr_std) share credit in ways that depend on the
model family. Rankings agree only moderately across model families (rho
0.67-0.80), so feature-level claims should be limited to the robust top group.

## 7. MMASH sleep analysis (RQ1, sleep side)

**Design.** One row per participant (n = 21). Predictors come only from the
pre-sleep window; the target is that night's actigraphy sleep efficiency (SE;
mean 83.6 %, SD 6.8). Validation is leave-one-subject-out (LOSO).

**Correlations** [`mmash_correlations.csv`, `mmash_correlation_heatmap.png`]: 115
feature-target correlations; **none survived FDR correction**. The strongest
uncorrected ones were STAI-state with SE (rho = 0.53, bootstrap 95 % CI
0.05-0.85, p = 0.014, FDR p = 0.32) and with WASO (rho = -0.49). Their
direction (more state anxiety, better sleep) is counter-intuitive and
consistent with chance at this n.

**LOSO regression** [`mmash_loso_results.csv`, `mmash_permutation_tested_models.csv`, `mmash_loso_sleep_efficiency.png`]:

| Model (sleep efficiency) | MAE | R^2 | Permutation p | Holm p |
|---|---|---|---|---|
| Predict the mean (baseline) | 6.04 | -0.10 | - | - |
| Behav + Psych, random forest (lowest MAE) | 5.42 | +0.01 | 0.090 | 1.00 |
| Behav, random forest | 5.52 | +0.10 | 0.090 | 1.00 |
| Phys, Ridge (best Ridge) | 6.13 | -0.14 | 0.451 | 1.00 |
| Phys + Behav + Psych, Ridge early fusion | 6.20 | -0.19 | 0.392 | 1.00 |
| Phys + Behav + Psych, Ridge late fusion | 6.26 | -0.20 | 0.650 | 1.00 |

(The baseline's R^2 is negative because under LOSO each held-out person is
predicted by the mean of the *other* 20.) Fifteen configurations were
permutation-tested (11 Ridge with 1000 permutations, 4 random forests with 200).
None reached p < 0.05 even before correction. Secondary targets (all-modality
Ridge): TST p = 0.72, WASO p = 0.44, fragmentation index p = 0.85, PSQI p = 0.35.
The bootstrap CI of the error reduction of the lowest-MAE model spans zero
(-0.55 to +1.72 SE points).

**Answer to RQ1 (sleep side).** With 21 participants there is no evidence that
physiological, behavioural or psychological pre-sleep information, alone or
fused, predicts one night's sleep efficiency. Multimodal fusion cannot be shown
to help, and learned (weighted) fusion was deliberately not attempted. The
sleep prediction endpoint is therefore locked (HTTP 423). This is a valid
negative result for this dataset, not a failed experiment: the design rules
out the inflated results that overlapping windows or leakage would have
produced. SHAP for the all-modality Ridge model is reported only descriptively
[`shap_bar_mmash_sleep_efficiency.png`].

## 8. Deviation from the initial pipeline (reported for transparency)

The first full run computed ECG HRV after NeuroKit2's built-in artefact
correction only. Checking the figures showed 18 ECG strips with physiologically
implausible values (SDNN up to 390 ms, RMSSD up to 549 ms in 30 s), caused by
missed or spurious beats. The beat-level artefact rule already used for MMASH
was then applied to the ECG as well (RR 0.3-2.0 s, within 20 % of the local
median), plus a QC rule (strips with > 20 % rejected beats are set missing;
14 strips). The design was otherwise unchanged and every downstream step was
re-run. This reduced the complete-case sample from 57 to 45 participants.
It changed the ECG-only result from chance level (AUC 0.52) to 0.62: the
first run's conclusion that "ECG carries no information" was an artefact of
data quality. Only the corrected results are reported.

## 9. RQ6 - Common physiological patterns across datasets [`cross_dataset_*.csv`, `cross_dataset_indicators.png`]

Indicators computed identically (30-s windows): HR, ln RMSSD, SDNN.

* **State effects inside each dataset.** MMASH asleep vs awake: HR -21.9 bpm
  (21/21 participants lower), ln RMSSD +0.38 (90 % higher), both p < 1e-5:
  the textbook autonomic pattern. QoL stress vs relax: ECG HR +0.7 bpm (p = 0.73),
  ln RMSSD +0.07 (p = 0.55), SDNN +9.7 ms (p = 0.02). There was no
  vagal-withdrawal signature in the 30-s strips.
* **Distributions.** MMASH awake (ambulatory daytime) had higher HR (median 76
  vs 67 bpm; Cliff's delta 0.57) and lower ln RMSSD (3.30 vs 3.91; delta -0.56)
  than QoL seated relax. The same held when restricted to QoL men and when MMASH
  was represented by a single random 30-s window. MMASH *asleep* HRV matched QoL
  relax HRV (ln RMSSD delta 0.01, FDR p = 0.95).
* **Associations with self-report.** No HR or HRV indicator correlated with
  STAI, PSS-14 or the Daily Stress Inventory in either dataset (all FDR p > 0.9).

**Answer to RQ6.** The same indicators behave as expected for a strong
physiological state change (sleep), but barely move for the mild laboratory
stressor, and track trait/state questionnaires in neither dataset. Devices,
populations and protocols differ, so the distribution differences are
descriptive. No transfer experiment was run and **no cross-dataset
generalisation is claimed**.

## 10. External validation with STAI [`qol_external_validation.json`]

STAI was never a model input. Across the 45 participants, the change in
out-of-fold stress probability (stress - relax) was not associated with STAI
change (rho = 0.14, 95 % CI -0.19 to 0.44, p = 0.35) or PSS-14 (rho = -0.09).
The stress-phase probability showed a weak, non-significant association with
stress-phase STAI (rho = 0.26, p = 0.08). The models detect the *experimental
condition*. Within this sample, we found no evidence that they track
*how anxious a person felt*.

## 11. Limitations

1. **Fixed order.** Relax always preceded stress. Anything that drifts with
   time on the device (EDA electrode settling, posture, fatigue) is confounded
   with the label and cannot be separated in this design.
2. **Small samples.** 45 complete-case participants (QoL) and 21 (MMASH). CIs
   are wide, and differences of ±0.05 AUC between configurations are not
   resolvable.
3. **Very short recordings.** One 30-s ECG and one ~2.7-min EDA scan per phase.
   30-s SDNN/pNN50 have low reliability.
4. **Undocumented details.** ECG and EDA units, the stressor type, and the EDA
   `activation` definition are not in the release. The ECG sampling rate was
   verified from data, not documentation.
5. **Phase alignment for Fitbit HR** relies on questionnaire timestamps (same
   clock, verified). It is not a recorded phase label.
6. **Model selection optimism** from comparing 54 configurations (Section 5).
7. **Homogeneous MMASH sample** (22 young men, one night): sleep findings do
   not generalise.
8. Laboratory, consumer-grade devices; results do not transfer to free-living
   stress or to clinical populations. The system is not a medical device.

## 12. Conclusions

* A participant-independent, nested, permutation-tested pipeline gives an
  honest estimate: consumer-wearable stress detection in this protocol reaches
  AUC ≈ 0.69-0.74, clearly above chance and well below the near-perfect figures
  often reported with leaky validation.
* EDA variability and heart-rate variability/trend features carry most of the
  signal; artefact-cleaned 30-s ECG adds smaller, complementary information.
* Multimodal fusion is numerically best but not significantly better than the
  best single modality at this sample size. Late fusion matters mainly for
  robustness to missing sensors.
* MMASH (n = 21) does not support sleep prediction from pre-sleep data. The
  correct use of that dataset here is descriptive.

## 13. Reproducibility

`python -m src.pipelines.run_all` regenerates every number and figure (seeds 42-46;
library versions in `requirements.txt` and `models/registry.json`). Each served
model has a model card in `models/stress/*.json` with features,
hyper-parameters, seed, validation scheme and verdict.
