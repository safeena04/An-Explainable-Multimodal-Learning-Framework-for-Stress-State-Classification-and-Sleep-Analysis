# Phase 1 - Dataset inspection and feasibility report

All numbers are produced by `python -m src.pipelines.run_phase1`
(`results/reports/phase1_inspection.json`) and the preprocessing QC logs in
`data/processed/`. Nothing in this report is estimated or assumed.

---

## 1. Dataset structure

### QoL_Stress (PhysioNet 1.0.0, CC BY 4.0)

| File | Rows | Participants | Key columns | Missing values | Exact duplicate rows |
|---|---|---|---|---|---|
| participants.csv | 66 | 66 | username, sex, age, education_level, job_category | 0 | 0 |
| ecg.csv | 132 | 66 | reading_time, heart_rate, result_classification, phase, waveform_samples | 0 | 0 |
| eda.csv | 20,094 | 64 | phase, time_sec, valid_data, activation, scl_avg | 0 | 1,542 (genuine repeated readings; kept) |
| fitbit_heart_rate_per_second.csv | 54,462 | 66 | time, value | 0 | 0 |
| fitbit_heart_rate_per_minute.csv | 2,131 | 66 | time, value | 0 | 0 |
| fitbit_heart_rate_summary.csv | 264 | 66 | zone_name, min, max, minutes, caloriesOut | 0 | 0 |
| pss_14.csv | 66 | 66 | time, score, 14 items | 0 | 0 |
| stai.csv | 132 | 66 | time, phase, score, 20 items | 0 | 0 |

Demographics: 39 men, 27 women; age 22-53 (median 27).

### MMASH (PhysioNet 1.0.0, ODbL)

22 participant folders, all men, age 20-40 (median 27; user 18 has age = 0, set
to missing). Every folder has `user_info`, `questionnaire`, `sleep`, `Activity`,
`Actigraph` and `RR`; `saliva.csv` is absent for user 21.

| File | Rows (per participant) | Notes |
|---|---|---|
| RR.csv | 62,669-102,340 beats, ~24 h | Polar H7 inter-beat intervals, 1-s timestamps |
| Actigraph.csv | 58,616-71,045 | ActiGraph wGT3X-BT, 1 row/s with recurrent 7-s gaps |
| Activity.csv | 7-35 diary entries | self-reported activity codes 1-12 |
| sleep.csv | 0-2 episodes | actigraphy-scored sleep metrics |
| questionnaire.csv | 1 | MEQ, STAI1/2, PSQI, Daily stress, BIS/BAS, PANAS x 5 times |
| saliva.csv | 2 (before sleep, wake up) | cortisol, melatonin |

---

## 2. Participant and record counts that the analysis can actually use

| | QoL_Stress | MMASH |
|---|---|---|
| Participants in release | 66 | 22 |
| ECG records | 132 (2 per person); 115 pass QC (see Section 5) | - |
| EDA sessions | 124 (60 people both phases, 4 relax only, 2 none) | - |
| Fitbit HR phase windows | 129 of 132 pass QC | - |
| All 3 modalities in both phases (complete case) | **45 participants, 90 rows** | - |
| Participants with a sleep record | - | **21** (user 11 has none) |

---

## 3. Available modalities

* **QoL_Stress:** ECG (30-s strip per phase), EDA (~160-s scan per phase),
  Fitbit optical heart rate (continuous, no phase label), questionnaires (STAI
  per phase, PSS-14 once), demographics. The Fitbit zone summary is a whole-day
  aggregate and is not phase-specific.
* **MMASH:** physiological (RR intervals, ~24 h), behavioural (actigraphy +
  activity diary), psychological (questionnaires), biochemical (saliva, 21 people).

## 4. Available prediction targets

* **QoL_Stress:** `phase` (relax / stress), balanced by design (66 / 66).
  STAI rises from relax to stress (mean 33.5 -> 36.7; Wilcoxon p = 2.6e-5;
  Cohen's dz = 0.51), but only 43 of 66 participants report an increase (11 no
  change, 12 a decrease). The stress induction worked on average but was mild
  and inconsistent - this caps achievable classification performance.
* **MMASH:** sleep efficiency (73.5-94.2 %, mean 83.6), TST (236-578 min), WASO
  (21-118 min), sleep fragmentation index (6.7-45.5), PSQI (2-9). n = 21 each.

## 5. Missing-value and integrity analysis

| Issue | Dataset | Handling |
|---|---|---|
| Two ECG recordings are byte-identical under Q15 and Q48 (same reading_id) | QoL | Timestamps fit Q15's session; shifted exactly 4 h for Q48, whose Fitbit data also end before its relax phase. Kept for Q15, set missing for Q48 |
| 18 ECG waveforms contain one split token (`- 39`) | QoL | merged into one missing sample, linearly interpolated (1 of 7,500 samples) |
| 1 ECG timestamp in a different format (`Tue Dec 17 13:01:42`) | QoL | time of day parsed |
| 1 ECG strip fails QC (detected vs device HR differ by >10 bpm) | QoL | ECG missing for that phase (logged) |
| 14 ECG strips have > 20 % of RR intervals rejected as artefacts (missed/spurious beats left after NeuroKit2 correction; before filtering, 18 strips showed implausible SDNN up to 390 ms in 30 s) | QoL | the MMASH beat-level rule is applied to ECG too; strips above 20 % rejected beats set missing (logged) |
| EDA absent for Q15, Q44 (both phases) and Q01, Q36, Q37, Q61 (stress) | QoL | EDA missing (logged) |
| 5 EDA sessions have a ~-58 min `time_sec` offset | QoL | row order (verified 1 row/s) used as time axis |
| Fitbit window too sparse: Q01 relax (28 samples), Q48 both phases (0) | QoL | HR missing (logged) |
| Day coded -29 instead of 2 (users 8, 9) | MMASH | recoded to 2 (chronology verified) |
| No sleep record (user 11) | MMASH | excluded from sleep analyses (logged) |
| STAI2 = 0 (user 7), Age = 0 (user 18) | MMASH | impossible values -> missing |
| PANAS at 14 h missing (user 13) | MMASH | mean of available PANAS times |
| Activity code 0 (34 rows, undocumented), 2 entries without end time | MMASH | excluded from diary features (counted) |
| BIS/BAS Reward > 20 for 8 participants | MMASH | consistent with a scoring variant, not entry errors: left unchanged |
| Melatonin = 0 (user 16, wake-up sample) | MMASH | not used as a predictor |

## 6. Sampling and timestamp analysis

| Signal | Stated rate | Verified from data |
|---|---|---|
| QoL ECG | not stated | 7,488-7,501 samples per record; **250 Hz**: R-peak HR matches device HR within 5 bpm for 99.2 % of records (125 Hz: 13.6 %; 300/500 Hz: 0 %) |
| QoL EDA | not stated | **1 row/s**; `time_sec` rounded to ~5-s blocks |
| QoL Fitbit HR "per second" | per second | irregular: median 2 s (1 s: 16 %, 2 s: 51 %, 3 s: 32 %) |
| MMASH RR | beat-to-beat | 1-s timestamps; a **recurring ~8-s dropout** (user 1: 2,337 gaps) caps beat coverage at ~80 % per 5-min segment |
| MMASH Actigraph | 1 Hz | 1-s rows plus recurrent 7-s gaps |

Timestamps are time-of-day only in QoL_Stress (no dates), mixed `HH:MM:SS` /
`H:MM:SS`. MMASH uses a day index (1/2) + time of day.

## 7. Participant-level alignment (QoL_Stress)

Relative to the relax-STAI submission (minute 0), the median session is: PSS at
-12.6 min, relax ECG at -5.8 min, stress ECG at +6.2 min, stress STAI at +11.9
min; the Fitbit stream runs from -19.8 to +12.1 min. Relax always precedes
stress (66/66). Fitbit HR is therefore aligned by taking the 8 minutes that end
at each phase's STAI submission. The ECG phase labels come from the file
itself; for 11 relax and 8 stress ECG strips the timestamp lies outside the
expected window (possible clock offsets) - a sensitivity analysis excludes these
participants.

## 8. Data-leakage risks

| Risk | Where | Control |
|---|---|---|
| Self-report of the label | STAI per phase | never an input; external validation only |
| Participant identity | 2 rows per person | participant-grouped CV, asserted on every split |
| Fixed phase order | relax always first | absolute times never used; acknowledged as an unremovable confound |
| Target computed from inputs | MMASH sleep metrics from actigraphy | only pre-sleep data as inputs |
| Post-outcome measures | PANAS 9+1, wake-up saliva | excluded |
| Self-reported sleep efficiency | PSQI component | PSQI excluded as predictor |
| Preprocessing on all data | imputation, scaling, tuning, fusion weights | fitted inside training folds |
| Inflated n | overlapping windows | not used; one row per participant-phase / participant |

## 9. Feasibility of the sleep analysis (MMASH)

**Supervised multimodal prediction is not statistically meaningful at n = 21.**
A 5-fold test fold would hold 4 people; 19 predictors exceed the sample size;
the night-to-night variability of one actigraphy-scored night is large. The
defensible design is **participant-level statistical modelling**: correlations
with bootstrap CIs and FDR control, plus *exploratory* LOSO regression tested
against a mean baseline with permutation tests. Weighted fusion is not
estimable and is not run.

## 10. Feasibility of stress classification (QoL_Stress)

Feasible, with caveats. 45-66 participants in a balanced, paired design support
participant-independent CV. Caveats: (i) one 30-s ECG and one ~2.7-min EDA scan
per phase - very little signal per observation; (ii) a mild, inconsistent
stress response (Section 4); (iii) the fixed relax -> stress order.

## 11. Recommended validation strategies

* QoL: 5-fold `GroupKFold` by participant x 5 repeats, nested 3-fold inner CV,
  within-participant label permutation tests, corrected resampled t-tests.
* MMASH: leave-one-subject-out, label permutation (1000), bootstrap CIs.

## 12. Recommended baseline models

Chance (prior), single-feature logistic regression on mean HR, logistic
regression, RBF-SVM, random forest, XGBoost, gradient boosting (QoL);
predict-the-mean, Ridge, shallow random forest (MMASH). No deep learning.

## 13. Is multimodal fusion statistically defensible?

* **QoL_Stress: yes**, as a controlled comparison of 7 modality subsets on the
  same 45 participants with early, late and stacked fusion (66 in a sensitivity analysis).
* **MMASH: no** for learned (weighted) fusion; early and equal-weight late
  fusion are reported only as exploratory LOSO comparisons.

## 14. Recommended research contribution

A rigorously validated, explainable comparison of ECG, EDA and heart-rate
modalities (and their fusion) for laboratory stress detection with consumer
wearables, with participant-independent nested validation and honest effect
sizes; complemented by a cautious, participant-level sleep analysis that shows
what n = 21 can and cannot support, and a descriptive cross-dataset comparison
of comparable HR/HRV indicators.
