# HealthFusion learning guide

This guide explains the project from scratch, step by step, and defines every
technical term the first time it appears. Read it top to bottom once; after
that, use the glossary at the end as a reference.

---

## Part 1 - What problem are we solving?

We have two public datasets of people wearing sensors.

1. **QoL_Stress** - 66 adults sat through two short periods in one session:
   first a **relax phase**, then a **stress phase** (a task designed to make
   people feel stressed). During each phase their wrist device (a Fitbit
   Charge 6) recorded three kinds of signals. The question: *can a computer tell,
   from the signals alone, which phase a recording came from?* That is a
   **binary classification** task (two possible answers: relax or stress).

2. **MMASH** - 22 young men wore a chest heart-rate strap and an activity
   monitor for about 24 hours, including one night of sleep, and filled in
   questionnaires. The question: *do daytime signals and questionnaire answers
   relate to how well a person slept that night?* "How well" is a number (sleep
   efficiency, 0-100 %), so this is a **regression** task (predict a number).

The two datasets are **never mixed**. Different people, devices and goals make
mixing them scientifically meaningless.

---

## Part 2 - The signals (what each sensor measures)

**Heart beats and the RR interval.** Each heartbeat produces a sharp spike in
the electrical heart signal called the **R-peak**. The time between two
consecutive R-peaks is the **RR interval** (also called inter-beat interval,
**IBI**), usually 0.6-1.2 seconds at rest.

**ECG (electrocardiogram).** A recording of the heart's electrical activity.
QoL_Stress gives a 30-second ECG per phase as a list of 7,500 numbers. 7,500
numbers in 30 seconds means 250 numbers per second: the **sampling rate** (or
sampling frequency, written **fs**) is 250 **Hz** (hertz = per second). The files
do not state this, so we *verified* it: at 250 Hz the heart rate we compute
matches the device's own heart rate for 99 % of recordings; at any other
rate it is badly wrong.

**HRV (heart-rate variability).** The RR intervals are never exactly equal.
How much they vary tells us about the **autonomic nervous system** - the part
of the nervous system that runs the body automatically. Its "rest" branch
(parasympathetic, vagal) makes RR intervals vary more; its "fight-or-flight"
branch (sympathetic) tends to raise heart rate and reduce variability. We use
three standard HRV numbers:

* **SDNN** - the standard deviation of all RR intervals (overall variability), in ms.
* **RMSSD** - the root mean square of differences between *successive* RR
  intervals (beat-to-beat variability, linked to vagal activity), in ms.
  Formula: take each difference RR(i+1) - RR(i), square it, average the squares,
  take the square root.
* **pNN50** - the percentage of successive differences larger than 50 ms.

HRV from 30 seconds is called **ultra-short-term HRV**. RMSSD is fairly
reliable at that length; SDNN and pNN50 much less so. We say so in the report.

**Artefacts.** Wrong values caused by movement or poor contact (for example
an "RR interval" of 27 s because beats were missed). We remove intervals that
are physiologically impossible (outside 0.3-2.0 s) or that jump by more than
20 % from their neighbours, and we never compute a successive difference across
a removed beat or a recording gap.

**EDA (electrodermal activity).** Sweat glands respond to arousal; more sweat
makes skin conduct electricity better. The Fitbit EDA scan reports the **skin
conductance level (SCL)** about once per second and flags **EDA responses**
(short bursts). We compute the mean, spread, slope and range of SCL and the
responses per minute.

**Fitbit optical heart rate.** A light sensor estimates heart rate every ~2 s.
It is smoothed by the device, so it gives heart-rate *level and trend*, not
true beat-to-beat HRV.

**Actigraphy (MMASH).** An accelerometer on the body counts movement.
**Vector magnitude** combines the three movement axes into one number. The
same device estimates sleep and wake from movement, which produces the sleep
targets (sleep efficiency = minutes asleep / minutes in bed x 100).

**Questionnaires.** STAI (anxiety right now = *state*; anxiety in general =
*trait*), PSS-14 (stress over the past month), PSQI (sleep quality over the
past month), MEQ (morning vs evening person), PANAS (positive / negative mood),
BIS/BAS (sensitivity to punishment / reward), Daily Stress Inventory.

---

## Part 3 - From raw signals to a table (preprocessing and features)

A machine-learning model needs a **table**: one row per **observation**, one
column per **feature** (an input number), plus the **label** or **target** (the
answer we want to predict).

* QoL_Stress table: one row per participant per phase = up to 132 rows,
  15 feature columns, label = 0 (relax) or 1 (stress).
* MMASH table: one row per participant = 21 rows, 19 feature columns,
  target = sleep efficiency.

**Modality** = a group of features from one kind of source. QoL_Stress has
three (ECG, EDA, HR); MMASH has three (Physiological, Behavioural,
Psychological).

**Quality control (QC).** Before a value enters the table it must pass checks
(enough beats, detected heart rate within 10 bpm of the device's, enough valid
EDA samples, enough Fitbit samples in the window). Every decision is written to
`data/processed/*_qc_log.csv` with its reason, so nothing is dropped silently.

**Alignment.** The Fitbit heart-rate file has no "phase" column. We used the
times when participants submitted each STAI questionnaire (the end of each
phase) to cut an 8-minute window for each phase.

---

## Part 4 - Leakage: the mistake that makes results look too good

**Data leakage** means information about the answer sneaks into the inputs.
The model then looks accurate in testing but would fail in real use. Examples
we prevent:

* **Target leakage** - using STAI (filled in *at the end of each phase*, it
  basically reports how stressed the person felt) to predict the phase; using
  sleep columns to predict sleep.
* **Temporal leakage** - using something measured *after* the outcome (the
  morning-after mood questionnaire to predict the previous night's sleep).
* **Participant leakage** - the same person appearing in both training and
  test data. People have personal "fingerprints" (resting heart rate, skin
  conductance level), so a model can recognise *the person* rather than *the
  state*. We keep every person entirely on one side.
* **Preprocessing leakage** - computing averages (for filling missing values)
  or scaling factors on all data including the test part. We fit these on the
  training part only, inside a **pipeline**.

There is also a **confound** we cannot remove: in QoL_Stress the relax phase
always came first. Anything that drifts with time (sensor warming up, sitting
longer) is mixed up with "stress". We report this as a limitation.

---

## Part 5 - Testing a model honestly (validation)

**Training set / test set.** We learn from one part of the data and measure
performance on a part the model has never seen.

**Cross-validation (CV).** Split the people into k groups (**folds**). Train on
k-1 folds, test on the remaining one, and repeat so every fold is tested once.
**GroupKFold** makes the split by participant. We use 5 folds, and repeat the
whole thing 5 times with different random splits (25 test results in total)
so that one lucky split cannot mislead us.

**Leave-one-subject-out (LOSO).** The extreme case of CV: each person in turn is
the whole test set. Used for MMASH because it has only 21 people.

**Hyper-parameters** are settings chosen before training (e.g. how deep the trees
may grow). Choosing them using the test fold would be leakage, so we use
**nested CV**: an *inner* CV inside each training fold chooses them, and the
*outer* CV measures performance.

**Metrics (classification).** **Accuracy** = share of correct answers.
**Precision** = of the observations called "stress", the share that really
were stress. **Recall** = of the real stress observations, the share we found.
**F1** = the harmonic mean of precision and recall. **Macro-F1** = F1 averaged
over both classes. **Balanced accuracy** = average of recall for each class.
**Confusion matrix** = the 2x2 table of true vs predicted classes. **ROC-AUC**
= probability that a random stress observation gets a higher stress score than a
random relax observation (0.5 = guessing, 1.0 = perfect). **Pairwise accuracy**
(our addition) = for each test person, was their stress recording scored above
their own relax recording?

**Metrics (regression).** **MAE** = mean absolute error (average size of
mistakes, in the target's units). **RMSE** = root mean squared error (punishes big
mistakes more). **R^2** = how much better than always guessing the average
(0 = no better, negative = worse).

**Is it better than chance?** A **permutation test** shuffles the labels many
times, re-runs the whole evaluation each time, and asks how often the shuffled
data score as well as the real data. That fraction is the **p-value**. For
QoL_Stress we shuffle *within* each person (swap their relax/stress labels or
not) so the structure of the data is preserved.

**Is model A better than model B?** Fold scores are not independent (training
sets overlap), so a normal t-test is too optimistic. The **corrected resampled
t-test** (Nadeau and Bengio, 2003) inflates the variance to account for this.

**Multiple comparisons.** Testing 15 features at p < 0.05 would give false
positives by chance. **Benjamini-Hochberg FDR** correction controls the
expected share of false discoveries.

---

## Part 6 - The models

* **Logistic regression** - a weighted sum of features turned into a probability.
  Simple, stable, interpretable. **Ridge regression** is its regression cousin
  with a penalty that keeps weights small.
* **SVM (support vector machine) with RBF kernel** - finds a flexible boundary
  between classes.
* **Random forest** - many decision trees on random subsets, averaged.
* **Gradient boosting / XGBoost** - trees added one after another, each fixing
  the previous ones' errors.
* **Baselines** - a model that always predicts the majority/average, and a model
  using only mean heart rate. A real model must beat these.

---

## Part 7 - Fusion: combining modalities

* **Early fusion** - put all modalities' features side by side in one table and
  train one model.
* **Late fusion** - train one model per modality, then combine their outputs.
  We combine **log-odds** (the logit, log(p / (1 - p))), equally weighted.
  Advantage: if a sensor is missing, its model is simply left out.
* **Stacking (weighted late fusion)** - learn how much to trust each modality.
  Inside the training data, each modality model makes predictions for people it
  was not trained on (**out-of-fold predictions**); a small logistic model with
  non-negative weights learns the best combination. The test data never
  influence the weights.

---

## Part 8 - Explainability (SHAP)

**SHAP** (SHapley Additive exPlanations) splits one prediction into
contributions, one per feature, that add up to the model's output:
`output = base value + sum of SHAP values`. The **base value** is the average
output; a positive SHAP value pushes the prediction towards stress.

* **Global importance** = average absolute SHAP value of a feature across people.
* **Beeswarm plot** = one dot per person per feature: position = SHAP value,
  colour = feature value.
* **Waterfall plot** = the contributions for one single prediction.
* **Modality contribution** = the share of total |SHAP| belonging to each modality.

SHAP explains **what the model does**, not **what the body does**. A high SHAP
value for EDA slope means the model relies on it, not that EDA slope *causes*
or *proves* stress.

---

## Part 9 - How to run everything

```bash
pip install -r requirements.txt
python scripts/setup_data.py --mmash <mmash zip> --qol <qol_stress zip>
python -m src.pipelines.run_all          # every phase, ~40 min on 2 cores
python -m pytest -q                      # tests
uvicorn api.main:app --reload            # API at http://127.0.0.1:8000/docs
```

Then open `notebooks/` in order (01 -> 08).

---

## Glossary (alphabetical)

| Term | Meaning |
|---|---|
| Actigraphy | Measuring movement with a body-worn accelerometer |
| Artefact | A wrong measurement caused by movement, poor contact or a device glitch |
| AUC (ROC-AUC) | Probability a random positive is scored above a random negative; 0.5 = chance |
| Balanced accuracy | Mean of per-class recall; not fooled by class imbalance |
| Baseline model | A deliberately simple model that a real model must beat |
| Benjamini-Hochberg (FDR) | Correction for testing many hypotheses at once |
| Bootstrap CI | Confidence interval from re-sampling the data with replacement many times |
| Cliff's delta | Effect size for two groups: from -1 to 1, 0 = no difference |
| Confound | A variable mixed up with the one you study (here: time order and stress) |
| Cross-validation | Repeated train/test splitting so every observation is tested once |
| EDA / SCL | Electrodermal activity / skin conductance level |
| Early / late fusion | Combine features before / predictions after modelling |
| Feature | One input number per observation |
| GroupKFold | Cross-validation that keeps all rows of a person in one fold |
| HRV | Heart-rate variability: how RR intervals vary |
| Hyper-parameter | A setting chosen before training (e.g. tree depth) |
| Imputation | Filling missing values (here: the training-fold median) |
| Leakage | Information about the answer reaching the inputs or the training data |
| Log-odds (logit) | log(p / (1 - p)); turns a probability into an unbounded score |
| LOSO | Leave-one-subject-out cross-validation |
| MAE / RMSE / R^2 | Regression error measures (see Part 5) |
| Modality | A group of features from one kind of source |
| Nested CV | Inner CV for tuning inside each outer training fold |
| Permutation test | Significance test that shuffles labels to build a "no signal" reference |
| Pipeline | Chained steps (impute -> scale -> model) fitted together on training data |
| pNN50 / RMSSD / SDNN | Time-domain HRV measures (see Part 2) |
| Rank-biserial r | Effect size for the Wilcoxon test: -1 to 1 |
| RR interval / IBI | Time between two heartbeats |
| Sampling rate (fs) | Measurements per second (Hz) |
| SHAP | Additive per-feature explanation of a prediction |
| Stacking | Learning how to weight several models' predictions |
| Ultra-short HRV | HRV from less than 5 minutes of data (here 30 s) |
| Wilcoxon signed-rank test | Paired test (same person, two conditions) that does not assume normality |
