# HealthFusion

**An explainable multimodal learning framework for stress-state and sleep analysis**
on two public PhysioNet datasets: QoL_Stress (stress vs relax classification, the
main quantitative contribution) and MMASH (participant-level sleep analysis).

> New to any of the terms below? Read **[docs/GUIDE.md](docs/GUIDE.md)** first: it
> explains the whole project step by step and defines every technical term.

<!-- RESULTS:START -->
## Headline results (all numbers from `results/metrics/`)

**QoL_Stress: stress vs relax** (45 complete-case participants; 5-fold participant-grouped CV x 5 repeats, nested tuning)

| Modality set | Best configuration | ROC-AUC (95 % CI) | Balanced acc. | Pairwise acc. | Served by API |
|---|---|---|---|---|---|
| ECG + EDA + HR | early fusion, logistic regression | **0.737** (0.697-0.776) | 0.642 | 0.778 | yes |
| EDA + HR | early fusion, random forest | 0.734 (0.695-0.774) | 0.640 | 0.791 | yes |
| ECG + HR | early fusion, logistic regression | 0.696 (0.655-0.738) | 0.631 | 0.707 | yes |
| ECG + EDA | late fusion, XGBoost | 0.691 (0.638-0.744) | 0.633 | 0.676 | yes |
| EDA | random forest | 0.682 (0.630-0.735) | 0.656 | 0.724 | yes |
| HR | logistic regression | 0.680 (0.644-0.715) | 0.647 | 0.676 | yes |
| ECG | logistic regression | 0.621 (0.574-0.667) | 0.640 | 0.596 | **no** (fails validation) |
| Baseline | mean heart rate only | 0.558 | 0.542 | 0.689 | - |

* Better than chance: within-participant permutation p = 0.005 (best model).
* Fusion vs best single modality (EDA): +0.054 AUC, corrected resampled t-test
  p = 0.47, so **not a significant improvement**. Fusion vs HR-level baseline:
  p = 0.0004.
* SHAP modality share (best model): EDA 39 %, HR 33 %, ECG 28 %. The top
  features are variability/trend features (HR slope, ECG HR range, SCL range and SD).
* Model stress probability does not track self-reported anxiety change (rho = 0.14, p = 0.35).
* Sensitivity: all 66 participants gives AUC 0.685.

**MMASH: sleep efficiency** (21 participants, leave-one-subject-out): no model beat
the predict-the-mean baseline (15 permutation-tested configurations, all p > 0.05;
Holm-adjusted p = 1.0), and no feature-outcome correlation survived FDR. The sleep
prediction endpoint is therefore **locked (HTTP 423)** and MMASH is reported
descriptively.

Full write-up: **[results/reports/02_final_research_report.md](results/reports/02_final_research_report.md)**.
Phase 1 data audit: **[results/reports/01_phase1_feasibility_report.md](results/reports/01_phase1_feasibility_report.md)**.
<!-- RESULTS:END -->

> **Notes on these numbers.** Rows are the models served by the API (see `models/stress/*.json`).
> The 95 % intervals are mean ± 1.96·SD/√25 over 25 overlapping cross-validation folds and are
> optimistic; a participant-level bootstrap of the pooled predictions gives roughly 0.60–0.77
> for the best model. The best configuration was chosen as the maximum of 54 configurations on the
> same folds, so 0.737 is optimistic. XGBoost-based results can differ slightly between runs.

---

## Project structure

```
healthfusion/
├── data/
│   ├── raw/                 # extracted PhysioNet archives (created by scripts/setup_data.py; not shipped)
│   ├── processed/           # feature tables, QC logs, feature dictionary
│   └── sample/              # real example API payloads (one participant)
├── notebooks/               # 01-08: readable, executed views over src/ and results/
├── src/
│   ├── config.py            # every constant, seed and verified sampling rate
│   ├── data/                # loaders + Phase 1 inspection
│   ├── preprocessing/       # QoL signals (ECG/EDA/HR), QoL + MMASH pipelines
│   ├── features/            # HRV maths, feature dictionary
│   ├── models/              # model zoo, model registry (model cards)
│   ├── fusion/              # early / late / stacked fusion
│   ├── evaluation/          # nested grouped CV, metrics, statistics
│   ├── explainability/      # SHAP (out-of-fold, fusion-aware)
│   ├── visualization/       # paper figures
│   └── pipelines/           # one runnable module per phase + run_all
├── api/                     # FastAPI backend (main.py, routes/, services.py, schemas.py)
├── models/                  # trained models (.joblib) + model cards (.json) + registry.json
├── results/
│   ├── figures/             # PNG (300 dpi) + PDF
│   ├── metrics/             # every number reported anywhere (CSV/JSON)
│   └── reports/             # Phase 1 feasibility report, final research report, run logs
├── docs/                    # GUIDE, RESEARCH_DESIGN, FEATURE_DICTIONARY, API
├── scripts/                 # setup_data, make_sample, build_notebooks
├── tests/                   # pytest suite (unit, leakage guards, data contracts, API)
└── requirements.txt
```

## Quick start

```bash
# 1. environment (Python 3.11+; results were produced with 3.13)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. data: point the script at the two PhysioNet zip files
python scripts/setup_data.py --mmash mmash-1.0.0.zip \
    --qol qol_stress-a-multimodal-dataset-of-physiological-and-self-reported-stress-responses-1.0.0.zip

# 3. reproduce everything (about 40 min on 2 CPU cores; --quick for a smoke run)
python -m src.pipelines.run_all

# 4. tests
python -m pytest -q

# 5. API
uvicorn api.main:app --reload        # open http://127.0.0.1:8000/docs
```

The shipped `results/`, `models/` and `data/processed/` already contain the
outputs of step 3, so steps 4-5 work without re-running the experiments
(raw data are still needed to rebuild them).

### Running one phase at a time

| Phase | Command | Main outputs |
|---|---|---|
| 1 Inspection | `python -m src.pipelines.run_phase1` | `results/reports/phase1_inspection.json` |
| 3-4 Preprocessing + features | `python -m src.pipelines.run_preprocessing` | `data/processed/*`, `docs/FEATURE_DICTIONARY.md` |
| 5-7 QoL models, fusion, validation | `python -m src.pipelines.run_qol_experiments` | `results/metrics/qol_*` |
| MMASH sleep analysis | `python -m src.pipelines.run_mmash_experiments` | `results/metrics/mmash_*` |
| 8 Explainability | `python -m src.pipelines.run_explainability` | `results/metrics/shap_*`, SHAP figures |
| 9 Cross-dataset | `python -m src.pipelines.run_cross_dataset` | `results/metrics/cross_dataset_*` |
| Final models | `python -m src.pipelines.train_final_models` | `models/` |
| 11 Figures | `python -m src.pipelines.make_figures` | `results/figures/` |
| Notebooks | `python scripts/build_notebooks.py --execute` | `notebooks/*.ipynb` |

## API overview

Full reference with request/response examples: **[docs/API.md](docs/API.md)**.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, number of served models, library versions |
| GET | `/datasets`, `/datasets/{qol_stress,mmash}`, `/datasets/{name}/qc` | dataset facts, Phase 1 inspection, QC decisions |
| GET | `/stress/models`, `/stress/models/{key}` | registered stress models and model cards |
| POST | `/stress/predict` | classify from phase-level features (+ SHAP explanation) |
| POST | `/stress/predict/raw` | classify from raw ECG / EDA / heart-rate signals |
| GET | `/stress/statistics` | paired tests, manipulation check, STAI external validation |
| GET | `/sleep/analysis`, `/sleep/model` | MMASH correlations, LOSO results, model card |
| POST | `/sleep/predict` | **locked (HTTP 423)** unless a sleep model passed validation |
| GET | `/performance/stress`, `/performance/stress/folds`, `/performance/sleep` | all CV metrics, comparisons, permutation and sensitivity tests |
| GET | `/modalities/{stress,sleep}` | modality comparison, stacking weights, SHAP modality shares |
| GET | `/explain/stress/global`, `/explain/stress/instance/{pid}/{phase}`, `/explain/sleep/global` | SHAP |
| GET | `/cross-dataset` | common-indicator comparison |
| GET | `/figures/{name}.png` | every paper figure |

The stress endpoint picks the trained model for **exactly the modalities you
send** (ECG, EDA, HR or any combination), so a missing sensor is never filled in
with invented values. A modality set whose model failed validation is refused
with HTTP 422 and the evidence.

## Environment used for the reported results

* OS: Windows
* RAM: 8 GB
* GPU: none used
* Python 3.13, libraries pinned in `requirements.txt`
* Random seeds: 42–46 (see `src/config.py`)

## Research-quality safeguards

* No synthetic participants, no overlapping windows, no fabricated values.
  Synthetic signals appear only in unit tests of the signal-processing code.
* Every exclusion is logged with its reason (`data/processed/*_qc_log.csv`).
* Participant-grouped splits are asserted on every fold; imputation, scaling,
  tuning and fusion weights are fitted inside training folds only.
* Leakage guards are unit-tested (`tests/test_leakage_and_contracts.py`).
* Seeds, hyper-parameters, library versions and validation verdicts are stored
  in each model card (`models/*/*.json`, `models/registry.json`).
* SHAP is reported as model behaviour, never as causation.

## Data licences and citation

* QoL_Stress: Calvo A., Cejudo A., Martin C. *QoL_Stress: a multimodal dataset of
  physiological and self-reported stress responses* (v1.0.0). PhysioNet. CC BY 4.0.
  Download: https://physionet.org/content/qol-stress/1.0.0/
* MMASH: Rossi A. et al. *Multilevel Monitoring of Activity and Sleep in Healthy
  People* (v1.0.0). PhysioNet. ODbL.
  Download: https://physionet.org/content/mmash/1.0.0/
* Cite PhysioNet: Goldberger A. et al. Circulation 101(23):e215-e220 (2000).

The raw data are not redistributed in this repository; download the two zip files from the
links above and run `python scripts/setup_data.py` (see Quick start). `data/sample/` contains one
QoL_Stress participant's recording, shared under the dataset's CC BY 4.0 licence.