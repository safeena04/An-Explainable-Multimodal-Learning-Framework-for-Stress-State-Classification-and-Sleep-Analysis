# HealthFusion API reference

Start the server from the project root with `uvicorn api.main:app --reload` and open `http://127.0.0.1:8000/docs` for the interactive Swagger UI. Every example below was captured from the running API by `scripts/build_api_docs.py`.

Status codes: `200` OK; `404` unknown resource; `422` invalid input, incomplete modality, or a modality set whose model failed validation; `423` prediction locked because the model was not validated; `503` pipeline outputs missing.

## Service

### `GET /health`

Response `200` (lists trimmed):
```json
{
  "status": "ok",
  "stress_models": 7,
  "stress_models_served": 6,
  "sleep_model_served": false,
  "environment": {
    "python": "3.13.18",
    "numpy": "2.5.3",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.1",
    "xgboost": "3.4.1",
    "shap": "0.54.1",
    "neurokit2": "0.3.0"
  }
}
```

## Datasets

### `GET /datasets`

Response `200` (lists trimmed):
```json
{
  "datasets": [
    {
      "name": "QoL_Stress",
      "source": "PhysioNet, version 1.0.0",
      "license": "CC BY 4.0",
      "design": "66 adults, one session each, relax phase followed by stress phase (fixed order)",
      "modalities": {
        "ECG": "Fitbit Charge 6 ECG app, 30 s per phase, 250 Hz (verified)",
        "EDA": "Fitbit EDA scan, ~160 s per phase, 1 row/s (verified)",
        "HR": "Fitbit optical heart rate, irregular ~2 s spacing, no phase label"
      },
      "role_in_project": "main quantitative contribution: stress vs relax classification",
      "n_participants": 66
    },
    {
      "name": "MMASH",
      "source": "PhysioNet, version 1.0.0",
      "license": "ODbL",
      "design": "22 healthy young men, ~24 h free-living recording including one night of sleep",
      "modalities": {
        "Physiological": "Polar H7 inter-beat intervals",
        "Behavioural": "ActiGraph wGT3X-BT (1 Hz) + activity diary",
        "Psychological": "questionnaires"
      },
      "role_in_project": "participant-level sleep analysis (n = 21 with a sleep record), cautious validation",
      "n_participant_folders": 22
    }
  ]
}
```

### `GET /datasets/qol_stress/qc?only_excluded=true`

Response `200` (lists trimmed):
```json
{
  "dataset": "qol_stress",
  "n": 28,
  "records": [
    {
      "pid": "Q01",
      "phase": "relax",
      "modality": "HR",
      "included": false,
      "reason": "only 28 Fitbit samples in the 8-min window",
      "detected_hr": null,
      "device_hr": null
    },
    {
      "pid": "Q01",
      "phase": "stress",
      "modality": "EDA",
      "included": false,
      "reason": "no EDA session in release",
      "detected_hr": null,
      "device_hr": null
    },
    {
      "pid": "Q02",
      "phase": "stress",
      "modality": "ECG",
      "included": false,
      "reason": "detected HR 76.3 vs device 65 bpm",
      "detected_hr": 76.2829,
      "device_hr": 65.0
    },
    {
      "pid": "Q07",
      "phase": "stress",
      "modality": "ECG",
      "included": false,
      "reason": "31 % of RR intervals rejected as artefacts",
      "detected_hr": 68.5643,
      "device_hr": 69.0
    },
    "... 24 more"
  ]
}
```

## Stress classification (QoL_Stress)

### `GET /stress/models`

Response `200` (lists trimmed):
```json
{
  "models": [
    {
      "key": "stress_ecg_eda_hr",
      "modalities": [
        "ECG",
        "EDA",
        "HR"
      ],
      "strategy": "early",
      "algorithm": "logreg",
      "features": [
        "ecg_hr",
        "ecg_rmssd",
        "ecg_sdnn",
        "ecg_pnn50",
        "... 11 more"
      ],
      "validation": {
        "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
        "seeds": [
          42,
          43,
          44,
          45,
          "... 1 more"
        ],
        "roc_auc_mean": 0.7368,
        "roc_auc_sd": 0.1003,
        "balanced_accuracy_mean": 0.6422,
        "macro_f1_mean": 0.6315,
        "pairwise_accuracy_mean": 0.7778,
        "corrected_ttest_vs_chance": {
          "mean_diff": 0.2368,
          "t": 4.3848,
          "p_value": 0.0002
        },
        "permutation_p_value": 0.005,
        "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
        "validated": true
      },
      "served": true
    },
    {
      "key": "stress_eda_hr",
      "modalities": [
        "EDA",
        "HR"
      ],
      "strategy": "early",
      "algorithm": "random_forest",
      "features": [
        "eda_scl_mean",
        "eda_scl_std",
        "eda_scl_slope",
        "eda_scl_range",
        "... 6 more"
      ],
      "validation": {
        "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
        "seeds": [
          42,
          43,
          44,
          45,
          "... 1 more"
        ],
        "roc_auc_mean": 0.7343,
        "roc_auc_sd": 0.1004,
        "balanced_accuracy_mean": 0.64,
        "macro_f1_mean": 0.6304,
        "pairwise_accuracy_mean": 0.7911,
        "corrected_ttest_vs_chance": {
          "mean_diff": 0.2343,
          "t": 4.3326,
          "p_value": 0.0002
        },
        "permutation_p_value": null,
        "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
        "validated": true
      },
      "served": true
    },
    {
      "key": "stress_ecg_hr",
      "modalities": [
        "ECG",
        "HR"
      ],
      "strategy": "early",
      "algorithm": "logreg",
      "features": [
        "ecg_hr",
        "ecg_rmssd",
        "ecg_sdnn",
        "ecg_pnn50",
        "... 6 more"
      ],
      "validation": {
        "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
        "seeds": [
          42,
          43,
          44,
          45,
          "... 1 more"
        ],
        "roc_auc_mean": 0.6963,
        "roc_auc_sd": 0.1057,
        "balanced_accuracy_mean": 0.6311,
        "macro_f1_mean": 0.6115,
        "pairwise_accuracy_mean": 0.7067,
        "corrected_ttest_vs_chance": {
          "mean_diff": 0.1963,
          "t": 3.4498,
          "p_value": 0.0021
        },
        "permutation_p_value": null,
        "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
        "validated": true
      },
      "served": true
    },
    {
      "key": "stress_ecg_eda",
      "modalities": [
        "ECG",
        "EDA"
      ],
      "strategy": "late",
      "algorithm": "xgboost",
      "features": [
        "ecg_hr",
        "ecg_rmssd",
        "ecg_sdnn",
        "ecg_pnn50",
        "... 6 more"
      ],
      "validation": {
        "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
        "seeds": [
          42,
          43,
          44,
          45,
          "... 1 more"
        ],
        "roc_auc_mean": 0.6909,
        "roc_auc_sd": 0.136,
        "balanced_accuracy_mean": 0.6333,
        "macro_f1_mean": 0.6272,
        "pairwise_accuracy_mean": 0.6756,
        "corrected_ttest_vs_chance": {
          "mean_diff": 0.1909,
          "t": 2.6058,
          "p_value": 0.0155
        },
        "permutation_p_value": null,
        "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
        "validated": true
      },
      "served": true
    },
    "... 3 more"
  ]
}
```

### `POST /stress/predict`

Supply all features of each modality you want used. The model trained on exactly that modality set is selected.

Request body:
```json
{
  "features": {
    "ecg_hr": 72.146,
    "ecg_rmssd": 29.8115,
    "ecg_sdnn": 46.1709,
    "ecg_pnn50": 12.1212,
    "ecg_hr_range": 13.8629,
    "eda_scl_mean": 33.9948,
    "eda_scl_std": 1.0847,
    "eda_scl_slope": -1.1612,
    "eda_scl_range": 3.3105,
    "eda_resp_rate": 7.875,
    "fb_hr_mean": 92.0691,
    "fb_hr_std": 19.3969,
    "fb_hr_p05": 74.0,
    "fb_hr_p95": 140.0,
    "fb_hr_slope": 1.9629
  },
  "explain": true
}
```

Response `200` (lists trimmed):
```json
{
  "model_key": "stress_ecg_eda_hr",
  "modalities_used": [
    "ECG",
    "EDA",
    "HR"
  ],
  "ignored_partial_modalities": {},
  "strategy": "early",
  "algorithm": "logreg",
  "probability_stress": 0.5794,
  "threshold": 0.5,
  "predicted_label": "stress",
  "validation": {
    "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
    "seeds": [
      42,
      43,
      44,
      45,
      "... 1 more"
    ],
    "roc_auc_mean": 0.7368,
    "roc_auc_sd": 0.1003,
    "balanced_accuracy_mean": 0.6422,
    "macro_f1_mean": 0.6315,
    "pairwise_accuracy_mean": 0.7778,
    "corrected_ttest_vs_chance": {
      "mean_diff": 0.2368,
      "t": 4.3848,
      "p_value": 0.0002
    },
    "permutation_p_value": 0.005,
    "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
    "validated": true
  },
  "disclaimer": "Research prototype trained on a small sample of healthy adults in one laboratory protocol. Not a medical device; outputs must not be used for diagnosis or clini...",
  "explanation": {
    "method": "SHAP",
    "output_scale": "log-odds",
    "base_value": 0.0025,
    "contributions": [
      {
        "feature": "fb_hr_p95",
        "modality": "HR",
        "value": 140.0,
        "shap": 0.1972
      },
      {
        "feature": "fb_hr_std",
        "modality": "HR",
        "value": 19.3969,
        "shap": 0.1876
      },
      {
        "feature": "fb_hr_slope",
        "modality": "HR",
        "value": 1.9629,
        "shap": -0.1348
      },
      {
        "feature": "ecg_hr_range",
        "modality": "ECG",
        "value": 13.8629,
        "shap": -0.0496
      },
      "... 11 more"
    ],
    "by_modality": {
      "ECG": -0.0853,
      "EDA": 0.1131,
      "HR": 0.2901
    },
    "note": "Attributions describe the model's behaviour, not physiological causes."
  }
}
```

### `POST /stress/predict`

EDA only:

Request body:
```json
{
  "features": {
    "eda_scl_mean": 32.5418,
    "eda_scl_std": 0.9347,
    "eda_scl_slope": -0.9186,
    "eda_scl_range": 2.414,
    "eda_resp_rate": 9.0566
  },
  "explain": false
}
```

Response `200` (lists trimmed):
```json
{
  "model_key": "stress_eda",
  "modalities_used": [
    "EDA"
  ],
  "ignored_partial_modalities": {},
  "strategy": "single",
  "algorithm": "random_forest",
  "probability_stress": 0.5264,
  "threshold": 0.5,
  "predicted_label": "stress",
  "validation": {
    "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
    "seeds": [
      42,
      43,
      44,
      45,
      "... 1 more"
    ],
    "roc_auc_mean": 0.6825,
    "roc_auc_sd": 0.1334,
    "balanced_accuracy_mean": 0.6556,
    "macro_f1_mean": 0.6506,
    "pairwise_accuracy_mean": 0.7244,
    "corrected_ttest_vs_chance": {
      "mean_diff": 0.1825,
      "t": 2.5405,
      "p_value": 0.018
    },
    "permutation_p_value": 0.0392,
    "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
    "validated": true
  },
  "disclaimer": "Research prototype trained on a small sample of healthy adults in one laboratory protocol. Not a medical device; outputs must not be used for diagnosis or clini..."
}
```

### `POST /stress/predict`

ECG only - refused when the ECG-only model failed validation:

Request body:
```json
{
  "features": {
    "ecg_hr": 81.1711,
    "ecg_rmssd": 23.7885,
    "ecg_sdnn": 58.9535,
    "ecg_pnn50": 2.6316,
    "ecg_hr_range": 27.5674
  }
}
```

Response `422` (lists trimmed):
```json
{
  "detail": {
    "message": "The model for this modality set did not pass validation (its cross-validated AUC was not reliably above chance), so it is not served. Add another modality.",
    "error": "model_not_validated",
    "modalities": [
      "ECG"
    ],
    "model_key": "stress_ecg",
    "validation": {
      "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
      "seeds": [
        42,
        43,
        44,
        45,
        "... 1 more"
      ],
      "roc_auc_mean": 0.6207,
      "roc_auc_sd": 0.1185,
      "balanced_accuracy_mean": 0.64,
      "macro_f1_mean": 0.6283,
      "pairwise_accuracy_mean": 0.5956,
      "corrected_ttest_vs_chance": {
        "mean_diff": 0.1207,
        "t": 1.8921,
        "p_value": 0.0706
      },
      "permutation_p_value": 0.0149,
      "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
      "validated": false
    },
    "validated_modality_sets": [
      [
        "ECG",
        "EDA",
        "HR"
      ],
      [
        "EDA",
        "HR"
      ],
      [
        "ECG",
        "HR"
      ],
      [
        "ECG",
        "EDA"
      ],
      "... 2 more"
    ]
  }
}
```

### `POST /stress/predict/raw`

Raw signals are processed by the same code as the training pipeline (`src/preprocessing/qol_signals.py`). Body shape:
```json
{
  "ecg": {
    "samples": "[7500 numbers]",
    "sampling_rate": 250
  },
  "eda": {
    "scl": "[~160 numbers, 1 per second]",
    "activation": "[0/1 flags]",
    "valid": "[true/false, optional]",
    "sampling_rate": 1.0
  },
  "heart_rate": {
    "bpm": "[numbers]",
    "timestamps_s": "[seconds]"
  },
  "explain": true
}
```

### `POST /stress/predict/raw`

Response `200` (lists trimmed):
```json
{
  "model_key": "stress_ecg_eda_hr",
  "modalities_used": [
    "ECG",
    "EDA",
    "HR"
  ],
  "ignored_partial_modalities": {},
  "strategy": "early",
  "algorithm": "logreg",
  "probability_stress": 0.5157,
  "threshold": 0.5,
  "predicted_label": "stress",
  "validation": {
    "scheme": "5-fold participant-grouped CV x 5 repeats, nested tuning",
    "seeds": [
      42,
      43,
      44,
      45,
      "... 1 more"
    ],
    "roc_auc_mean": 0.7368,
    "roc_auc_sd": 0.1003,
    "balanced_accuracy_mean": 0.6422,
    "macro_f1_mean": 0.6315,
    "pairwise_accuracy_mean": 0.7778,
    "corrected_ttest_vs_chance": {
      "mean_diff": 0.2368,
      "t": 4.3848,
      "p_value": 0.0002
    },
    "permutation_p_value": 0.005,
    "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
    "validated": true
  },
  "disclaimer": "Research prototype trained on a small sample of healthy adults in one laboratory protocol. Not a medical device; outputs must not be used for diagnosis or clini...",
  "explanation": {
    "method": "SHAP",
    "output_scale": "log-odds",
    "base_value": 0.0025,
    "contributions": [
      {
        "feature": "ecg_hr_range",
        "modality": "ECG",
        "value": 27.5674,
        "shap": 0.0779
      },
      {
        "feature": "eda_scl_slope",
        "modality": "EDA",
        "value": -0.9186,
        "shap": 0.0299
      },
      {
        "feature": "fb_hr_slope",
        "modality": "HR",
        "value": 0.0745,
        "shap": -0.0272
      },
      {
        "feature": "fb_hr_std",
        "modality": "HR",
        "value": 3.8495,
        "shap": -0.0223
      },
      "... 11 more"
    ],
    "by_modality": {
      "ECG": 0.0764,
      "EDA": 0.0048,
      "HR": -0.0208
    },
    "note": "Attributions describe the model's behaviour, not physiological causes."
  },
  "extracted_features": {
    "ecg_hr": 81.1711,
    "ecg_rmssd": 23.7885,
    "ecg_sdnn": 58.9535,
    "ecg_pnn50": 2.6316,
    "ecg_hr_range": 27.5674,
    "eda_scl_mean": 32.5418,
    "eda_scl_std": 0.9347,
    "eda_scl_slope": -0.9186,
    "eda_scl_range": 2.414,
    "eda_resp_rate": 9.0566,
    "fb_hr_mean": 81.0469,
    "fb_hr_std": 3.8495,
    "fb_hr_p05": 76.0,
    "fb_hr_p95": 87.4,
    "fb_hr_slope": 0.0745
  },
  "signal_quality": {
    "ecg": {
      "ecg_n_beats": 40,
      "ecg_quality": 0.9041,
      "ecg_duration_s": 30.0,
      "ecg_pct_artefact": 0.0,
      "ecg_error": null
    },
    "eda": {
      "n_valid": 159,
      "n_rows": 163
    },
    "heart_rate": {
      "n_samples": 213,
      "span_min": 7.9667
    }
  }
}
```

### `GET /stress/statistics`

Response `200` (lists trimmed):
```json
{
  "manipulation_check": {
    "n": 66,
    "stai_relax_mean": 33.5152,
    "stai_stress_mean": 36.7424,
    "mean_increase": 3.2273,
    "cohen_dz": 0.5113,
    "wilcoxon_p": 0.0,
    "n_increase": 43,
    "n_equal": 11,
    "n_decrease": 12
  },
  "paired_tests": [
    {
      "feature": "ecg_hr",
      "modality": "ECG",
      "n_pairs": 51,
      "median_relax": 67.1824,
      "median_stress": 68.6261,
      "median_diff": 0.724,
      "wilcoxon_W": 626.0,
      "p_value": 0.7287,
      "rank_biserial_r": 0.0558,
      "frac_increase": 0.5294,
      "p_fdr": 0.7287
    },
    {
      "feature": "ecg_rmssd",
      "modality": "ECG",
      "n_pairs": 51,
      "median_relax": 44.4305,
      "median_stress": 41.5032,
      "median_diff": 2.4016,
      "wilcoxon_W": 597.0,
      "p_value": 0.5361,
      "rank_biserial_r": 0.0995,
      "frac_increase": 0.549,
      "p_fdr": 0.6186
    },
    {
      "feature": "ecg_sdnn",
      "modality": "ECG",
      "n_pairs": 51,
      "median_relax": 52.9983,
      "median_stress": 63.1792,
      "median_diff": 9.7109,
      "wilcoxon_W": 416.0,
      "p_value": 0.0206,
      "rank_biserial_r": 0.3725,
      "frac_increase": 0.5686,
      "p_fdr": 0.0386
    },
    {
      "feature": "ecg_pnn50",
      "modality": "ECG",
      "n_pairs": 51,
      "median_relax": 22.2222,
      "median_stress": 22.8571,
      "median_diff": 2.5641,
      "wilcoxon_W": 514.0,
      "p_value": 0.2332,
      "rank_biserial_r": 0.1937,
      "frac_increase": 0.5882,
      "p_fdr": 0.3498
    },
    "... 11 more"
  ],
  "external_validation": {
    "experiment": "ECG+EDA+HR | early | logreg",
    "delta_p_vs_stai_delta": {
      "n": 45,
      "rho": 0.143,
      "p_value": 0.3487,
      "ci_low": -0.1887,
      "ci_high": 0.4434
    },
    "delta_p_vs_pss14": {
      "n": 45,
      "rho": -0.0912,
      "p_value": 0.5515,
      "ci_low": -0.3627,
      "ci_high": 0.2143
    },
    "p_stress_vs_stai_stress": {
      "n": 45,
      "rho": 0.2616,
      "p_value": 0.0826,
      "ci_low": -0.0302,
      "ci_high": 0.4958
    },
    "note": "Out-of-fold probabilities averaged over CV repeats. STAI/PSS were never model inputs."
  }
}
```

## Sleep analysis (MMASH)

### `GET /sleep/analysis?target=sleep_efficiency`

Response `200` (lists trimmed):
```json
{
  "target": "sleep_efficiency",
  "n_participants": 21,
  "meta": {
    "n_participants": 21,
    "primary_target": "sleep_efficiency",
    "validation": "leave-one-subject-out",
    "permutations": 1000,
    "any_model_beats_baseline_primary": false,
    "weighted_fusion": "not run: fusion weights cannot be estimated reliably from 20 training participants",
    "runtime_s": 2471,
    "rf_permutations": 200
  },
  "correlations": [
    {
      "target": "sleep_efficiency",
      "feature": "psych_stai_state",
      "modality": "Psychological",
      "n": 21,
      "rho": 0.528,
      "p_value": 0.0139,
      "ci_low": 0.0491,
      "ci_high": 0.8482,
      "p_fdr": 0.3195
    },
    {
      "target": "sleep_efficiency",
      "feature": "behav_frac_standing",
      "modality": "Behavioural",
      "n": 21,
      "rho": -0.3688,
      "p_value": 0.0999,
      "ci_low": -0.6864,
      "ci_high": 0.0588,
      "p_fdr": 0.991
    },
    {
      "target": "sleep_efficiency",
      "feature": "cortisol_before_sleep",
      "modality": "Exploratory (not a predictor)",
      "n": 20,
      "rho": 0.2947,
      "p_value": 0.2071,
      "ci_low": -0.1775,
      "ci_high": 0.7141,
      "p_fdr": 0.991
    },
    {
      "target": "sleep_efficiency",
      "feature": "behav_steps_per_hour",
      "modality": "Behavioural",
      "n": 21,
      "rho": -0.274,
      "p_value": 0.2294,
      "ci_low": -0.6369,
      "ci_high": 0.1569,
      "p_fdr": 0.991
    },
    "... 19 more"
  ],
  "loso_regression": [
    {
      "target": "sleep_efficiency",
      "modalities": "Behav+Psych",
      "strategy": "early",
      "model": "random_forest",
      "n": 21,
      "n_features": 14,
      "mae": 5.4238,
      "rmse": 6.5718,
      "r2": 0.0072,
      "mae_gain_vs_mean": 0.6162,
      "mae_gain_ci_low": -0.5542,
      "mae_gain_ci_high": 1.7178,
      "perm_p_value": 0.0896,
      "perm_null_median_mae": 6.4047
    },
    {
      "target": "sleep_efficiency",
      "modalities": "Behav",
      "strategy": "single",
      "model": "random_forest",
      "n": 21,
      "n_features": 7,
      "mae": 5.5167,
      "rmse": 6.2637,
      "r2": 0.0981,
      "mae_gain_vs_mean": 0.5234,
      "mae_gain_ci_low": -0.7544,
      "mae_gain_ci_high": 1.7431,
      "perm_p_value": 0.0896,
      "perm_null_median_mae": 6.3458
    },
    {
      "target": "sleep_efficiency",
      "modalities": "Phys+Behav+Psych",
      "strategy": "early",
      "model": "random_forest",
      "n": 21,
      "n_features": 19,
      "mae": 5.6907,
      "rmse": 6.783,
      "r2": -0.0576,
      "mae_gain_vs_mean": 0.3494,
      "mae_gain_ci_low": -0.6697,
      "mae_gain_ci_high": 1.3657,
      "perm_p_value": 0.1741,
      "perm_null_median_mae": 6.4038
    },
    {
      "target": "sleep_efficiency",
      "modalities": "Phys+Behav",
      "strategy": "early",
      "model": "random_forest",
      "n": 21,
      "n_features": 12,
      "mae": 5.8149,
      "rmse": 6.6223,
      "r2": -0.0081,
      "mae_gain_vs_mean": 0.2252,
      "mae_gain_ci_low": -0.8056,
      "mae_gain_ci_high": 1.2215,
      "perm_p_value": 0.199,
      "perm_null_median_mae": 6.4088
    },
    "... 15 more"
  ]
}
```

### `POST /sleep/predict`

Request body:
```json
{
  "features": {
    "phys_hr_mean": 80
  }
}
```

Response `423` (lists trimmed):
```json
{
  "detail": {
    "message": "Sleep prediction is disabled: no model beat the predict-the-mean baseline under leave-one-subject-out validation on MMASH (n = 21). See /sleep/analysis.",
    "validation": {
      "scheme": "leave-one-subject-out",
      "mae": 5.4238,
      "rmse": 6.5718,
      "r2": 0.0072,
      "baseline_mae": 6.0401,
      "permutation_p_value": 0.0896,
      "permutation_p_holm": 1.0,
      "n_configurations_tested": 15,
      "criterion": "LOSO R2 > 0 and Holm-adjusted permutation p < 0.05 (lowest-MAE tested model)",
      "validated": false
    }
  }
}
```

## Performance, modalities, explanations, cross-dataset

### `GET /performance/stress?top=3`

Response `200` (lists trimmed):
```json
{
  "validation": "5-fold participant-grouped CV x 5 repeats (nested tuning), complete-case n = 45",
  "meta": {
    "best_overall": "ECG+EDA+HR | early | logreg",
    "best_single_modality": "EDA | single | random_forest",
    "n_participants_complete_case": 45,
    "outer_folds": 5,
    "repeats": 5,
    "seeds": [
      42,
      43,
      44,
      45,
      "... 1 more"
    ],
    "n_configurations_compared": 54,
    "selection_note": "best_overall is the highest mean outer-CV AUC among all configurations; choosing the maximum of many configurations is optimistic (winner's curse).",
    "runtime_s": 839
  },
  "summary": [
    {
      "experiment": "ECG+EDA+HR | early | logreg",
      "modalities": "ECG+EDA+HR",
      "strategy": "early",
      "model": "logreg",
      "n_participants": 45,
      "n_rows": 90,
      "accuracy_mean": 0.6422,
      "accuracy_sd": 0.104,
      "accuracy_ci_low": 0.6014,
      "accuracy_ci_high": 0.683,
      "balanced_accuracy_mean": 0.6422,
      "balanced_accuracy_sd": 0.104,
      "balanced_accuracy_ci_low": 0.6014,
      "balanced_accuracy_ci_high": 0.683,
      "precision_mean": 0.6774,
      "precision_sd": 0.1454,
      "precision_ci_low": 0.6204,
      "precision_ci_high": 0.7344,
      "recall_mean": 0.5822,
      "recall_sd": 0.2158,
      "recall_ci_low": 0.4976,
      "recall_ci_high": 0.6668,
      "f1_mean": 0.6006,
      "f1_sd": 0.156,
      "f1_ci_low": 0.5395,
      "f1_ci_high": 0.6618,
      "macro_f1_mean": 0.6315,
      "macro_f1_sd": 0.1107,
      "macro_f1_ci_low": 0.5882,
      "macro_f1_ci_high": 0.6749,
      "roc_auc_mean": 0.7368,
      "roc_auc_sd": 0.1003,
      "roc_auc_ci_low": 0.6975,
      "roc_auc_ci_high": 0.7761,
      "pairwise_accuracy_mean": 0.7778,
      "pairwise_accuracy_sd": 0.1064,
      "pairwise_accuracy_ci_low": 0.7361,
      "pairwise_accuracy_ci_high": 0.8195,
      "cm_tn": 31.6,
      "cm_fp": 13.4,
      "cm_fn": 18.8,
      "cm_tp": 26.2
    },
    {
      "experiment": "EDA+HR | early | random_forest",
      "modalities": "EDA+HR",
      "strategy": "early",
      "model": "random_forest",
      "n_participants": 45,
      "n_rows": 90,
      "accuracy_mean": 0.64,
      "accuracy_sd": 0.1053,
      "accuracy_ci_low": 0.5987,
      "accuracy_ci_high": 0.6813,
      "balanced_accuracy_mean": 0.64,
      "balanced_accuracy_sd": 0.1053,
      "balanced_accuracy_ci_low": 0.5987,
      "balanced_accuracy_ci_high": 0.6813,
      "precision_mean": 0.6288,
      "precision_sd": 0.1146,
      "precision_ci_low": 0.5839,
      "precision_ci_high": 0.6737,
      "recall_mean": 0.6444,
      "recall_sd": 0.2199,
      "recall_ci_low": 0.5582,
      "recall_ci_high": 0.7306,
      "f1_mean": 0.6247,
      "f1_sd": 0.1594,
      "f1_ci_low": 0.5622,
      "f1_ci_high": 0.6872,
      "macro_f1_mean": 0.6304,
      "macro_f1_sd": 0.1136,
      "macro_f1_ci_low": 0.5858,
      "macro_f1_ci_high": 0.6749,
      "roc_auc_mean": 0.7343,
      "roc_auc_sd": 0.1004,
      "roc_auc_ci_low": 0.695,
      "roc_auc_ci_high": 0.7737,
      "pairwise_accuracy_mean": 0.7911,
      "pairwise_accuracy_sd": 0.1255,
      "pairwise_accuracy_ci_low": 0.7419,
      "pairwise_accuracy_ci_high": 0.8403,
      "cm_tn": 28.6,
      "cm_fp": 16.4,
      "cm_fn": 16.0,
      "cm_tp": 29.0
    },
    {
      "experiment": "EDA+HR | late | xgboost",
      "modalities": "EDA+HR",
      "strategy": "late",
      "model": "xgboost",
      "n_participants": 45,
      "n_rows": 90,
      "accuracy_mean": 0.6733,
      "accuracy_sd": 0.1149,
      "accuracy_ci_low": 0.6283,
      "accuracy_ci_high": 0.7184,
      "balanced_accuracy_mean": 0.6733,
      "balanced_accuracy_sd": 0.1149,
      "balanced_accuracy_ci_low": 0.6283,
      "balanced_accuracy_ci_high": 0.7184,
      "precision_mean": 0.6807,
      "precision_sd": 0.1228,
      "precision_ci_low": 0.6326,
      "precision_ci_high": 0.7288,
      "recall_mean": 0.6711,
      "recall_sd": 0.1911,
      "recall_ci_low": 0.5962,
      "recall_ci_high": 0.746,
      "f1_mean": 0.6638,
      "f1_sd": 0.1378,
      "f1_ci_low": 0.6098,
      "f1_ci_high": 0.7178,
      "macro_f1_mean": 0.6661,
      "macro_f1_sd": 0.119,
      "macro_f1_ci_low": 0.6194,
      "macro_f1_ci_high": 0.7127,
      "roc_auc_mean": 0.7269,
      "roc_auc_sd": 0.1174,
      "roc_auc_ci_low": 0.6809,
      "roc_auc_ci_high": 0.7729,
      "pairwise_accuracy_mean": 0.7244,
      "pairwise_accuracy_sd": 0.1403,
      "pairwise_accuracy_ci_low": 0.6695,
      "pairwise_accuracy_ci_high": 0.7794,
      "cm_tn": 30.4,
      "cm_fp": 14.6,
      "cm_fn": 14.8,
      "cm_tp": 30.2
    }
  ],
  "comparisons": [
    {
      "model_a": "ECG+EDA+HR | early | logreg",
      "model_b": "EDA+HR | early | random_forest",
      "metric": "roc_auc",
      "mean_diff": 0.0025,
      "t": 0.0429,
      "p_value": 0.9661
    },
    {
      "model_a": "ECG+EDA+HR | early | logreg",
      "model_b": "ECG+HR | early | logreg",
      "metric": "roc_auc",
      "mean_diff": 0.0405,
      "t": 0.8803,
      "p_value": 0.3874
    },
    {
      "model_a": "ECG+EDA+HR | early | logreg",
      "model_b": "ECG+EDA | late | xgboost",
      "metric": "roc_auc",
      "mean_diff": 0.0459,
      "t": 0.7588,
      "p_value": 0.4554
    },
    {
      "model_a": "ECG+EDA+HR | early | logreg",
      "model_b": "EDA | single | random_forest",
      "metric": "roc_auc",
      "mean_diff": 0.0543,
      "t": 0.736,
      "p_value": 0.4689
    },
    "... 20 more"
  ],
  "permutation_tests": [
    {
      "experiment": "ECG | single | logreg",
      "observed_auc_single_repeat": 0.6346,
      "n_permutations": 200,
      "null_mean": 0.4986,
      "null_95th": 0.6002,
      "p_value": 0.0149
    },
    {
      "experiment": "EDA | single | logreg",
      "observed_auc_single_repeat": 0.6543,
      "n_permutations": 200,
      "null_mean": 0.4928,
      "null_95th": 0.6001,
      "p_value": 0.005
    },
    {
      "experiment": "EDA | single | random_forest",
      "observed_auc_single_repeat": 0.7111,
      "n_permutations": 50,
      "null_mean": 0.4912,
      "null_95th": 0.6052,
      "p_value": 0.0392
    },
    {
      "experiment": "HR | single | logreg",
      "observed_auc_single_repeat": 0.6963,
      "n_permutations": 200,
      "null_mean": 0.4988,
      "null_95th": 0.6173,
      "p_value": 0.005
    },
    "... 1 more"
  ],
  "sensitivity": [
    {
      "analysis": "available_case_all_66",
      "experiment": "ECG+EDA+HR | early | logreg",
      "n_participants": 66,
      "n_rows": 132,
      "roc_auc_mean": 0.6849,
      "roc_auc_sd": 0.0825,
      "balanced_accuracy_mean": 0.6241,
      "pairwise_accuracy_mean": 0.7415
    },
    {
      "analysis": "available_case_all_66",
      "experiment": "ECG+EDA+HR | late | logreg",
      "n_participants": 66,
      "n_rows": 132,
      "roc_auc_mean": 0.6571,
      "roc_auc_sd": 0.0811,
      "balanced_accuracy_mean": 0.6391,
      "pairwise_accuracy_mean": 0.6965
    },
    {
      "analysis": "available_case_all_66",
      "experiment": "ECG+EDA+HR | stacked | logreg",
      "n_participants": 66,
      "n_rows": 132,
      "roc_auc_mean": 0.6662,
      "roc_auc_sd": 0.079,
      "balanced_accuracy_mean": 0.6225,
      "pairwise_accuracy_mean": 0.6543
    },
    {
      "analysis": "complete_case_ecg_timestamp_consistent",
      "experiment": "ECG+EDA+HR | early | logreg",
      "n_participants": 34,
      "n_rows": 68,
      "roc_auc_mean": 0.6997,
      "roc_auc_sd": 0.0923,
      "balanced_accuracy_mean": 0.6224,
      "pairwise_accuracy_mean": 0.7543
    },
    "... 2 more"
  ]
}
```

### `GET /modalities/stress`

Response `200` (lists trimmed):
```json
{
  "task": "stress",
  "best_per_modality_set": [
    {
      "modalities": "ECG+EDA+HR",
      "experiment": "ECG+EDA+HR | early | logreg",
      "strategy": "early",
      "model": "logreg",
      "roc_auc_mean": 0.7368,
      "roc_auc_ci_low": 0.6975,
      "roc_auc_ci_high": 0.7761,
      "balanced_accuracy_mean": 0.6422,
      "macro_f1_mean": 0.6315,
      "pairwise_accuracy_mean": 0.7778
    },
    {
      "modalities": "EDA+HR",
      "experiment": "EDA+HR | early | random_forest",
      "strategy": "early",
      "model": "random_forest",
      "roc_auc_mean": 0.7343,
      "roc_auc_ci_low": 0.695,
      "roc_auc_ci_high": 0.7737,
      "balanced_accuracy_mean": 0.64,
      "macro_f1_mean": 0.6304,
      "pairwise_accuracy_mean": 0.7911
    },
    {
      "modalities": "ECG+HR",
      "experiment": "ECG+HR | early | logreg",
      "strategy": "early",
      "model": "logreg",
      "roc_auc_mean": 0.6963,
      "roc_auc_ci_low": 0.6549,
      "roc_auc_ci_high": 0.7377,
      "balanced_accuracy_mean": 0.6311,
      "macro_f1_mean": 0.6115,
      "pairwise_accuracy_mean": 0.7067
    },
    {
      "modalities": "ECG+EDA",
      "experiment": "ECG+EDA | late | xgboost",
      "strategy": "late",
      "model": "xgboost",
      "roc_auc_mean": 0.6909,
      "roc_auc_ci_low": 0.6375,
      "roc_auc_ci_high": 0.7442,
      "balanced_accuracy_mean": 0.6333,
      "macro_f1_mean": 0.6272,
      "pairwise_accuracy_mean": 0.6756
    },
    "... 3 more"
  ],
  "shap_modality_share": [
    {
      "experiment": "ECG+EDA+HR | early | logreg",
      "ECG": 0.2763,
      "EDA": 0.3936,
      "HR": 0.3301
    },
    {
      "experiment": "EDA | single | random_forest",
      "ECG": 0.0,
      "EDA": 1.0,
      "HR": 0.0
    },
    {
      "experiment": "ECG+EDA+HR | early | xgboost",
      "ECG": 0.2921,
      "EDA": 0.3773,
      "HR": 0.3306
    }
  ],
  "stacking_weights": [
    {
      "experiment": "ECG+EDA | stacked | logreg",
      "modality": "ECG",
      "weight_mean": 0.6014,
      "weight_sd": 0.1995
    },
    {
      "experiment": "ECG+EDA | stacked | logreg",
      "modality": "EDA",
      "weight_mean": 0.5055,
      "weight_sd": 0.1771
    },
    {
      "experiment": "ECG+EDA | stacked | xgboost",
      "modality": "ECG",
      "weight_mean": 0.5304,
      "weight_sd": 0.2398
    },
    {
      "experiment": "ECG+EDA | stacked | xgboost",
      "modality": "EDA",
      "weight_mean": 0.4123,
      "weight_sd": 0.1662
    },
    "... 14 more"
  ]
}
```

### `GET /explain/stress/global`

Response `200` (lists trimmed):
```json
{
  "experiment": "ECG+EDA+HR | early | logreg",
  "importance": [
    {
      "feature": "fb_hr_slope",
      "mean_abs_shap": 0.053,
      "fold_sd": 0.0366,
      "mean_shap": -0.0105,
      "modality": "HR"
    },
    {
      "feature": "ecg_hr_range",
      "mean_abs_shap": 0.0521,
      "fold_sd": 0.0113,
      "mean_shap": 0.0003,
      "modality": "ECG"
    },
    {
      "feature": "eda_scl_range",
      "mean_abs_shap": 0.0478,
      "fold_sd": 0.0077,
      "mean_shap": -0.0007,
      "modality": "EDA"
    },
    {
      "feature": "eda_scl_std",
      "mean_abs_shap": 0.0438,
      "fold_sd": 0.0134,
      "mean_shap": -0.003,
      "modality": "EDA"
    },
    "... 11 more"
  ],
  "details": {
    "output_scale": "log-odds",
    "rank_stability_mean_pairwise_spearman": 0.7393,
    "modality_share": {
      "ECG": 0.2763,
      "EDA": 0.3936,
      "HR": 0.3301
    },
    "top5": [
      {
        "feature": "fb_hr_slope",
        "mean_abs_shap": 0.053,
        "mean_shap": -0.0105
      },
      {
        "feature": "ecg_hr_range",
        "mean_abs_shap": 0.0521,
        "mean_shap": 0.0003
      },
      {
        "feature": "eda_scl_range",
        "mean_abs_shap": 0.0478,
        "mean_shap": -0.0007
      },
      {
        "feature": "eda_scl_std",
        "mean_abs_shap": 0.0438,
        "mean_shap": -0.003
      },
      "... 1 more"
    ],
    "individual_examples": {
      "confident_stress": {
        "pid": "Q13",
        "phase": "stress",
        "prob_stress": 0.7275
      },
      "confident_relax": {
        "pid": "Q05",
        "phase": "relax",
        "prob_stress": 0.3624
      },
      "misclassified": {
        "pid": "Q33",
        "phase": "relax",
        "prob_stress": 0.6332
      }
    }
  },
  "note": "out-of-fold SHAP: each value explains a held-out participant"
}
```

### `GET /explain/stress/instance/Q03/stress`

Response `200` (lists trimmed):
```json
{
  "experiment": "ECG+EDA+HR | early | logreg",
  "pid": "Q03",
  "phase": "stress",
  "prob_stress_out_of_fold": 0.5322,
  "base_value": 0.0007,
  "contributions": [
    {
      "feature": "fb_hr_p95",
      "shap": 0.1211
    },
    {
      "feature": "fb_hr_slope",
      "shap": -0.112
    },
    {
      "feature": "fb_hr_std",
      "shap": 0.1033
    },
    {
      "feature": "ecg_hr_range",
      "shap": -0.0421
    },
    "... 11 more"
  ]
}
```

### `GET /cross-dataset`

Response `200` (lists trimmed):
```json
{
  "summary": {
    "contexts_n": {
      "MMASH awake (pre-sleep)": 21,
      "MMASH asleep": 21,
      "QoL relax": 60,
      "QoL stress": 55
    },
    "state_effects": [
      {
        "dataset": "MMASH",
        "contrast": "asleep - awake",
        "metric": "hr",
        "n_pairs": 21,
        "median_change": -21.8523,
        "frac_increase": 0.0,
        "wilcoxon_p": 0.0,
        "direction": "decrease"
      },
      {
        "dataset": "MMASH",
        "contrast": "asleep - awake",
        "metric": "ln_rmssd",
        "n_pairs": 21,
        "median_change": 0.3785,
        "frac_increase": 0.9048,
        "wilcoxon_p": 0.0,
        "direction": "increase"
      },
      {
        "dataset": "MMASH",
        "contrast": "asleep - awake",
        "metric": "sdnn",
        "n_pairs": 21,
        "median_change": -4.7827,
        "frac_increase": 0.381,
        "wilcoxon_p": 0.8917,
        "direction": "decrease"
      },
      {
        "dataset": "QoL_Stress",
        "contrast": "stress - relax",
        "metric": "hr",
        "n_pairs": 51,
        "median_change": 0.724,
        "frac_increase": 0.5294,
        "wilcoxon_p": 0.7287,
        "direction": "increase"
      },
      "... 2 more"
    ],
    "n_significant_associations_fdr": 0,
    "transfer_experiment": "not performed; no cross-dataset generalisation is claimed",
    "comparability_notes": [
      "Different devices: Polar H7 chest strap (MMASH) vs Fitbit Charge 6 finger ECG (QoL).",
      "Different populations: 22 young men (MMASH) vs 66 adults, 27 women (QoL).",
      "MMASH participant values are medians of hundreds of 30-s windows; QoL values are single 30-s windows. The *_single columns use one random MMASH window to match ..."
    ]
  },
  "state_effects": [
    {
      "dataset": "MMASH",
      "contrast": "asleep - awake",
      "metric": "hr",
      "n_pairs": 21,
      "median_change": -21.8523,
      "frac_increase": 0.0,
      "wilcoxon_p": 0.0,
      "direction": "decrease"
    },
    {
      "dataset": "MMASH",
      "contrast": "asleep - awake",
      "metric": "ln_rmssd",
      "n_pairs": 21,
      "median_change": 0.3785,
      "frac_increase": 0.9048,
      "wilcoxon_p": 0.0,
      "direction": "increase"
    },
    {
      "dataset": "MMASH",
      "contrast": "asleep - awake",
      "metric": "sdnn",
      "n_pairs": 21,
      "median_change": -4.7827,
      "frac_increase": 0.381,
      "wilcoxon_p": 0.8917,
      "direction": "decrease"
    },
    {
      "dataset": "QoL_Stress",
      "contrast": "stress - relax",
      "metric": "hr",
      "n_pairs": 51,
      "median_change": 0.724,
      "frac_increase": 0.5294,
      "wilcoxon_p": 0.7287,
      "direction": "increase"
    },
    "... 2 more"
  ],
  "distributions": [
    {
      "context_a": "MMASH awake (pre-sleep)",
      "context_b": "QoL relax",
      "metric": "hr",
      "qol_subset": "all",
      "n_a": 21,
      "n_b": 60,
      "median_a": 76.322,
      "median_b": 67.3842,
      "mannwhitney_p": 0.0001,
      "cliffs_delta": 0.5698,
      "p_fdr": 0.0005
    },
    {
      "context_a": "MMASH awake (pre-sleep)",
      "context_b": "QoL relax",
      "metric": "hr",
      "qol_subset": "QoL males only",
      "n_a": 21,
      "n_b": 35,
      "median_a": 76.322,
      "median_b": 65.92,
      "mannwhitney_p": 0.0001,
      "cliffs_delta": 0.6517,
      "p_fdr": 0.0004
    },
    {
      "context_a": "MMASH awake (pre-sleep)",
      "context_b": "QoL relax",
      "metric": "ln_rmssd",
      "qol_subset": "all",
      "n_a": 21,
      "n_b": 60,
      "median_a": 3.2979,
      "median_b": 3.9138,
      "mannwhitney_p": 0.0002,
      "cliffs_delta": -0.5587,
      "p_fdr": 0.0005
    },
    {
      "context_a": "MMASH awake (pre-sleep)",
      "context_b": "QoL relax",
      "metric": "ln_rmssd",
      "qol_subset": "QoL males only",
      "n_a": 21,
      "n_b": 35,
      "median_a": 3.2979,
      "median_b": 3.925,
      "mannwhitney_p": 0.0005,
      "cliffs_delta": -0.5646,
      "p_fdr": 0.0009
    },
    "... 26 more"
  ],
  "associations": [
    {
      "dataset": "MMASH",
      "indicator": "hr (awake)",
      "measure": "STAI state",
      "n": 21,
      "rho": -0.013,
      "p_value": 0.9554,
      "ci_low": -0.4257,
      "ci_high": 0.4155,
      "p_fdr": 0.9554
    },
    {
      "dataset": "MMASH",
      "indicator": "ln_rmssd (awake)",
      "measure": "STAI state",
      "n": 21,
      "rho": 0.223,
      "p_value": 0.3312,
      "ci_low": -0.2626,
      "ci_high": 0.62,
      "p_fdr": 0.9273
    },
    {
      "dataset": "MMASH",
      "indicator": "hr (awake)",
      "measure": "STAI trait",
      "n": 20,
      "rho": 0.0467,
      "p_value": 0.8449,
      "ci_low": -0.4323,
      "ci_high": 0.4973,
      "p_fdr": 0.9554
    },
    {
      "dataset": "MMASH",
      "indicator": "ln_rmssd (awake)",
      "measure": "STAI trait",
      "n": 20,
      "rho": 0.2743,
      "p_value": 0.2419,
      "ci_low": -0.1679,
      "ci_high": 0.6829,
      "p_fdr": 0.9273
    },
    "... 10 more"
  ]
}
```
