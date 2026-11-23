"""Generate (and optionally execute) the 8 research notebooks in notebooks/.

The notebooks are thin, readable views over the tested code in src/ and the
saved outputs in results/ - no analysis logic lives only in a notebook.

Run:  python scripts/build_notebooks.py --execute
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)
M = ROOT / 'results' / 'metrics'; F = ROOT / 'results' / 'figures'; P = ROOT / 'data' / 'processed'
"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


NOTEBOOKS = {
    "01_mmash_exploration.ipynb": [
        md("# 01 - MMASH exploration (Phase 1)\n\nWhat is in the 22 participant folders, what is missing, and what "
           "is odd. All numbers come from `src/data/inspection.py` (re-run with `python -m src.pipelines.run_phase1`)."),
        code(SETUP + "insp = json.load(open(ROOT/'results/reports/phase1_inspection.json'))['mmash']"),
        md("## Rows per file per participant\n0 means the file is absent."),
        code("pd.DataFrame(insp['rows_per_file_per_user']).T"),
        md("## Documented load repairs and missing files"),
        code("print('\\n'.join(insp['load_issues'])); insp['files_missing']"),
        md("## Inter-beat intervals: range and artefact share per participant\n"
           "Values outside 0.3-2.0 s (30-200 bpm) are physiologically implausible."),
        code("pd.DataFrame(insp['rr_per_user'])"),
        md("## Sleep episodes, questionnaire validity, diary codes"),
        code("display(pd.DataFrame(insp['sleep_episodes']))\n"
             "display(pd.DataFrame(insp['questionnaire_out_of_range']))\n"
             "print('diary codes:', insp['activity_codes'], '| undocumented code 0 rows:', "
             "insp['activity_undocumented_code0_rows'])"),
        md("**Take-away.** 22 men, one night each; user 11 has no sleep record, so at most 21 participants can "
           "carry a sleep target. That sample size decides the whole MMASH design (participant-level, LOSO)."),
    ],
    "02_qol_stress_exploration.ipynb": [
        md("# 02 - QoL_Stress exploration (Phase 1)"),
        code(SETUP + "insp = json.load(open(ROOT/'results/reports/phase1_inspection.json'))['qol_stress']"),
        md("## Files, rows, missing values"),
        code("pd.DataFrame({k: {kk: v[kk] for kk in ('n_rows','n_columns','exact_duplicate_rows')} | "
             "{'participants': v.get('n_participants')} for k, v in insp['files'].items()}).T"),
        md("## Verifying the ECG sampling rate from the data\nThe release does not state it. If the rate is right, the "
           "heart rate computed from detected R-peaks must match the device's own `heart_rate` field."),
        code("from src.data.qol_loader import load_qol, parse_waveform\nfrom src.preprocessing.qol_signals import "
             "process_ecg\nd = load_qol(); rows = []\nfor _, r in d['ecg'].head(20).iterrows():\n"
             "    x = parse_waveform(r.waveform_samples).samples\n"
             "    rows.append({'device_hr': r.heart_rate, **{f'hr_at_{fs}Hz': process_ecg(x, fs)['ecg_hr'] "
             "for fs in (125, 250, 500)}})\npd.DataFrame(rows).round(1)"),
        md("## Session timeline (minutes relative to the relax-phase STAI)\nFitbit HR has no phase column; this "
           "timeline is what the phase windows are anchored on."),
        code("pd.DataFrame(insp['timeline_minutes_relative_to_stai_relax'])"),
        code("insp['ecg_time_consistency']"),
        md("## Duplicated ECG recordings and EDA availability"),
        code("display(insp['ecg']['duplicated_reading_ids']); insp['eda_sessions']"),
        md("## Manipulation check"),
        code("display(Image(F/'qol_stai_manipulation.png', width=380)); "
             "json.load(open(M/'qol_manipulation_check.json'))"),
    ],
    "03_mmash_preprocessing.ipynb": [
        md("# 03 - MMASH preprocessing\n\nPre-sleep (prospective) window, HRV with gap-aware successive "
           "differences, actigraphy and diary features, questionnaire cleaning. Code: "
           "`src/preprocessing/mmash_preprocess.py`."),
        code(SETUP + "from src.preprocessing.mmash_preprocess import build_mmash, MMASH_MODALITIES, TARGETS\n"
             "out = build_mmash(save=False)"),
        md("## QC log - every inclusion decision"),
        code("out['qc']"),
        md("## Participant-level features by modality"),
        code("for m, cols in MMASH_MODALITIES.items():\n    display(Markdown(f'### {m}'))\n"
             "    display(out['features'][['user'] + cols].round(2))"),
        md("## Targets"),
        code("out['features'][['user'] + TARGETS].round(2).describe()"),
    ],
    "04_qol_stress_preprocessing.ipynb": [
        md("# 04 - QoL_Stress preprocessing\n\nECG decoding + NeuroKit2, EDA cleaning, Fitbit phase windows, QC. "
           "Code: `src/preprocessing/qol_preprocess.py`."),
        code(SETUP + "from src.preprocessing.qol_preprocess import build_qol, MODALITIES\nout = build_qol(save=False)\n"
             "f = out['features']"),
        md("## Every exclusion, with its reason"),
        code("out['qc'].query('included == False')"),
        md("## Modality availability (rows with the modality, and participants with it in both phases)"),
        code("pd.DataFrame({m: {'rows': int(f[f'has_{m}'].sum()), 'participants_both_phases': "
             "int(f.loc[f[f'pair_{m}'], 'pid'].nunique())} for m in MODALITIES}).T"),
        code("print('complete-case participants:', f.loc[f.complete_case, 'pid'].nunique())"),
        md("## Phase medians"),
        code("f.groupby('phase')[[c for cols in MODALITIES.values() for c in cols]].median().T.round(2)"),
    ],
    "05_feature_engineering.ipynb": [
        md("# 05 - Feature engineering\n\nThe documented feature dictionary and the within-participant statistics "
           "for each feature."),
        code(SETUP + "pd.read_csv(P/'feature_dictionary.csv')"),
        md("## Relax vs stress: paired Wilcoxon tests (BH-FDR corrected)"),
        code("pd.read_csv(M/'qol_paired_tests.csv').round(4)"),
        code("display(Image(F/'qol_paired_effects.png', width=650))"),
        md("## MMASH: feature-target correlations (exploratory, n = 21)"),
        code("c = pd.read_csv(M/'mmash_correlations.csv'); c.sort_values('p_value').head(15).round(3)"),
        code("display(Image(F/'mmash_correlation_heatmap.png', width=600))"),
    ],
    "06_baseline_models.ipynb": [
        md("# 06 - Baselines and single-modality models\n\nQoL: 5-fold participant-grouped CV x 5 repeats, "
           "nested tuning. MMASH: leave-one-subject-out."),
        code(SETUP + "s = pd.read_csv(M/'qol_cv_summary.csv')\n"
             "cols = ['experiment','roc_auc_mean','roc_auc_sd','roc_auc_ci_low','roc_auc_ci_high',"
             "'balanced_accuracy_mean','macro_f1_mean','pairwise_accuracy_mean']\n"
             "s[s.strategy.isin(['single']) | s.experiment.str.startswith('Baseline')][cols].round(3)"),
        md("## MMASH sleep efficiency (LOSO)"),
        code("r = pd.read_csv(M/'mmash_loso_results.csv'); r[r.target=='sleep_efficiency'].sort_values('mae').round(3)"),
        code("display(Image(F/'mmash_loso_sleep_efficiency.png', width=650))"),
    ],
    "07_multimodal_fusion.ipynb": [
        md("# 07 - Multimodal fusion, comparisons, sensitivity"),
        code(SETUP + "s = pd.read_csv(M/'qol_cv_summary.csv')\n"
             "cols = ['experiment','roc_auc_mean','roc_auc_ci_low','roc_auc_ci_high','balanced_accuracy_mean',"
             "'pairwise_accuracy_mean']\ns[s.modalities.str.contains('\\+', regex=True)][cols].round(3)"),
        code("display(Image(F/'qol_modality_comparison.png', width=700)); "
             "display(Image(F/'qol_fusion_strategies.png', width=700))"),
        md("## Statistical comparisons (Nadeau-Bengio corrected resampled t-test on fold AUCs)"),
        code("pd.read_csv(M/'qol_model_comparisons.csv').round(4)"),
        md("## Learned stacking weights"),
        code("display(Image(F/'qol_stacking_weights.png', width=650))"),
        md("## Permutation tests and sensitivity analyses"),
        code("display(pd.read_csv(M/'qol_permutation_tests.csv').round(4)); pd.read_csv(M/'qol_sensitivity.csv').round(3)"),
        code("display(Image(F/'qol_roc_curves.png', width=420)); display(Image(F/'qol_confusion_matrix_best.png', width=330))"),
    ],
    "08_explainability.ipynb": [
        md("# 08 - Explainability and cross-dataset analysis\n\nSHAP values explain *model behaviour*; they are not "
           "causal effects."),
        code(SETUP + "sh = json.load(open(M/'shap_summary.json'))\njson.dumps(sh['qol_stress'], indent=1)[:3000]"),
        code("for p in sorted(F.glob('shap_bar_*.png')): display(Markdown(f'**{p.stem}**')); display(Image(p, width=560))"),
        code("for p in sorted(F.glob('shap_beeswarm_*.png'))[:2]: display(Image(p, width=620))"),
        code("display(Image(F/'shap_modality_share.png', width=520))"),
        code("for p in sorted(F.glob('shap_waterfall_*.png')): display(Image(p, width=520))"),
        md("## Cross-dataset: common indicators"),
        code("display(pd.read_csv(M/'cross_dataset_state_effects.csv').round(4))\n"
             "display(pd.read_csv(M/'cross_dataset_associations.csv').round(3))\n"
             "display(Image(F/'cross_dataset_indicators.png', width=750))"),
    ],
}


def main(execute: bool) -> None:
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        nbf.write(nb, path)
        if execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": str(NB)}})
            nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    main(ap.parse_args().execute)
