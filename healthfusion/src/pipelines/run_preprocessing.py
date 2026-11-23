"""Phases 3-4: preprocessing + feature engineering for both datasets.

Run:  python -m src.pipelines.run_preprocessing
"""
from __future__ import annotations

from src.features.registry import write_feature_dictionary
from src.preprocessing.mmash_preprocess import build_mmash
from src.preprocessing.qol_preprocess import build_qol


def main() -> None:
    q = build_qol()
    f = q["features"]
    print(f"QoL_Stress: {len(f)} participant-phase rows; complete-case participants: "
          f"{f.loc[f.complete_case, 'pid'].nunique()}")
    print(q["qc"].query("included == False")[["pid", "phase", "modality", "reason"]].to_string(index=False))
    m = build_mmash()
    print(f"MMASH: {len(m['features'])} participants with a sleep record "
          f"({(~m['qc']['included']).sum()} excluded, reason logged)")
    write_feature_dictionary()
    print("Feature dictionary written to docs/FEATURE_DICTIONARY.md")


if __name__ == "__main__":
    main()
