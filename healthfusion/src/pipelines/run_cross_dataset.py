"""Phase 9: cross-dataset investigation of common physiological indicators.

Only indicators that can be computed *the same way* in both datasets are used:
heart rate, RMSSD and SDNN from 30-s inter-beat-interval windows (QoL: its
single 30-s ECG strip per phase; MMASH: 30-s segments of the Polar H7 RR stream,
summarised per participant by the median, plus one randomly drawn 30-s window to
match QoL's single-window sampling noise).

What is compared:
  1. distributions (Mann-Whitney U, Cliff's delta) - different people, devices
     and settings, so differences are descriptive, not attributable to one cause;
  2. the direction of within-person state effects (MMASH sleep vs wake;
     QoL stress vs relax), each tested within its own dataset;
  3. the sign of associations with self-reported anxiety/stress.
No model is transferred between datasets and no generalisation claim is made.

Run:  python -m src.pipelines.run_cross_dataset
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import GLOBAL_SEED, METRICS_DIR, PROCESSED_DIR
from src.evaluation.stats import benjamini_hochberg, spearman_with_ci
from src.visualization import plots


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def build_table() -> pd.DataFrame:
    seg = pd.read_csv(PROCESSED_DIR / "mmash_hrv_segments.csv")
    s30 = seg[seg.segment == "30s"].dropna(subset=["rmssd"])
    rng = np.random.default_rng(GLOBAL_SEED)
    rows = []
    for (u, period), g in s30.groupby(["user", "period"]):
        ctx = "MMASH awake (pre-sleep)" if period == "day_pre_sleep" else "MMASH asleep"
        one = g.iloc[rng.integers(len(g))]
        rows.append({"dataset": "MMASH", "context": ctx, "id": f"M{u:02d}", "hr": g["hr"].median(),
                     "rmssd": g["rmssd"].median(), "sdnn": g["sdnn"].median(), "n_windows": len(g),
                     "hr_single": one["hr"], "rmssd_single": one["rmssd"], "sdnn_single": one["sdnn"]})
    q = pd.read_csv(PROCESSED_DIR / "qol_phase_features.csv")
    for _, r in q.dropna(subset=["ecg_rmssd"]).iterrows():
        rows.append({"dataset": "QoL_Stress", "context": f"QoL {r.phase}", "id": r.pid, "hr": r.ecg_hr,
                     "rmssd": r.ecg_rmssd, "sdnn": r.ecg_sdnn, "n_windows": 1,
                     "hr_single": r.ecg_hr, "rmssd_single": r.ecg_rmssd, "sdnn_single": r.ecg_sdnn})
    t = pd.DataFrame(rows)
    for c in ("rmssd", "rmssd_single"):
        t[f"ln_{c}"] = np.log(t[c])
    order = ["MMASH awake (pre-sleep)", "MMASH asleep", "QoL relax", "QoL stress"]
    t["context"] = pd.Categorical(t["context"], order, ordered=True)
    return t.sort_values(["context", "id"]).reset_index(drop=True)


def main() -> None:
    t = build_table()
    t.to_csv(METRICS_DIR / "cross_dataset_indicator_table.csv", index=False)
    qn = pd.read_csv(PROCESSED_DIR / "qol_questionnaires.csv")
    mm = pd.read_csv(PROCESSED_DIR / "mmash_participant_features.csv")

    # 1) distributions: MMASH awake vs QoL relax (both resting-ish wakefulness) --------
    dist = []
    pairs = [("MMASH awake (pre-sleep)", "QoL relax"), ("MMASH awake (pre-sleep)", "QoL stress"),
             ("MMASH asleep", "QoL relax")]
    male_q = set(qn.loc[qn.sex == "Male", "pid"])
    for a, b in pairs:
        for metric in ("hr", "ln_rmssd", "sdnn", "hr_single", "ln_rmssd_single"):
            for subset in ("all", "QoL males only"):
                xa = t.loc[t.context == a, metric].dropna()
                tb = t[t.context == b]
                if subset != "all":
                    tb = tb[tb.id.isin(male_q)]
                xb = tb[metric].dropna()
                u = stats.mannwhitneyu(xa, xb)
                dist.append({"context_a": a, "context_b": b, "metric": metric, "qol_subset": subset,
                             "n_a": len(xa), "n_b": len(xb), "median_a": xa.median(), "median_b": xb.median(),
                             "mannwhitney_p": u.pvalue, "cliffs_delta": cliffs_delta(xa, xb)})
    dist = pd.DataFrame(dist)
    dist["p_fdr"] = benjamini_hochberg(dist["mannwhitney_p"].to_numpy())
    dist.to_csv(METRICS_DIR / "cross_dataset_distributions.csv", index=False)

    # 2) within-person state effects, each dataset on its own -----------------------------
    eff = []
    for ds, a, b, label in (("MMASH", "MMASH awake (pre-sleep)", "MMASH asleep", "asleep - awake"),
                            ("QoL_Stress", "QoL relax", "QoL stress", "stress - relax")):
        for metric in ("hr", "ln_rmssd", "sdnn"):
            w = t[t.context.isin([a, b])].pivot(index="id", columns="context", values=metric)[[a, b]].dropna()
            d = w[b] - w[a]
            p = stats.wilcoxon(w[b], w[a]).pvalue
            eff.append({"dataset": ds, "contrast": label, "metric": metric, "n_pairs": len(d),
                        "median_change": float(d.median()), "frac_increase": float((d > 0).mean()),
                        "wilcoxon_p": float(p), "direction": "increase" if d.median() > 0 else "decrease"})
    eff = pd.DataFrame(eff)
    eff.to_csv(METRICS_DIR / "cross_dataset_state_effects.csv", index=False)

    # 3) associations with self-report --------------------------------------------------
    assoc = []
    awake = t[t.context == "MMASH awake (pre-sleep)"].assign(user=lambda d: d.id.str[1:].astype(int))
    m = awake.merge(mm, on="user")
    for f, lab in (("psych_stai_state", "STAI state"), ("psych_stai_trait", "STAI trait"),
                   ("psych_daily_stress", "Daily Stress Inventory"), ("sleep_efficiency", "sleep efficiency")):
        for metric in ("hr", "ln_rmssd"):
            assoc.append({"dataset": "MMASH", "indicator": f"{metric} (awake)", "measure": lab,
                          **spearman_with_ci(m[metric], m[f], seed=GLOBAL_SEED)})
    qw = t[t.dataset == "QoL_Stress"].pivot(index="id", columns="context", values=["hr", "ln_rmssd"])
    qw.columns = [f"{a}_{b.split()[-1]}" for a, b in qw.columns]
    qw = qw.join(qn.set_index("pid")[["pss14", "stai_relax", "stai_stress", "stai_delta"]])
    for metric in ("hr", "ln_rmssd"):
        assoc.append({"dataset": "QoL_Stress", "indicator": f"{metric} (relax)", "measure": "STAI (relax)",
                      **spearman_with_ci(qw[f"{metric}_relax"], qw["stai_relax"], seed=GLOBAL_SEED)})
        assoc.append({"dataset": "QoL_Stress", "indicator": f"{metric} (relax)", "measure": "PSS-14",
                      **spearman_with_ci(qw[f"{metric}_relax"], qw["pss14"], seed=GLOBAL_SEED)})
        assoc.append({"dataset": "QoL_Stress", "indicator": f"{metric} change (stress - relax)",
                      "measure": "STAI change", **spearman_with_ci(qw[f"{metric}_stress"] - qw[f"{metric}_relax"],
                                                                   qw["stai_delta"], seed=GLOBAL_SEED)})
    assoc = pd.DataFrame(assoc)
    assoc["p_fdr"] = benjamini_hochberg(assoc["p_value"].to_numpy())
    assoc.to_csv(METRICS_DIR / "cross_dataset_associations.csv", index=False)

    plots.cross_dataset_distributions(t, [("hr", "heart rate (bpm)"), ("ln_rmssd", "ln RMSSD (ln ms)"),
                                          ("sdnn", "SDNN (ms)")], "cross_dataset_indicators")
    summary = {
        "contexts_n": t.groupby("context", observed=True).size().to_dict(),
        "state_effects": eff.to_dict("records"),
        "n_significant_associations_fdr": int((assoc.p_fdr < 0.05).sum()),
        "transfer_experiment": "not performed; no cross-dataset generalisation is claimed",
        "comparability_notes": [
            "Different devices: Polar H7 chest strap (MMASH) vs Fitbit Charge 6 finger ECG (QoL).",
            "Different populations: 22 young men (MMASH) vs 66 adults, 27 women (QoL).",
            "MMASH participant values are medians of hundreds of 30-s windows; QoL values are single 30-s windows. "
            "The *_single columns use one random MMASH window to match sampling noise.",
        ],
    }
    json.dump(summary, open(METRICS_DIR / "cross_dataset_summary.json", "w"), indent=2, default=str)
    print(eff.to_string(index=False))
    print(dist[dist.qol_subset == "all"][["context_a", "context_b", "metric", "median_a", "median_b",
                                          "cliffs_delta", "p_fdr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
