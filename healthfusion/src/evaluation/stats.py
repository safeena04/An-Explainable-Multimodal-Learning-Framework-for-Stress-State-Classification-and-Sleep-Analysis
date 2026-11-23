"""Statistical tools: paired tests, multiple-comparison control, bootstrap CIs,
corrected resampled t-test and group-preserving permutation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted p-values (Benjamini & Hochberg, 1995)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni family-wise-error adjusted p-values (NaN-aware)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    adj = np.maximum.accumulate([(m - k) * p[i] for k, i in enumerate(order)])
    out[order] = np.minimum(adj, 1.0)
    return out


def paired_wilcoxon(df: pd.DataFrame, features: list[str], id_col="pid", cond_col="phase",
                    a="relax", b="stress") -> pd.DataFrame:
    """Wilcoxon signed-rank test per feature on complete pairs.

    Effect size = matched-pairs rank-biserial correlation r_rb in [-1, 1];
    positive means higher values under condition b (stress).
    """
    rows = []
    for f in features:
        w = df.pivot(index=id_col, columns=cond_col, values=f).dropna()
        d = (w[b] - w[a]).to_numpy()
        n = int(len(d))
        if n < 6 or np.allclose(d, 0):
            rows.append({"feature": f, "n_pairs": n})
            continue
        res = stats.wilcoxon(w[b], w[a], zero_method="wilcox")
        nz = d[d != 0]
        ranks = stats.rankdata(np.abs(nz))
        r_rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
        rows.append({"feature": f, "n_pairs": n, f"median_{a}": float(w[a].median()),
                     f"median_{b}": float(w[b].median()), "median_diff": float(np.median(d)),
                     "wilcoxon_W": float(res.statistic), "p_value": float(res.pvalue),
                     "rank_biserial_r": float(r_rb), "frac_increase": float(np.mean(d > 0))})
    out = pd.DataFrame(rows)
    if "p_value" in out:
        out["p_fdr"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out


def bootstrap_ci(x: np.ndarray, func=np.mean, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([func(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def spearman_with_ci(x, y, n_boot: int = 2000, seed: int = 0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return {"n": int(len(x)), "rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            boots.append(stats.spearmanr(x[i], y[i])[0])
    return {"n": int(len(x)), "rho": float(rho), "p_value": float(p),
            "ci_low": float(np.nanpercentile(boots, 2.5)), "ci_high": float(np.nanpercentile(boots, 97.5))}


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.

    The naive t-test on fold scores is anti-conservative because training sets
    overlap; the variance is inflated by (1/k_total + n_test/n_train).
    """
    d = np.asarray(diffs, float)
    k = d.size
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": np.nan, "p_value": np.nan}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p)}


def permute_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels inside each participant (keeps one relax + one stress per
    person, so class balance and the grouped structure are preserved)."""
    y = np.asarray(y).copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        y[idx] = y[rng.permutation(idx)]
    return y
