"""Paper-ready static figures (matplotlib, PNG 300 dpi + PDF).

Colour roles follow one fixed categorical order so a modality keeps the same
colour in every figure (validated palette, first three slots are safe for
colour-vision deficiency in all pairings):
    ECG = blue, EDA = orange, HR = aqua, fusion / multimodal = neutral ink.
MMASH modalities reuse the same three slots in the same order.
Text always uses ink colours, never series colours.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8d8c86"
GRID = "#e4e3df"
SURFACE = "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODALITY_COLORS = {"ECG": SERIES[0], "EDA": SERIES[1], "HR": SERIES[2], "Fusion": "#3d3c39",
                   "Physiological": SERIES[0], "Behavioural": SERIES[1], "Psychological": SERIES[2]}
RELAX, STRESS = SERIES[0], SERIES[1]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "text.color": INK,
    })


def save(fig, name: str) -> str:
    fig.tight_layout()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _modality_color(mods: str) -> str:
    parts = mods.split("+")
    return MODALITY_COLORS.get(parts[0], INK_2) if len(parts) == 1 else MODALITY_COLORS["Fusion"]


def dot_ci(ax, labels, mean, lo, hi, colors, ref=None, ref_label=None, xlabel=""):
    y = np.arange(len(labels))[::-1]
    for yi, m, l, h, c in zip(y, mean, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.scatter([m], [yi], s=46, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(h + 0.008, yi, f"{m:.3f}", va="center", fontsize=8, color=INK_2)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--")
        if ref_label:
            ax.text(ref + 0.004, -0.75, ref_label, ha="left", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_ylim(-1.0, len(labels) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- QoL_Stress figures
def qol_modality_comparison(best: pd.DataFrame, baselines: pd.DataFrame, n: int | None = None) -> str:
    apply_style()
    df = pd.concat([best, baselines]).sort_values("roc_auc_mean", ascending=False)
    labels = [f"{r.modalities.replace('+', ' + ')}  ({r.strategy}, {r.model})"
              if not r.experiment.startswith("Baseline") else r.experiment.replace("Baseline | ", "Baseline: ")
              for r in df.itertuples()]
    colors = [MUTED if e.startswith("Baseline") else _modality_color(m) for e, m in zip(df.experiment, df.modalities)]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(df) + 1.2))
    dot_ci(ax, labels, df.roc_auc_mean, df.roc_auc_ci_low, df.roc_auc_ci_high, colors, ref=0.5,
           ref_label="chance", xlabel="ROC-AUC (mean and 95 % CI over 25 participant-grouped folds)")
    ax.set_title(f"Stress vs relaxation: best model per modality set (complete-case, n = {n})", loc="left")
    return save(fig, "qol_modality_comparison")


def qol_fusion_strategies(summary: pd.DataFrame) -> str:
    apply_style()
    combos = [c for c in ["ECG+EDA", "ECG+HR", "EDA+HR", "ECG+EDA+HR"] if c in set(summary.modalities)]
    strategies = [("early", SERIES[0]), ("late", SERIES[1]), ("stacked", SERIES[2])]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(combos))
    for i, (s, c) in enumerate(strategies):
        vals = []
        for combo in combos:
            g = summary[(summary.modalities == combo) & (summary.strategy == s)]
            vals.append(g.sort_values("roc_auc_mean").iloc[-1] if len(g) else None)
        xs = x + (i - 1) * 0.22
        for xi, v in zip(xs, vals):
            if v is None:
                continue
            ax.plot([xi, xi], [v.roc_auc_ci_low, v.roc_auc_ci_high], color=c, lw=2)
            ax.scatter([xi], [v.roc_auc_mean], color=c, s=46, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=f"{s} fusion" if xi == xs[0] else None)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("+", " + ") for c in combos])
    ax.set_ylabel("ROC-AUC (best base model per strategy)")
    ax.set_title("Fusion strategy comparison (mean and 95 % CI)", loc="left")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    return save(fig, "qol_fusion_strategies")


def qol_roc_curves(oof: pd.DataFrame, experiments: dict[str, str]) -> str:
    from sklearn.metrics import roc_auc_score, roc_curve
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    for label, exp in experiments.items():
        g = oof[oof.experiment == exp]
        if g.empty:
            continue
        p = g.groupby(["pid", "y"])["prob"].mean().reset_index()
        fpr, tpr, _ = roc_curve(p["y"], p["prob"])
        auc = roc_auc_score(p["y"], p["prob"])
        key = label.split(" ")[0]
        ax.plot(fpr, tpr, color=MODALITY_COLORS.get(key, MODALITY_COLORS["Fusion"]),
                label=f"{label}: pooled AUC {auc:.2f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, out-of-fold predictions\n(probabilities averaged over 5 CV repeats)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    ax.set_aspect("equal")
    return save(fig, "qol_roc_curves")


def confusion_matrix_plot(cm: np.ndarray, title: str, name: str, labels=("relax", "stress")) -> str:
    apply_style()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap=cmap, vmin=0)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center",
                    color=SURFACE if cm[i, j] > cm.max() * 0.55 else INK, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    return save(fig, name)


def qol_paired_effects(tests: pd.DataFrame) -> str:
    apply_style()
    df = tests.dropna(subset=["rank_biserial_r"]).sort_values("rank_biserial_r")
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(df) + 1.2))
    y = np.arange(len(df))
    colors = [MODALITY_COLORS[m] for m in df.modality]
    ax.hlines(y, 0, df.rank_biserial_r, color=colors, lw=2)
    ax.scatter(df.rank_biserial_r, y, color=colors, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for yi, r in zip(y, df.itertuples()):
        star = "  FDR<0.05" if r.p_fdr < 0.05 else ""
        ax.text(r.rank_biserial_r + (0.02 if r.rank_biserial_r >= 0 else -0.02), yi,
                f"{r.rank_biserial_r:+.2f}{star}", va="center", ha="left" if r.rank_biserial_r >= 0 else "right",
                fontsize=7.5, color=INK_2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df.feature)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Matched-pairs rank-biserial r  (positive = higher under stress)")
    ax.set_title("Within-participant change, relax -> stress (Wilcoxon signed-rank)", loc="left")
    handles = [matplotlib.lines.Line2D([], [], color=MODALITY_COLORS[m], marker="o", lw=0, label=m)
               for m in ("ECG", "EDA", "HR")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="y", visible=False)
    return save(fig, "qol_paired_effects")


def qol_stai_manipulation(q: pd.DataFrame) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    for _, r in q.iterrows():
        ax.plot([0, 1], [r.stai_relax, r.stai_stress], color=GRID if r.stai_stress <= r.stai_relax else "#b7d3f6",
                lw=1, zorder=1)
    for x, col, c in ((0, "stai_relax", RELAX), (1, "stai_stress", STRESS)):
        ax.scatter(np.full(len(q), x) + np.random.default_rng(0).uniform(-0.04, 0.04, len(q)), q[col],
                   s=12, color=c, alpha=0.8, zorder=2)
        ax.scatter([x], [q[col].mean()], s=70, color=INK, marker="D", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["relax", "stress"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("STAI state score")
    ax.set_title("Manipulation check (STAI)", loc="left")
    ax.grid(axis="x", visible=False)
    return save(fig, "qol_stai_manipulation")


def qol_stacking_weights(w: pd.DataFrame) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    exps = list(w.index)
    mods = ["ECG", "EDA", "HR"]
    x = np.arange(len(exps))
    for i, m in enumerate(mods):
        if (f"w_{m}", "mean") not in w.columns:
            continue
        vals = w[(f"w_{m}", "mean")].to_numpy()
        sds = w[(f"w_{m}", "std")].to_numpy()
        ax.bar(x + (i - 1) * 0.25, np.nan_to_num(vals), 0.23, yerr=np.nan_to_num(sds), color=MODALITY_COLORS[m],
               label=m, capsize=2, error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" | stacked | ", "\n") for e in exps], fontsize=7.5)
    ax.set_ylabel("learned weight (mean ± SD over folds)")
    ax.set_title("Modality weights learned by stacking (inner CV, training folds only)", loc="left")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    return save(fig, "qol_stacking_weights")


# ---------------------------------------------------------------- explainability
def shap_importance_bar(imp: pd.DataFrame, title: str, name: str) -> str:
    apply_style()
    df = imp.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(5.8, 0.28 * len(df) + 1.1))
    y = np.arange(len(df))
    ax.barh(y, df.mean_abs_shap, xerr=df.get("fold_sd"), color=[MODALITY_COLORS.get(m, INK_2) for m in df.modality],
            height=0.65, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels(df.feature)
    ax.set_xlabel("mean |SHAP| (log-odds), out-of-fold")
    ax.set_title(title, loc="left")
    mods = [m for m in dict.fromkeys(df.modality)]
    ax.legend(handles=[matplotlib.patches.Patch(color=MODALITY_COLORS.get(m, INK_2), label=m) for m in mods],
              loc="lower right")
    ax.grid(axis="y", visible=False)
    return save(fig, name)


def modality_share_bar(share: pd.DataFrame, title: str, name: str) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    rows = list(share.index)
    y = np.arange(len(rows))[::-1]
    for yi, model in zip(y, rows):
        left = 0.0
        for m in share.columns:
            v = share.loc[model, m]
            ax.barh(yi, v, left=left, color=MODALITY_COLORS.get(m, INK_2), height=0.55,
                    edgecolor=SURFACE, linewidth=2, label=m if yi == y[0] else None)
            if v > 0.17:
                ax.text(left + v / 2, yi, f"{m} {v:.0%}", ha="center", va="center", fontsize=7.5, color=SURFACE)
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of total mean |SHAP|")
    ax.set_title(title, loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    ax.grid(False)
    return save(fig, name)


# ---------------------------------------------------------------- MMASH figures
def mmash_loso(res: pd.DataFrame, target: str) -> str:
    apply_style()
    df = res[(res.target == target) & (res.model == "ridge")].copy()
    base = res[(res.target == target) & (res.strategy == "baseline")]["mae"].iloc[0]
    df["label"] = df.modalities.str.replace("+", " + ") + "  (" + df.strategy + ")"
    df = df.sort_values("mae")
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(df) + 1.5))
    y = np.arange(len(df))[::-1]
    lo, hi = min(base, df.mae.min()), df.mae.max()
    span = hi - lo
    ax.set_xlim(lo - 0.15 * span, hi + 1.1 * span)
    for yi, r in zip(y, df.itertuples()):
        c = _modality_color({"Phys": "Physiological", "Behav": "Behavioural", "Psych": "Psychological"}
                            .get(r.modalities, r.modalities))
        ax.scatter([r.mae], [yi], color=c, s=46, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        p = "" if np.isnan(r.perm_p_value) else f"   R² {r.r2:+.2f}, perm p = {r.perm_p_value:.2f}"
        ax.text(r.mae + 0.03 * span, yi, f"{r.mae:.2f}{p}", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(base, color=MUTED, ls="--", lw=1)
    ax.text(base + 0.02 * span, -0.85, f"predict-the-mean baseline (MAE {base:.2f})", ha="left", fontsize=7.5,
            color=MUTED)
    ax.set_ylim(-1.1, len(df) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df.label)
    ax.set_xlabel(f"LOSO mean absolute error, {target.replace('_', ' ')} (lower is better)")
    ax.set_title(f"MMASH (n = {int(df.n.iloc[0])}): Ridge regression by modality set", loc="left")
    ax.grid(axis="y", visible=False)
    return save(fig, f"mmash_loso_{target}")


def mmash_correlation_heatmap(corr: pd.DataFrame) -> str:
    apply_style()
    piv = corr.pivot(index="feature", columns="target", values="rho")
    pf = corr.pivot(index="feature", columns="target", values="p_fdr")
    praw = corr.pivot(index="feature", columns="target", values="p_value")
    order = corr.drop_duplicates("feature")["feature"].tolist()
    piv, pf, praw = piv.loc[order], pf.loc[order], praw.loc[order]
    fig, ax = plt.subplots(figsize=(5.6, 0.27 * len(piv) + 1.4))
    im = ax.imshow(piv.to_numpy(), cmap=DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            mark = "**" if pf.iat[i, j] < 0.05 else ("*" if praw.iat[i, j] < 0.05 else "")
            ax.text(j, i, f"{v:.2f}{mark}", ha="center", va="center", fontsize=6.5,
                    color=SURFACE if abs(v) > 0.55 else INK)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace("_", "\n") for c in piv.columns], fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Spearman rho")
    ax.set_title("MMASH: feature-outcome correlations (n = 21)\n* p < 0.05 uncorrected, ** FDR < 0.05", loc="left")
    return save(fig, "mmash_correlation_heatmap")


# ---------------------------------------------------------------- cross-dataset
def cross_dataset_distributions(df: pd.DataFrame, metrics: list[tuple[str, str]], name: str) -> str:
    apply_style()
    contexts = list(dict.fromkeys(df["context"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6))
    for ax, (col, lab) in zip(np.atleast_1d(axes), metrics):
        data = [df.loc[df.context == c, col].dropna().to_numpy() for c in contexts]
        colors = [SERIES[0] if c.startswith("MMASH") else SERIES[1] for c in contexts]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        rng = np.random.default_rng(0)
        for i, (d, c) in enumerate(zip(data, colors)):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=9, color=c, alpha=0.8, zorder=3)
        ax.set_xticks(range(1, len(contexts) + 1))
        ax.set_xticklabels([c.replace(" ", "\n", 1) for c in contexts], fontsize=7.5)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Common indicators across contexts (participant-level medians, 30-s windows)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    return save(fig, name)
