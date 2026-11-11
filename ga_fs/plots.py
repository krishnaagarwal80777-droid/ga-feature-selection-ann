import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ga_fs.ann import build_ann
from ga_fs.data import expand_group_mask, get_splits, load_dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(RES, "tables")
DATASETS = ("breast_cancer", "german_credit")
NAMES = {"breast_cancer": "Breast cancer", "german_credit": "German credit"}
CLASSES = {"breast_cancer": ["malignant", "benign"], "german_credit": ["bad", "good"]}
METHODS = ("all_features", "GA", "MI", "RFE", "PCA", "Random")
COMPARED = ("all_features", "MI", "RFE", "PCA", "Random")
COLORS = {
    "all_features": "#eb6834",
    "GA": "#2a78d6",
    "MI": "#1baf7a",
    "RFE": "#eda100",
    "PCA": "#e87ba4",
    "Random": "#008300",
}
LABELS = {
    "all_features": "All features",
    "GA": "GA",
    "MI": "MI",
    "RFE": "RFE",
    "PCA": "PCA (k components,\nall attributes)",
    "Random": "Random",
}
INK = "#333333"


def _style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e8e8e8",
        "axes.edgecolor": "#999999",
        "axes.titleweight": "bold",
    })


def _save(fig, name, made):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    made.append(name)


def _read(name):
    return pd.read_csv(os.path.join(RES, name), float_precision="round_trip")


def _band(ax, x, frame, style, label, alpha):
    m, s = frame.mean(axis=1), frame.std(axis=1)
    ax.plot(x, m, style, color=COLORS["GA"], lw=2, label=label)
    ax.fill_between(x, m - s, m + s, color=COLORS["GA"], alpha=alpha, lw=0)


def plot_convergence(dataset, made):
    frames = []
    for seed in range(10):
        h = _read(f"ga_history_{dataset}_seed{seed}.csv")
        frames.append(h.assign(seed=seed))
    h = pd.concat(frames)
    # best is final once a seed stops, mean only over seeds still running
    best = h.pivot(index="generation", columns="seed", values="best_fitness").ffill()
    mean = h.pivot(index="generation", columns="seed", values="mean_fitness")
    # drop the tail where too few seeds are left to average
    mean = mean[mean.notna().sum(axis=1) >= 3]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _band(ax, best.index, best, "-", "best fitness", 0.18)
    _band(ax, mean.index, mean, "--", "mean fitness (while 3+ seeds run)", 0.08)
    ax.set(xlabel="generation", ylabel="fitness", title=NAMES[dataset])
    ax.legend(frameon=False, loc="lower right")
    _save(fig, f"ga_convergence_{dataset}.png", made)


def _bar_label(m, n_attr, total):
    if m == "all_features":
        return f"{total} attr"
    if m == "PCA":
        return f"{total} attr\nk={n_attr:.1f}"
    return f"{n_attr:.1f} attr"


def plot_methods(runs, made):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(METHODS))
    for ax, dataset in zip(axes, DATASETS):
        g = runs[runs["dataset"] == dataset].groupby("method")
        mean = g["balanced_accuracy"].mean().reindex(METHODS)
        std = g["balanced_accuracy"].std().reindex(METHODS)
        n_attr = g["n_groups"].mean().reindex(METHODS)
        total = int(n_attr["all_features"])
        ax.bar(x, mean, yerr=std, width=0.7, color=[COLORS[m] for m in METHODS],
               capsize=4, error_kw={"lw": 1.2, "ecolor": INK})
        for i, m in enumerate(METHODS):
            ax.text(i, mean[m] + std[m] + 0.008, _bar_label(m, n_attr[m], total),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.set(title=NAMES[dataset], ylabel="balanced accuracy (axis starts at chance, 0.5)")
        ax.grid(axis="x", visible=False)
    _save(fig, "method_comparison.png", made)


def plot_reduction(runs, made):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(DATASETS))
    ga = runs[runs["method"] == "GA"].groupby("dataset")["n_groups"]
    total = runs[runs["method"] == "all_features"].groupby("dataset")["n_groups"].first()
    before = total.reindex(DATASETS).to_numpy()
    after, sd = ga.mean().reindex(DATASETS).to_numpy(), ga.std().reindex(DATASETS).to_numpy()
    ax.bar(x - 0.18, before, 0.34, color=COLORS["all_features"], label="before (all attributes)")
    ax.bar(x + 0.18, after, 0.34, yerr=sd, capsize=4, color=COLORS["GA"],
           error_kw={"ecolor": INK}, label="after GA (mean ± std)")
    for i in range(len(DATASETS)):
        ax.text(i - 0.18, before[i] + 0.4, f"{before[i]:.0f}", ha="center", va="bottom", color=INK)
        ax.text(i + 0.18, after[i] + sd[i] + 0.4,
                f"{after[i]:.1f}\n(-{100 * (1 - after[i] / before[i]):.0f}%)",
                ha="center", va="bottom", color=INK)
    ax.set_xticks(x, [NAMES[d] for d in DATASETS])
    ax.set(ylabel="attributes", ylim=(0, before.max() * 1.25))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "attribute_reduction.png", made)


def plot_frequency(runs, dataset, made):
    names = _read(f"feature_frequency_{dataset}.csv")["attribute"].tolist()
    sel = runs[(runs["dataset"] == dataset) & (runs["method"] == "GA")]["selected_attributes"]
    counts = pd.Series(0, index=names)
    for value in sel:
        counts[json.loads(value)] += 1
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(counts) + 1.2))
    ax.barh(counts.index, counts.values, color=COLORS["GA"], height=0.7)
    ax.set(xlabel=f"seeds selecting the attribute (of {len(sel)})", xlim=(0, len(sel)),
           title=NAMES[dataset])
    ax.set_xticks(range(len(sel) + 1))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    _save(fig, f"selection_frequency_{dataset}.png", made)


def plot_noise_rejection(noise_runs, made):
    methods = [m for m in METHODS if m in set(noise_runs["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(methods))
    for ax, dataset in zip(axes, DATASETS):
        g = noise_runs[noise_runs["dataset"] == dataset].groupby("method")
        real = g["real_groups_kept"].mean().reindex(methods)
        noise = g["noise_groups_kept"].mean().reindex(methods)
        colors = [COLORS[m] for m in methods]
        ax.bar(x, real, 0.7, color=colors, edgecolor="white", lw=1.5)
        ax.bar(x, noise, 0.7, bottom=real, color=colors, alpha=0.3,
               edgecolor=colors, hatch="///", lw=0.8)
        for i, m in enumerate(methods):
            ax.text(i, real[m] + noise[m] + 0.5, f"{noise[m]:.1f} noise",
                    ha="center", va="bottom", fontsize=9, color=INK)
        n_real = int(real["all_features"])
        ax.set_xticks(x, [LABELS[m] for m in methods], fontsize=9)
        ax.set(ylabel="attributes kept (mean of 5 seeds)",
               title=f"{NAMES[dataset]} ({n_real} real + {int(noise['all_features'])} noise)")
        ax.set_ylim(0, (real + noise).max() * 1.15)
        ax.grid(axis="x", visible=False)
    handles = [Patch(color="#888888", label="real attributes"),
               Patch(facecolor="#88888855", edgecolor="#888888", hatch="///",
                     label="noise attributes")]
    axes[1].legend(handles=handles, frameon=False, loc="upper right")
    _save(fig, "noise_rejection.png", made)


def _sweep(sens, dataset, noise, parameter):
    d = sens[(sens["dataset"] == dataset) & (sens["noise"] == noise)
             & (sens["parameter"] == parameter)]
    return d.groupby("value")


def _errline(ax, x, g, col, color=COLORS["GA"], label=None):
    ax.errorbar(x, g[col].mean(), yerr=g[col].std(), color=color, lw=2,
                marker="o", ms=7, capsize=4, label=label)


def plot_alpha(sens, made):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for j, dataset in enumerate(DATASETS):
        g = _sweep(sens, dataset, False, "alpha")
        alphas = np.array(sorted(g.groups))
        pos = np.arange(len(alphas))
        for i, (col, ylabel) in enumerate((("n_groups_selected", "attributes selected"),
                                           ("independent_train_cv", "independent train CV\n(balanced accuracy)"))):
            ax = axes[i, j]
            d = list(alphas).index(0.01)
            ax.axvspan(d - 0.3, d + 0.3, color="#eeeeee", lw=0)
            _errline(ax, pos, g, col)
            ax.set_xticks(pos, [f"{a:g}" for a in alphas])
            ax.set(ylabel=ylabel, xlabel="alpha (shaded = default 0.01)" if i else "")
            if i == 0:
                ax.set_title(NAMES[dataset])
    _save(fig, "alpha_tradeoff.png", made)


def plot_noise_alpha(sens, noise_runs, made):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dataset in zip(axes, DATASETS):
        g = _sweep(sens, dataset, True, "alpha")
        alphas = np.array(sorted(g.groups))
        pos = np.arange(len(alphas))
        _errline(ax, pos, g, "noise_groups_kept", label="GA")
        base = noise_runs[noise_runs["dataset"] == dataset].groupby("method")["noise_groups_kept"].mean()
        for m in ("MI", "RFE"):
            ax.axhline(base[m], color=COLORS[m], ls="--", lw=2,
                       label=f"{m} (same k as GA at alpha 0.01)")
        ax.set_xticks(pos, [f"{a:g}" for a in alphas])
        ax.set(xlabel="alpha", ylabel="noise attributes kept", title=NAMES[dataset])
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1],
                  frameon=False, fontsize=9)
    _save(fig, "noise_vs_alpha.png", made)


def plot_pop_size(sens, made):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for j, dataset in enumerate(DATASETS):
        g = _sweep(sens, dataset, False, "pop_size")
        sizes = sorted(g.groups)
        _errline(axes[0, j], sizes, g, "independent_train_cv")
        _errline(axes[1, j], sizes, g, "ga_runtime")
        axes[0, j].set(title=NAMES[dataset], ylabel="independent train CV\n(balanced accuracy)")
        axes[1, j].set(xlabel="population size", ylabel="GA runtime (s)")
        for ax in axes[:, j]:
            ax.set_xticks(sizes)
        axes[1, j].set_ylim(bottom=0)
    _save(fig, "pop_size.png", made)


def plot_mutation(sens, made):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dataset in zip(axes, DATASETS):
        g = _sweep(sens, dataset, False, "mutation_rate")
        rates = np.array(sorted(g.groups))
        pos = np.arange(len(rates))
        _errline(ax, pos, g, "independent_train_cv")
        n = round(rates[1] ** -1)
        ax.set_xticks(pos, [f"{r * n:g}/n\n({r:.3f})" for r in rates])
        ax.set(xlabel=f"per-bit mutation rate (n = {n} attributes)",
               ylabel="independent train CV\n(balanced accuracy)", title=NAMES[dataset])
    _save(fig, "mutation_rate.png", made)


def _seed0_preds(runs, dataset):
    # predictions were never saved, refit seed 0 and check against the csv
    X, y, _, groups, names = load_dataset(dataset)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, 0)
    out = {}
    for method in ("all_features", "GA"):
        row = runs[(runs["dataset"] == dataset) & (runs["seed"] == 0)
                   & (runs["method"] == method)].iloc[0]
        mask = expand_group_mask(np.isin(names, json.loads(row["selected_attributes"])), groups)
        model = build_ann(0).fit(X_tr[:, mask], y_tr)
        pred = model.predict(X_te[:, mask])
        prob = model.predict_proba(X_te[:, mask])[:, list(model.classes_).index(1)]
        if not (np.isclose(balanced_accuracy_score(y_te, pred), row["balanced_accuracy"])
                and np.isclose(roc_auc_score(y_te, prob), row["roc_auc"])):
            raise RuntimeError(f"seed 0 refit does not match csv: {dataset} {method}")
        out[method] = (y_te, pred, prob, int(row["n_groups"]))
    return out


def plot_confusion(preds, dataset, made):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    mats = {m: confusion_matrix(p[0], p[1]) for m, p in preds.items()}
    vmax = max(m.max() for m in mats.values())
    for ax, (method, mat) in zip(axes, mats.items()):
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax, cbar=False,
                    square=True, linewidths=2, linecolor="white", ax=ax,
                    xticklabels=CLASSES[dataset], yticklabels=CLASSES[dataset],
                    annot_kws={"fontsize": 13})
        ax.set(xlabel="predicted", ylabel="true",
               title=f"{LABELS[method]} ({preds[method][3]} attr)")
    fig.suptitle(f"{NAMES[dataset]}, seed 0 test set", fontweight="bold")
    _save(fig, f"confusion_{dataset}.png", made)


def plot_roc(preds, dataset, made):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#aaaaaa", lw=1.2, label="chance")
    for method, (y_te, _, prob, n) in preds.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, color=COLORS[method], lw=2,
                label=f"{LABELS[method]}, {n} attr (AUC {roc_auc_score(y_te, prob):.3f})")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title=f"{NAMES[dataset]}, seed 0 (positive = {CLASSES[dataset][1]})",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save(fig, f"roc_{dataset}.png", made)


def plot_effects(stats, made):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    y = np.arange(len(COMPARED))[::-1]
    for ax, dataset in zip(axes, DATASETS):
        d = stats[stats["dataset"] == dataset].set_index("comparison")
        for yi, m in zip(y, COMPARED):
            r = d.loc[f"GA_vs_{m}"]
            diff = r["mean_balanced_accuracy_difference"]
            err = [[diff - r["nb_ci95_lower"]], [r["nb_ci95_upper"] - diff]]
            ax.errorbar(diff, yi, xerr=err, fmt="o", ms=8, color=COLORS[m],
                        capsize=4, lw=2)
            ax.annotate(f"p={r['nb_pvalue_holm']:.2f}", (1.01, yi),
                        xycoords=("axes fraction", "data"), va="center",
                        fontsize=9, color=INK)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, [LABELS[m].replace("\n", " ") for m in COMPARED])
        ax.set(title=NAMES[dataset],
               xlabel="GA minus method, balanced accuracy\n(Nadeau-Bengio 95% CI, Holm p)")
        ax.grid(axis="y", visible=False)
    _save(fig, "effect_sizes.png", made)


def _pm(g, col, digits=4):
    m, s = g[col].mean(), g[col].std()
    return [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(m, s)]


def _write(frame, name, made):
    frame.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    made.append(f"tables/{name}")


def table_main(runs, made):
    g = runs.groupby(["dataset", "method"], sort=False)
    out = g.size().reset_index()[["dataset", "method"]]
    out["attributes"] = _pm(g, "n_groups", 1)
    for col in ("balanced_accuracy", "macro_f1", "minority_recall", "roc_auc", "accuracy"):
        out[col] = _pm(g, col)
    out["method"] = out["method"].map(lambda m: LABELS[m].replace("\n", " "))
    out.loc[out["method"].str.startswith("PCA"), "attributes"] += " components"
    _write(out, "main_comparison.csv", made)


def table_stats(stats, name, made):
    out = pd.DataFrame({
        "dataset": stats["dataset"],
        "comparison": stats["comparison"].str.replace("_vs_", " vs ").str.replace("all_features", "all features"),
        "n_pairs": stats["n_pairs"],
        "mean_diff": stats["mean_balanced_accuracy_difference"].map("{:+.4f}".format),
        "nb_ci95": [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(stats["nb_ci95_lower"], stats["nb_ci95_upper"])],
        "nb_p_holm": stats["nb_pvalue_holm"].map("{:.4f}".format),
        "t_ci95": [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(stats["ci95_lower"], stats["ci95_upper"])],
        "t_p_holm": stats["paired_t_pvalue_holm"].map("{:.4f}".format),
        "wilcoxon_p_holm": stats["wilcoxon_pvalue_holm"].map("{:.4f}".format),
        "wilcoxon_nonzero": stats["wilcoxon_n_nonzero_differences"],
        "min_possible_wilcoxon_p": stats["min_possible_wilcoxon_p"].map("{:.4f}".format),
    })
    _write(out, name, made)


def table_noise(noise_runs, made):
    g = noise_runs.groupby(["dataset", "method"], sort=False)
    out = g.size().reset_index()[["dataset", "method"]]
    out["attributes_kept"] = _pm(g, "n_groups", 1)
    out["real_kept"] = _pm(g, "real_groups_kept", 1)
    out["noise_kept"] = _pm(g, "noise_groups_kept", 1)
    out["balanced_accuracy"] = _pm(g, "balanced_accuracy")
    out["method"] = out["method"].map(lambda m: LABELS[m].replace("\n", " "))
    _write(out, "noise_rejection.csv", made)


def table_alpha(sens, made):
    d = sens[sens["parameter"] == "alpha"]
    g = d.groupby(["dataset", "noise", "value"])
    out = g.size().reset_index()[["dataset", "noise", "value"]].rename(columns={"value": "alpha"})
    out["attributes"] = _pm(g, "n_groups_selected", 1)
    out["noise_kept"] = _pm(g, "noise_groups_kept", 1)
    for col in ("ga_fitness_cv", "independent_train_cv", "test_balanced_accuracy"):
        out[col] = _pm(g, col)
    out.loc[~out["noise"], "noise_kept"] = "-"
    _write(out, "alpha_sweep.csv", made)


def run_plots():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    _style()
    runs, noise_runs = _read("all_runs.csv"), _read("noise_all_runs.csv")
    sens, stats = _read("sensitivity_runs.csv"), _read("stats.csv")
    made = []
    for dataset in DATASETS:
        plot_convergence(dataset, made)
    plot_methods(runs, made)
    plot_reduction(runs, made)
    for dataset in DATASETS:
        plot_frequency(runs, dataset, made)
    plot_noise_rejection(noise_runs, made)
    plot_alpha(sens, made)
    plot_noise_alpha(sens, noise_runs, made)
    plot_pop_size(sens, made)
    plot_mutation(sens, made)
    for dataset in DATASETS:
        preds = _seed0_preds(runs, dataset)
        plot_confusion(preds, dataset, made)
        plot_roc(preds, dataset, made)
    plot_effects(stats, made)
    table_main(runs, made)
    table_stats(stats, "stats.csv", made)
    table_stats(_read("noise_stats.csv"), "noise_stats.csv", made)
    table_noise(noise_runs, made)
    table_alpha(sens, made)
    for name in made:
        print(f"saved {name}")
