import json
import os
import time
import warnings

os.environ["PYTHONWARNINGS"] = (
    "ignore:Stochastic Optimizer:Warning:"
    "sklearn.neural_network._multilayer_perceptron"
)

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.exceptions import ConvergenceWarning

from ga_fs.ann import evaluate
from ga_fs.baselines import (
    mutual_info_mask,
    pca_transform,
    random_mask,
    rfe_mask,
)
from ga_fs.data import expand_group_mask, get_splits, load_dataset
from ga_fs.ga import run_ga


DATASETS = ("breast_cancer", "german_credit")
SEEDS = range(10)
METHODS = ("all_features", "GA", "Random", "MI", "RFE", "PCA")
NOISE_SEEDS = range(5)
NOISE_COUNTS = {"breast_cancer": 30, "german_credit": 20}
NOISE_SEED = 2026
NOISE_METHODS = ("all_features", "GA", "Random", "MI", "RFE")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "roc_auc",
)
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
RUNS_PATH = os.path.join(RESULT_DIR, "all_runs.csv")
NOISE_RUNS_PATH = os.path.join(RESULT_DIR, "noise_all_runs.csv")


def _save_csv(frame, path):
    temp_path = f"{path}.tmp"
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _load_runs(path=RUNS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _complete_runs(frame, methods=METHODS):
    complete = set()
    if frame.empty:
        return complete

    for (dataset, seed), rows in frame.groupby(["dataset", "seed"]):
        if set(rows["method"]) >= set(methods):
            complete.add((dataset, int(seed)))
    return complete


def _method_row(dataset, seed, method, group_mask, groups, metrics, ga_runtime,
                ga_generations, selected_names, n_columns=None):
    if n_columns is None:
        n_columns = int(expand_group_mask(group_mask, groups).sum())

    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "n_groups": int(group_mask.sum()),
        "n_columns": int(n_columns),
        **{name: float(metrics[name]) for name in METRICS},
        "ga_runtime": float(ga_runtime),
        "ga_generations": int(ga_generations),
        "selected_attributes": json.dumps(selected_names),
    }


def _run_seed(dataset, seed):
    X, y, _, groups, group_names = load_dataset(dataset)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)

    start = time.perf_counter()
    ga_mask, history = run_ga(
        X_tr,
        y_tr,
        groups,
        seed=seed,
        pop_size=30,
        n_gen=40,
        alpha=0.01,
        metric="balanced_accuracy",
    )
    ga_runtime = time.perf_counter() - start
    ga_generations = len(history)
    k = int(ga_mask.sum())
    masks = {
        "all_features": np.ones(len(group_names), dtype=bool),
        "GA": ga_mask,
        "MI": mutual_info_mask(X_tr, y_tr, groups, k, seed=seed),
        "RFE": rfe_mask(X_tr, y_tr, groups, k, seed=seed),
    }

    rows = []
    for method, mask in masks.items():
        column_mask = expand_group_mask(mask, groups)
        metrics = evaluate(X_tr, y_tr, X_te, y_te, column_mask, seed)
        rows.append(
            _method_row(
                dataset, seed, method, mask, groups, metrics, ga_runtime,
                ga_generations, group_names[mask].tolist(),
            )
        )

    random_scores = []
    random_column_counts = []
    for draw in range(10):
        mask = random_mask(len(group_names), k, seed=seed * 10 + draw)
        column_mask = expand_group_mask(mask, groups)
        random_scores.append(evaluate(X_tr, y_tr, X_te, y_te, column_mask, seed))
        random_column_counts.append(int(column_mask.sum()))
    random_metrics = {
        metric: float(np.mean([score[metric] for score in random_scores]))
        for metric in METRICS
    }
    random_mask_for_row = np.zeros(len(group_names), dtype=bool)
    random_mask_for_row[:k] = True
    rows.append(
        _method_row(
            dataset, seed, "Random", random_mask_for_row, groups, random_metrics,
            ga_runtime, ga_generations, [],
            n_columns=float(np.mean(random_column_counts)),
        )
    )

    X_tr_pca, X_te_pca = pca_transform(X_tr, X_te, k, seed=seed)
    pca_metrics = evaluate(
        X_tr_pca, y_tr, X_te_pca, y_te, np.ones(k, dtype=bool), seed
    )
    rows.append(
        _method_row(
            dataset, seed, "PCA", np.ones(k, dtype=bool),
            np.arange(k), pca_metrics, ga_runtime, ga_generations, [],
            n_columns=k,
        )
    )

    history_frame = pd.DataFrame(history)
    history_frame.insert(0, "generation", np.arange(1, len(history) + 1))
    history_path = os.path.join(
        RESULT_DIR, f"ga_history_{dataset}_seed{seed}.csv"
    )
    _save_csv(history_frame, history_path)
    return rows


def _write_summary(runs, filename="summary.csv"):
    if runs.empty:
        return
    columns = ["n_groups", "n_columns", *METRICS]
    columns.extend(
        name for name in ("noise_groups_kept", "real_groups_kept")
        if name in runs.columns
    )
    stats = runs.groupby(["dataset", "method"])[columns].agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    _save_csv(stats.reset_index(), os.path.join(RESULT_DIR, filename))


def _write_feature_frequency(runs, noise=False):
    if runs.empty:
        return
    for dataset in DATASETS:
        selected = runs[
            (runs["dataset"] == dataset) & (runs["method"] == "GA")
        ]["selected_attributes"]
        if selected.empty:
            continue
        counts = {}
        for value in selected:
            for name in json.loads(value):
                counts[name] = counts.get(name, 0) + 1
        _, _, _, _, group_names = load_dataset(dataset)
        if noise:
            group_names = np.concatenate(
                (
                    group_names,
                    np.asarray(
                        [
                            f"noise_{i + 1}"
                            for i in range(NOISE_COUNTS[dataset])
                        ]
                    ),
                )
            )
        frequency = pd.DataFrame(
            {
                "attribute": group_names,
                "selected_count": [counts.get(name, 0) for name in group_names],
            }
        )
        frequency["frequency"] = frequency["selected_count"] / len(selected)
        _save_csv(
            frequency,
            os.path.join(
                RESULT_DIR,
                f"feature_frequency_{dataset}{'_noise' if noise else ''}.csv",
            ),
        )


def _write_paired_stats(runs, filename="stats.csv"):
    rows = []
    for dataset in DATASETS:
        data = runs[runs["dataset"] == dataset]
        for comparison in ("all_features", "RFE"):
            paired = data[data["method"].isin(("GA", comparison))]
            wide = paired.pivot(index="seed", columns="method", values="balanced_accuracy")
            if not {"GA", comparison}.issubset(wide.columns):
                continue
            wide = wide.dropna(subset=["GA", comparison]).sort_index()
            if len(wide) < 2:
                continue
            ga_scores = wide["GA"].to_numpy()
            other_scores = wide[comparison].to_numpy()
            diff = ga_scores - other_scores
            t_result = ttest_rel(ga_scores, other_scores)
            mean_diff = float(np.mean(diff))
            margin = float(
                t.ppf(0.975, len(diff) - 1)
                * np.std(diff, ddof=1)
                / np.sqrt(len(diff))
            )
            n_nonzero = int(np.count_nonzero(diff))
            if n_nonzero == 0:
                wilcoxon_stat, wilcoxon_p = 0.0, 1.0
            else:
                w_result = wilcoxon(ga_scores, other_scores)
                wilcoxon_stat = float(w_result.statistic)
                wilcoxon_p = float(w_result.pvalue)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"GA_vs_{comparison}",
                    "n_pairs": len(wide),
                    "mean_balanced_accuracy_difference": mean_diff,
                    "ci95_lower": mean_diff - margin,
                    "ci95_upper": mean_diff + margin,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_pvalue": wilcoxon_p,
                    "wilcoxon_n_nonzero_differences": n_nonzero,
                }
            )
    columns = [
        "dataset",
        "comparison",
        "n_pairs",
        "mean_balanced_accuracy_difference",
        "ci95_lower",
        "ci95_upper",
        "paired_t_statistic",
        "paired_t_pvalue",
        "wilcoxon_statistic",
        "wilcoxon_pvalue",
        "wilcoxon_n_nonzero_differences",
    ]
    _save_csv(
        pd.DataFrame(rows, columns=columns),
        os.path.join(RESULT_DIR, filename),
    )


def _print_summary(runs):
    summary = runs.groupby(["dataset", "method"]).agg(
        n_groups=("n_groups", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
    )
    for dataset in DATASETS:
        if dataset not in runs["dataset"].values:
            continue
        dataset_summary = summary.loc[dataset]
        all_groups = int(runs.loc[runs["dataset"] == dataset, "n_groups"].max())
        ga_groups = dataset_summary.loc["GA", "n_groups"]
        reduction = 100.0 * (1.0 - ga_groups / all_groups)
        baseline = dataset_summary.drop(index=["all_features", "GA"])[
            "balanced_accuracy"
        ]
        best_method = baseline.idxmax()
        print(
            f"{dataset}: attributes {all_groups} -> {ga_groups:.1f} "
            f"({reduction:.1f}% reduction); balanced accuracy "
            f"all={dataset_summary.loc['all_features', 'balanced_accuracy']:.4f}, "
            f"GA={dataset_summary.loc['GA', 'balanced_accuracy']:.4f}, "
            f"best baseline={best_method} "
            f"({baseline[best_method]:.4f})"
        )


def run_all():
    os.makedirs(RESULT_DIR, exist_ok=True)
    runs = _load_runs()
    completed = _complete_runs(runs)
    pending = [
        (dataset, seed)
        for dataset in DATASETS
        for seed in SEEDS
        if (dataset, seed) not in completed
    ]
    if not pending:
        print("all dataset and seed runs are already complete")
        _write_summary(runs)
        _write_feature_frequency(runs)
        _write_paired_stats(runs)
        _print_summary(runs)
        return

    sample_dataset, sample_seed = pending.pop(0)
    print(f"timing run: {sample_dataset}, seed={sample_seed}", flush=True)
    sample_start = time.perf_counter()
    sample_rows = _run_seed(sample_dataset, sample_seed)
    sample_runtime = time.perf_counter() - sample_start
    runs = runs[
        ~(
            (runs["dataset"] == sample_dataset)
            & (runs["seed"] == sample_seed)
        )
    ] if not runs.empty else runs
    runs = pd.concat([runs, pd.DataFrame(sample_rows)], ignore_index=True)
    _save_csv(runs, RUNS_PATH)
    _write_summary(runs)
    _write_feature_frequency(runs)
    _write_paired_stats(runs)

    estimated_total = sample_runtime * (len(pending) + 1)
    print(
        f"timed run took {sample_runtime / 60:.1f} min; estimated total "
        f"for {len(pending) + 1} remaining dataset-seed runs: "
        f"{estimated_total / 60:.1f} min",
        flush=True,
    )

    for dataset, seed in pending:
        print(f"running {dataset}, seed={seed}", flush=True)
        rows = _run_seed(dataset, seed)
        runs = pd.concat([runs, pd.DataFrame(rows)], ignore_index=True)
        _save_csv(runs, RUNS_PATH)
        _write_summary(runs)
        _write_feature_frequency(runs)
        _write_paired_stats(runs)

    _print_summary(runs)


def _load_noise_dataset(dataset):
    X, y, _, groups, group_names = load_dataset(dataset)
    n_noise = NOISE_COUNTS[dataset]
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.standard_normal((X.shape[0], n_noise))
    noise_names = np.asarray([f"noise_{i + 1}" for i in range(n_noise)])
    noise_groups = np.arange(len(group_names), len(group_names) + n_noise)
    return (
        np.column_stack((X, noise)),
        y,
        np.concatenate((groups, noise_groups)),
        np.concatenate((group_names, noise_names)),
    )


def _run_noise_seed(dataset, seed):
    X, y, groups, group_names = _load_noise_dataset(dataset)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)

    start = time.perf_counter()
    ga_mask, history = run_ga(
        X_tr,
        y_tr,
        groups,
        seed=seed,
        pop_size=30,
        n_gen=40,
        alpha=0.01,
        metric="balanced_accuracy",
    )
    ga_runtime = time.perf_counter() - start
    ga_generations = len(history)
    k = int(ga_mask.sum())
    masks = {
        "all_features": np.ones(len(group_names), dtype=bool),
        "GA": ga_mask,
        "MI": mutual_info_mask(X_tr, y_tr, groups, k, seed=seed),
        "RFE": rfe_mask(X_tr, y_tr, groups, k, seed=seed),
    }

    rows = []
    for method, mask in masks.items():
        column_mask = expand_group_mask(mask, groups)
        metrics = evaluate(X_tr, y_tr, X_te, y_te, column_mask, seed)
        row = _method_row(
            dataset,
            seed,
            method,
            mask,
            groups,
            metrics,
            ga_runtime,
            ga_generations,
            group_names[mask].tolist(),
        )
        row["noise_groups_kept"] = int(
            np.sum(mask & np.char.startswith(group_names.astype(str), "noise_"))
        )
        row["real_groups_kept"] = int(mask.sum() - row["noise_groups_kept"])
        rows.append(row)

    random_scores = []
    random_column_counts = []
    random_noise_counts = []
    for draw in range(10):
        mask = random_mask(len(group_names), k, seed=seed * 10 + draw)
        column_mask = expand_group_mask(mask, groups)
        random_scores.append(evaluate(X_tr, y_tr, X_te, y_te, column_mask, seed))
        random_column_counts.append(int(column_mask.sum()))
        random_noise_counts.append(
            int(np.sum(mask & np.char.startswith(group_names.astype(str), "noise_")))
        )
    random_metrics = {
        metric: float(np.mean([score[metric] for score in random_scores]))
        for metric in METRICS
    }
    random_mask_for_row = np.zeros(len(group_names), dtype=bool)
    random_mask_for_row[:k] = True
    random_row = _method_row(
        dataset,
        seed,
        "Random",
        random_mask_for_row,
        groups,
        random_metrics,
        ga_runtime,
        ga_generations,
        [],
        n_columns=float(np.mean(random_column_counts)),
    )
    random_row["noise_groups_kept"] = float(np.mean(random_noise_counts))
    random_row["real_groups_kept"] = float(k - random_row["noise_groups_kept"])
    rows.append(random_row)

    history_frame = pd.DataFrame(history)
    history_frame.insert(0, "generation", np.arange(1, len(history) + 1))
    history_path = os.path.join(
        RESULT_DIR, f"ga_history_noise_{dataset}_seed{seed}.csv"
    )
    _save_csv(history_frame, history_path)
    return rows


def _save_noise_outputs(runs):
    _save_csv(runs, NOISE_RUNS_PATH)
    _write_summary(runs, filename="noise_summary.csv")
    _write_feature_frequency(runs, noise=True)
    _write_paired_stats(runs, filename="noise_stats.csv")


def run_noise():
    os.makedirs(RESULT_DIR, exist_ok=True)
    runs = _load_runs(NOISE_RUNS_PATH)
    completed = _complete_runs(runs, methods=NOISE_METHODS)
    pending = [
        (dataset, seed)
        for dataset in DATASETS
        for seed in NOISE_SEEDS
        if (dataset, seed) not in completed
    ]
    if not pending:
        print("all noise dataset and seed runs are already complete")
        _save_noise_outputs(runs)
        return

    dataset, seed = pending.pop(0)
    print(f"timing noise run: {dataset}, seed={seed}", flush=True)
    start = time.perf_counter()
    rows = _run_noise_seed(dataset, seed)
    elapsed = time.perf_counter() - start
    runs = runs[
        ~((runs["dataset"] == dataset) & (runs["seed"] == seed))
    ] if not runs.empty else runs
    runs = pd.concat([runs, pd.DataFrame(rows)], ignore_index=True)
    _save_noise_outputs(runs)

    estimate = elapsed * (len(pending) + 1)
    print(
        f"timed noise run took {elapsed / 60:.1f} min; estimated total "
        f"for {len(pending) + 1} remaining dataset-seed runs: "
        f"{estimate / 60:.1f} min",
        flush=True,
    )

    for dataset, seed in pending:
        print(f"running noise {dataset}, seed={seed}", flush=True)
        rows = _run_noise_seed(dataset, seed)
        runs = pd.concat([runs, pd.DataFrame(rows)], ignore_index=True)
        _save_noise_outputs(runs)