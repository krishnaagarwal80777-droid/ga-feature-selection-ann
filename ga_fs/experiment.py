import json
import os
import shutil
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
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ga_fs.ann import build_ann, cv_score, evaluate
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
SENSITIVITY_RUNS_PATH = os.path.join(RESULT_DIR, "sensitivity_runs.csv")
SENSITIVITY_SEEDS = range(5)
SENSITIVITY_ALPHAS = (0, 0.005, 0.01, 0.02, 0.05, 0.1)
NOISE_SENSITIVITY_ALPHAS = (0, 0.01, 0.05, 0.1, 0.2)
COMPARISONS = ("all_features", "MI", "RFE", "PCA", "Random")
# nadeau-bengio n_test / n_train
TEST_TRAIN_RATIO = 0.2 / 0.8


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


def _holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def _paired_row(dataset, comparison, ga, other):
    diff = ga - other
    n = len(diff)
    mean = float(diff.mean())
    sd = float(np.std(diff, ddof=1))
    crit = t.ppf(0.975, n - 1)
    nb_se = sd * np.sqrt(1.0 / n + TEST_TRAIN_RATIO)
    if sd == 0 and mean == 0:
        t_stat, t_p, nb_stat, nb_p = 0.0, 1.0, 0.0, 1.0
    else:
        t_stat, t_p = ttest_rel(ga, other)
        nb_stat = mean / nb_se
        nb_p = 2 * t.sf(abs(nb_stat), n - 1)

    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = wilcoxon(ga, other)

    return {
        "dataset": dataset,
        "comparison": f"GA_vs_{comparison}",
        "n_pairs": n,
        "mean_balanced_accuracy_difference": mean,
        "ci95_lower": mean - crit * sd / np.sqrt(n),
        "ci95_upper": mean + crit * sd / np.sqrt(n),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "nb_t_statistic": float(nb_stat),
        "nb_pvalue": float(nb_p),
        "nb_ci95_lower": mean - crit * nb_se,
        "nb_ci95_upper": mean + crit * nb_se,
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "wilcoxon_n_nonzero_differences": n_nonzero,
        "min_possible_wilcoxon_p": min(1.0, 2.0 ** (1 - n_nonzero)),
    }


def _write_paired_stats(runs, filename="stats.csv"):
    frames = []
    for dataset in DATASETS:
        data = runs[runs["dataset"] == dataset]
        wide = data.pivot(index="seed", columns="method", values="balanced_accuracy")
        rows = []
        for comparison in COMPARISONS:
            if "GA" not in wide.columns or comparison not in wide.columns:
                continue
            pair = wide[["GA", comparison]].dropna().sort_index()
            if len(pair) < 2:
                continue
            rows.append(
                _paired_row(
                    dataset, comparison, pair["GA"].to_numpy(),
                    pair[comparison].to_numpy(),
                )
            )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        for col in ("paired_t_pvalue", "nb_pvalue", "wilcoxon_pvalue"):
            frame[f"{col}_holm"] = _holm(frame[col])
        frames.append(frame)
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _save_csv(stats, os.path.join(RESULT_DIR, filename))


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


def _fixed_baseline_rows(dataset, seed, noise, ga_row):
    if noise:
        X, y, groups, group_names = _load_noise_dataset(dataset)
    else:
        X, y, _, groups, group_names = load_dataset(dataset)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)
    ga_mask = np.isin(group_names, json.loads(ga_row["selected_attributes"]))
    k = int(ga_mask.sum())
    if k != int(ga_row["n_groups"]):
        raise ValueError(f"ga mask mismatch: {dataset} seed={seed}")
    runtime, gens = ga_row["ga_runtime"], ga_row["ga_generations"]

    mi = mutual_info_mask(X_tr, y_tr, groups, k, seed=seed)
    metrics = evaluate(
        X_tr, y_tr, X_te, y_te, expand_group_mask(mi, groups), seed
    )
    rows = {
        "MI": _method_row(
            dataset, seed, "MI", mi, groups, metrics, runtime, gens,
            group_names[mi].tolist(),
        )
    }
    if noise:
        n_noise = int(
            np.sum(mi & np.char.startswith(group_names.astype(str), "noise_"))
        )
        rows["MI"]["noise_groups_kept"] = n_noise
        rows["MI"]["real_groups_kept"] = k - n_noise
        return rows

    X_tr_pca, X_te_pca = pca_transform(X_tr, X_te, k, seed=seed)
    ones = np.ones(k, dtype=bool)
    metrics = evaluate(X_tr_pca, y_tr, X_te_pca, y_te, ones, seed)
    rows["PCA"] = _method_row(
        dataset, seed, "PCA", ones, np.arange(k), metrics, runtime, gens, [],
        n_columns=k,
    )
    return rows


def _rerun_file(path, noise):
    backup = path.replace(".csv", "_before_baseline_fix.csv")
    # never overwrite the original backup on a second run
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    # round_trip so untouched rows keep exact float digits
    old = pd.read_csv(backup, float_precision="round_trip")
    runs = pd.read_csv(path, float_precision="round_trip")
    for _, ga_row in runs[runs["method"] == "GA"].iterrows():
        dataset, seed = ga_row["dataset"], int(ga_row["seed"])
        print(f"rerun {dataset} seed={seed} noise={noise}", flush=True)
        for method, row in _fixed_baseline_rows(dataset, seed, noise, ga_row).items():
            idx = runs.index[
                (runs["dataset"] == dataset)
                & (runs["seed"] == seed)
                & (runs["method"] == method)
            ]
            if len(idx) != 1:
                raise ValueError(f"expected one {method} row: {dataset} seed={seed}")
            runs.loc[idx[0], list(row)] = list(row.values())
        _save_csv(runs, path)
    return old, runs


def _print_baseline_change(old, new, methods):
    for dataset in DATASETS:
        for method in methods:
            before = old[(old["dataset"] == dataset) & (old["method"] == method)]
            after = new[(new["dataset"] == dataset) & (new["method"] == method)]
            print(
                f"{dataset} {method}: {before['balanced_accuracy'].mean():.4f}"
                f" -> {after['balanced_accuracy'].mean():.4f}"
            )


def rerun_baselines():
    old, runs = _rerun_file(RUNS_PATH, noise=False)
    _write_summary(runs)
    _write_paired_stats(runs)
    noise_old, noise_runs = _rerun_file(NOISE_RUNS_PATH, noise=True)
    _write_summary(noise_runs, filename="noise_summary.csv")
    _write_paired_stats(noise_runs, filename="noise_stats.csv")
    print("main runs")
    _print_baseline_change(old, runs, ("MI", "PCA"))
    print("noise runs")
    _print_baseline_change(noise_old, noise_runs, ("MI",))


def _sensitivity_specs(dataset, noise, n_groups):
    base_mutation = 1.0 / n_groups
    if noise:
        return [
            ("alpha", alpha, alpha, base_mutation, 30)
            for alpha in NOISE_SENSITIVITY_ALPHAS
        ]

    specs = [
        ("alpha", alpha, alpha, base_mutation, 30)
        for alpha in SENSITIVITY_ALPHAS
    ]
    specs.extend(
        ("mutation_rate", rate, 0.01, rate, 30)
        for rate in (0.5 / n_groups, base_mutation, 2.0 / n_groups)
    )
    specs.extend(
        ("pop_size", size, 0.01, base_mutation, size)
        for size in (10, 20, 30, 50)
    )
    return specs


def _sensitivity_config_key(row):
    return (
        row["dataset"],
        bool(row["noise"]),
        int(row["seed"]),
        round(float(row["alpha"]), 12),
        round(float(row["mutation_rate"]), 12),
        int(row["pop_size"]),
    )


def _source_ga_row(dataset, noise, seed):
    path = NOISE_RUNS_PATH if noise else RUNS_PATH
    if not os.path.exists(path):
        return None

    source = pd.read_csv(path)
    rows = source[
        (source["dataset"] == dataset)
        & (source["method"] == "GA")
        & (pd.to_numeric(source["seed"]) == seed)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _sensitivity_arrays(dataset, noise, seed):
    if noise:
        X, y, groups, group_names = _load_noise_dataset(dataset)
    else:
        X, y, _, groups, group_names = load_dataset(dataset)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)
    raw_X_tr, _, raw_y_tr, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    return X_tr, X_te, y_tr, y_te, groups, group_names, raw_X_tr, raw_y_tr


def _sensitivity_scores(dataset, noise, seed, mask, arrays):
    X_tr, X_te, y_tr, y_te, groups, _, raw_X_tr, raw_y_tr = arrays
    column_mask = expand_group_mask(mask, groups)
    fitness_cv = cv_score(
        X_tr,
        y_tr,
        column_mask,
        seed,
        metric="balanced_accuracy",
        fast=True,
    )

    model = make_pipeline(StandardScaler(), build_ann(seed))
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=seed + 1000
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        independent_scores = cross_val_score(
            model,
            raw_X_tr[:, column_mask],
            raw_y_tr,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )

    test_metrics = evaluate(X_tr, y_tr, X_te, y_te, column_mask, seed)
    return fitness_cv, float(independent_scores.mean()), test_metrics[
        "balanced_accuracy"
    ]


def _sensitivity_record(
    dataset,
    noise,
    seed,
    parameter,
    value,
    alpha,
    mutation_rate,
    pop_size,
    mask,
    groups,
    group_names,
    fitness_cv,
    independent_cv,
    test_score,
    ga_runtime,
    ga_generations,
):
    noise_mask = np.char.startswith(group_names.astype(str), "noise_")
    noise_kept = int(np.sum(mask & noise_mask)) if noise else 0
    return {
        "dataset": dataset,
        "noise": bool(noise),
        "seed": seed,
        "parameter": parameter,
        "value": float(value),
        "alpha": float(alpha),
        "mutation_rate": float(mutation_rate),
        "pop_size": int(pop_size),
        "n_groups_selected": int(mask.sum()),
        "noise_groups_kept": noise_kept,
        "real_groups_kept": int(mask.sum() - noise_kept),
        "ga_fitness_cv": float(fitness_cv),
        "independent_train_cv": float(independent_cv),
        "test_balanced_accuracy": float(test_score),
        "ga_runtime": float(ga_runtime),
        "ga_generations": int(ga_generations),
        "selected_attributes": json.dumps(group_names[mask].tolist()),
    }


def _reuse_default_sensitivity(dataset, noise, seed, arrays):
    source = _source_ga_row(dataset, noise, seed)
    if source is None:
        return None

    group_names = arrays[5]
    selected_names = json.loads(source["selected_attributes"])
    mask = np.isin(group_names, selected_names)
    if int(mask.sum()) != int(source["n_groups"]):
        return None

    fitness_cv, independent_cv, _ = _sensitivity_scores(
        dataset, noise, seed, mask, arrays
    )
    return {
        "mask": mask,
        "fitness_cv": fitness_cv,
        "independent_cv": independent_cv,
        "test_score": float(source["balanced_accuracy"]),
        "ga_runtime": float(source["ga_runtime"]),
        "ga_generations": int(source["ga_generations"]),
    }


def _run_sensitivity_setting(
    dataset,
    noise,
    seed,
    parameter,
    value,
    alpha,
    mutation_rate,
    pop_size,
    arrays,
    cached_result,
):
    X_tr, X_te, y_tr, y_te, groups, group_names = arrays[:6]
    if cached_result is None:
        start = time.perf_counter()
        mask, history = run_ga(
            X_tr,
            y_tr,
            groups,
            seed=seed,
            pop_size=pop_size,
            n_gen=40,
            alpha=alpha,
            metric="balanced_accuracy",
            mutation_rate=mutation_rate,
        )
        ga_runtime = time.perf_counter() - start
        fitness_cv, independent_cv, test_score = _sensitivity_scores(
            dataset, noise, seed, mask, arrays
        )
        ga_generations = len(history)
        cached_result = {
            "mask": mask,
            "fitness_cv": fitness_cv,
            "independent_cv": independent_cv,
            "test_score": test_score,
            "ga_runtime": ga_runtime,
            "ga_generations": ga_generations,
        }

    row = _sensitivity_record(
        dataset,
        noise,
        seed,
        parameter,
        value,
        alpha,
        mutation_rate,
        pop_size,
        cached_result["mask"],
        groups,
        group_names,
        cached_result["fitness_cv"],
        cached_result["independent_cv"],
        cached_result["test_score"],
        cached_result["ga_runtime"],
        cached_result["ga_generations"],
    )
    return row, cached_result


def _write_sensitivity_summary(runs):
    if runs.empty:
        return
    group_cols = ["dataset", "noise", "parameter", "value"]
    value_cols = [
        "n_groups_selected",
        "noise_groups_kept",
        "real_groups_kept",
        "ga_fitness_cv",
        "independent_train_cv",
        "test_balanced_accuracy",
        "ga_runtime",
        "ga_generations",
    ]
    summary = runs.groupby(group_cols)[value_cols].agg(["mean", "std"])
    summary.columns = [
        f"{column}_{stat}" for column, stat in summary.columns
    ]
    _save_csv(
        summary.reset_index(),
        os.path.join(RESULT_DIR, "sensitivity_summary.csv"),
    )


def run_sensitivity():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if os.path.exists(SENSITIVITY_RUNS_PATH):
        runs = pd.read_csv(SENSITIVITY_RUNS_PATH)
    else:
        runs = pd.DataFrame()
    existing = set()
    result_cache = {}
    if not runs.empty:
        for _, row in runs.iterrows():
            identity = (
                row["dataset"],
                bool(row["noise"]),
                int(row["seed"]),
                row["parameter"],
                round(float(row["value"]), 12),
            )
            existing.add(identity)
            result_cache[_sensitivity_config_key(row)] = {
                "fitness_cv": float(row["ga_fitness_cv"]),
                "independent_cv": float(row["independent_train_cv"]),
                "test_score": float(row["test_balanced_accuracy"]),
                "ga_runtime": float(row["ga_runtime"]),
                "ga_generations": int(row["ga_generations"]),
                "selected_attributes": json.loads(row["selected_attributes"]),
            }

    total = sum(
        len(_sensitivity_specs(dataset, noise, 20 if dataset == "german_credit" else 30))
        * len(SENSITIVITY_SEEDS)
        for dataset in DATASETS
        for noise in (False, True)
    )
    completed = 0
    for noise in (False, True):
        for dataset in DATASETS:
            n_groups = (20 if dataset == "german_credit" else 30)
            if noise:
                n_groups += NOISE_COUNTS[dataset]
            specs = _sensitivity_specs(dataset, noise, n_groups)
            for seed in SENSITIVITY_SEEDS:
                missing_specs = [
                    spec
                    for spec in specs
                    if (
                        dataset,
                        noise,
                        seed,
                        spec[0],
                        round(float(spec[1]), 12),
                    )
                    not in existing
                ]
                if not missing_specs:
                    completed += len(specs)
                    continue

                arrays = _sensitivity_arrays(dataset, noise, seed)
                default_config = (
                    dataset,
                    noise,
                    seed,
                    0.01,
                    round(1.0 / n_groups, 12),
                    30,
                )
                needs_default = any(
                    alpha == 0.01
                    and pop_size == 30
                    and np.isclose(mutation_rate, 1.0 / n_groups)
                    for _, _, alpha, mutation_rate, pop_size in missing_specs
                )
                source_cache = None
                if needs_default and default_config not in result_cache:
                    source_cache = _reuse_default_sensitivity(
                        dataset, noise, seed, arrays
                    )
                for parameter, value, alpha, mutation_rate, pop_size in specs:
                    identity = (
                        dataset,
                        noise,
                        seed,
                        parameter,
                        round(float(value), 12),
                    )
                    if identity in existing:
                        completed += 1
                        continue

                    config = (
                        dataset,
                        noise,
                        seed,
                        round(float(alpha), 12),
                        round(float(mutation_rate), 12),
                        int(pop_size),
                    )
                    cached = result_cache.get(config)
                    if cached is not None:
                        cached = dict(cached)
                        cached["mask"] = np.isin(
                            arrays[5], cached["selected_attributes"]
                        )
                    if cached is None and (
                        alpha == 0.01
                        and pop_size == 30
                        and np.isclose(mutation_rate, 1.0 / n_groups)
                    ):
                        cached = source_cache

                    row, result = _run_sensitivity_setting(
                        dataset,
                        noise,
                        seed,
                        parameter,
                        value,
                        alpha,
                        mutation_rate,
                        pop_size,
                        arrays,
                        cached,
                    )
                    result_cache[config] = result
                    if "selected_attributes" not in result_cache[config]:
                        result_cache[config]["selected_attributes"] = json.loads(
                            row["selected_attributes"]
                        )
                    result_cache[config]["mask"] = np.isin(
                        arrays[5], result_cache[config]["selected_attributes"]
                    )
                    runs = pd.concat(
                        [runs, pd.DataFrame([row])], ignore_index=True
                    )
                    existing.add(identity)
                    completed += 1
                    _save_csv(runs, SENSITIVITY_RUNS_PATH)
                    _write_sensitivity_summary(runs)
                    print(
                        f"sensitivity {completed}/{total}: {dataset}, "
                        f"noise={noise}, seed={seed}, "
                        f"{parameter}={value}",
                        flush=True,
                    )

    _write_sensitivity_summary(runs)