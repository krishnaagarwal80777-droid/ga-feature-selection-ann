import json
import os
import shutil
import time
import warnings

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
EXPERIMENTS = {
    False: {"path": RUNS_PATH, "seeds": SEEDS, "methods": METHODS, "prefix": ""},
    True: {
        "path": NOISE_RUNS_PATH,
        "seeds": NOISE_SEEDS,
        "methods": NOISE_METHODS,
        "prefix": "noise_",
    },
}


def _save_csv(frame, path):
    tmp = f"{path}.tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_runs(path=RUNS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, float_precision="round_trip")


def _load_noise_dataset(dataset):
    X, y, _, groups, names = load_dataset(dataset)
    n_noise = NOISE_COUNTS[dataset]
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.standard_normal((X.shape[0], n_noise))
    noise_names = np.asarray([f"noise_{i + 1}" for i in range(n_noise)])
    noise_groups = np.arange(len(names), len(names) + n_noise)
    return (
        np.column_stack((X, noise)),
        y,
        np.concatenate((groups, noise_groups)),
        np.concatenate((names, noise_names)),
    )


def _load_data(dataset, noise):
    if noise:
        return _load_noise_dataset(dataset)
    X, y, _, groups, names = load_dataset(dataset)
    return X, y, groups, names


def _noise_kept(mask, names):
    return int(np.sum(mask & np.char.startswith(names.astype(str), "noise_")))


def _complete_runs(frame, methods=METHODS):
    complete = set()
    if frame.empty:
        return complete

    for (dataset, seed), rows in frame.groupby(["dataset", "seed"]):
        if set(rows["method"]) >= set(methods):
            complete.add((dataset, int(seed)))
    return complete


def _method_row(ctx, method, mask, metrics, selected, n_columns=None):
    if n_columns is None:
        n_columns = int(expand_group_mask(mask, ctx["groups"]).sum())

    return {
        "dataset": ctx["dataset"],
        "seed": ctx["seed"],
        "method": method,
        "n_groups": int(mask.sum()),
        "n_columns": int(n_columns),
        **{name: float(metrics[name]) for name in METRICS},
        "ga_runtime": float(ctx["ga_runtime"]),
        "ga_generations": int(ctx["ga_generations"]),
        "selected_attributes": json.dumps(selected),
    }


def _eval_mask(ctx, mask):
    X_tr, X_te, y_tr, y_te = ctx["split"]
    cols = expand_group_mask(mask, ctx["groups"])
    return evaluate(X_tr, y_tr, X_te, y_te, cols, ctx["seed"])


def _selector_rows(ctx, masks):
    rows = []
    for method, mask in masks.items():
        row = _method_row(
            ctx, method, mask, _eval_mask(ctx, mask), ctx["names"][mask].tolist()
        )
        if ctx["noise"]:
            row["noise_groups_kept"] = _noise_kept(mask, ctx["names"])
            row["real_groups_kept"] = int(mask.sum() - row["noise_groups_kept"])
        rows.append(row)
    return rows


def _random_row(ctx, k):
    n = len(ctx["names"])
    scores, n_cols, n_noise = [], [], []
    for draw in range(10):
        mask = random_mask(n, k, seed=ctx["seed"] * 10 + draw)
        scores.append(_eval_mask(ctx, mask))
        n_cols.append(int(expand_group_mask(mask, ctx["groups"]).sum()))
        n_noise.append(_noise_kept(mask, ctx["names"]))
    metrics = {m: float(np.mean([s[m] for s in scores])) for m in METRICS}
    row = _method_row(
        ctx, "Random", np.arange(n) < k, metrics, [],
        n_columns=float(np.mean(n_cols)),
    )
    if ctx["noise"]:
        row["noise_groups_kept"] = float(np.mean(n_noise))
        row["real_groups_kept"] = float(k - row["noise_groups_kept"])
    return row


def _pca_row(ctx, k):
    X_tr, X_te, y_tr, y_te = ctx["split"]
    X_tr_pca, X_te_pca = pca_transform(X_tr, X_te, k, seed=ctx["seed"])
    ones = np.ones(k, dtype=bool)
    metrics = evaluate(X_tr_pca, y_tr, X_te_pca, y_te, ones, ctx["seed"])
    return _method_row(ctx, "PCA", ones, metrics, [], n_columns=k)


def _save_history(history, dataset, seed, noise):
    frame = pd.DataFrame(history)
    frame.insert(0, "generation", np.arange(1, len(history) + 1))
    prefix = EXPERIMENTS[noise]["prefix"]
    _save_csv(frame, os.path.join(RESULT_DIR, f"ga_history_{prefix}{dataset}_seed{seed}.csv"))


def _run_seed(dataset, seed, noise=False):
    X, y, groups, names = _load_data(dataset, noise)
    split = get_splits(X, y, seed)
    X_tr, _, y_tr, _ = split

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
    ctx = {
        "dataset": dataset, "seed": seed, "noise": noise, "split": split,
        "groups": groups, "names": names,
        "ga_runtime": time.perf_counter() - start, "ga_generations": len(history),
    }
    k = int(ga_mask.sum())
    masks = {
        "all_features": np.ones(len(names), dtype=bool),
        "GA": ga_mask,
        "MI": mutual_info_mask(X_tr, y_tr, groups, k, seed=seed),
        "RFE": rfe_mask(X_tr, y_tr, groups, k, seed=seed),
    }
    rows = _selector_rows(ctx, masks)
    rows.append(_random_row(ctx, k))
    if not noise:
        rows.append(_pca_row(ctx, k))
    _save_history(history, dataset, seed, noise)
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
        names = _load_data(dataset, noise)[3]
        frequency = pd.DataFrame(
            {
                "attribute": names,
                "selected_count": [counts.get(name, 0) for name in names],
            }
        )
        frequency["frequency"] = frequency["selected_count"] / len(selected)
        suffix = "_noise" if noise else ""
        _save_csv(
            frequency,
            os.path.join(RESULT_DIR, f"feature_frequency_{dataset}{suffix}.csv"),
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
    if runs.empty:
        return
    means = runs.groupby(["dataset", "method"])[["n_groups", "balanced_accuracy"]].mean()
    for dataset in DATASETS:
        if dataset not in runs["dataset"].values:
            continue
        m = means.loc[dataset]
        acc = m["balanced_accuracy"]
        best = acc.drop(index=["all_features", "GA"]).idxmax()
        print(
            f"{dataset}: attrs {m.loc['all_features', 'n_groups']:.0f}"
            f" -> {m.loc['GA', 'n_groups']:.1f}, all={acc['all_features']:.4f}"
            f" ga={acc['GA']:.4f} {best}={acc[best]:.4f}"
        )


def _write_derived(runs, noise):
    prefix = EXPERIMENTS[noise]["prefix"]
    _write_summary(runs, filename=f"{prefix}summary.csv")
    _write_feature_frequency(runs, noise=noise)
    _write_paired_stats(runs, filename=f"{prefix}stats.csv")


def _run_experiment(noise):
    cfg = EXPERIMENTS[noise]
    os.makedirs(RESULT_DIR, exist_ok=True)
    runs = _load_runs(cfg["path"])
    done = _complete_runs(runs, methods=cfg["methods"])
    pending = [
        (dataset, seed)
        for dataset in DATASETS
        for seed in cfg["seeds"]
        if (dataset, seed) not in done
    ]
    if not pending:
        print("all runs done")
        _write_derived(runs, noise)
    for i, (dataset, seed) in enumerate(pending):
        print(f"run {dataset} seed={seed}", flush=True)
        start = time.perf_counter()
        rows = _run_seed(dataset, seed, noise)
        if not runs.empty:
            runs = runs[~((runs["dataset"] == dataset) & (runs["seed"] == seed))]
        runs = pd.concat([runs, pd.DataFrame(rows)], ignore_index=True)
        _save_csv(runs, cfg["path"])
        _write_derived(runs, noise)
        left = len(pending) - i - 1
        if i == 0 and left:
            mins = (time.perf_counter() - start) / 60
            print(f"took {mins:.1f} min, ~{mins * left:.0f} min for {left} more", flush=True)
    _print_summary(runs)


def run_all():
    _run_experiment(noise=False)


def run_noise():
    _run_experiment(noise=True)


def _fixed_baseline_rows(dataset, seed, noise, ga_row):
    X, y, groups, names = _load_data(dataset, noise)
    split = get_splits(X, y, seed)
    X_tr, _, y_tr, _ = split
    ga_mask = np.isin(names, json.loads(ga_row["selected_attributes"]))
    k = int(ga_mask.sum())
    if k != int(ga_row["n_groups"]):
        raise ValueError(f"ga mask mismatch: {dataset} seed={seed}")
    ctx = {
        "dataset": dataset, "seed": seed, "noise": noise, "split": split,
        "groups": groups, "names": names,
        "ga_runtime": ga_row["ga_runtime"], "ga_generations": ga_row["ga_generations"],
    }
    mi = mutual_info_mask(X_tr, y_tr, groups, k, seed=seed)
    rows = {"MI": _selector_rows(ctx, {"MI": mi})[0]}
    if not noise:
        rows["PCA"] = _pca_row(ctx, k)
    return rows


def _rerun_file(path, noise):
    backup = path.replace(".csv", "_before_baseline_fix.csv")
    # never overwrite the original backup on a second run
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    old = _load_runs(backup)
    runs = _load_runs(path)
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


def _spec(parameter, value, rate, alpha=0.01, pop_size=30):
    return {
        "parameter": parameter,
        "value": value,
        "alpha": alpha,
        "mutation_rate": rate,
        "pop_size": pop_size,
    }


def _sensitivity_specs(noise, n_groups):
    base = 1.0 / n_groups
    alphas = NOISE_SENSITIVITY_ALPHAS if noise else SENSITIVITY_ALPHAS
    specs = [_spec("alpha", a, base, alpha=a) for a in alphas]
    if noise:
        return specs
    specs += [
        _spec("mutation_rate", r, r) for r in (0.5 / n_groups, base, 2.0 / n_groups)
    ]
    specs += [_spec("pop_size", s, base, pop_size=s) for s in (10, 20, 30, 50)]
    return specs


def _row_id(dataset, noise, seed, spec):
    return (dataset, bool(noise), int(seed), spec["parameter"], round(float(spec["value"]), 12))


def _config_key(dataset, noise, seed, spec):
    return (
        dataset,
        bool(noise),
        int(seed),
        round(float(spec["alpha"]), 12),
        round(float(spec["mutation_rate"]), 12),
        int(spec["pop_size"]),
    )


def _sensitivity_data(dataset, noise, seed):
    X, y, groups, names = _load_data(dataset, noise)
    X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)
    raw_X_tr, _, raw_y_tr, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return {
        "dataset": dataset, "noise": noise, "seed": seed,
        "split": (X_tr, X_te, y_tr, y_te), "groups": groups, "names": names,
        "raw": (raw_X_tr, raw_y_tr),
    }


def _sensitivity_scores(data, mask):
    X_tr, X_te, y_tr, y_te = data["split"]
    raw_X_tr, raw_y_tr = data["raw"]
    seed = data["seed"]
    cols = expand_group_mask(mask, data["groups"])
    fitness_cv = cv_score(X_tr, y_tr, cols, seed, metric="balanced_accuracy", fast=True)

    model = make_pipeline(StandardScaler(), build_ann(seed))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed + 1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(
            model, raw_X_tr[:, cols], raw_y_tr, cv=cv,
            scoring="balanced_accuracy", n_jobs=-1,
        )

    test = evaluate(X_tr, y_tr, X_te, y_te, cols, seed)
    return fitness_cv, float(scores.mean()), test["balanced_accuracy"]


def _sensitivity_ga(data, spec):
    X_tr, _, y_tr, _ = data["split"]
    start = time.perf_counter()
    mask, history = run_ga(
        X_tr,
        y_tr,
        data["groups"],
        seed=data["seed"],
        pop_size=spec["pop_size"],
        n_gen=40,
        alpha=spec["alpha"],
        metric="balanced_accuracy",
        mutation_rate=spec["mutation_rate"],
    )
    runtime = time.perf_counter() - start
    fitness_cv, independent_cv, test_score = _sensitivity_scores(data, mask)
    return {
        "selected": data["names"][mask].tolist(),
        "fitness_cv": fitness_cv,
        "independent_cv": independent_cv,
        "test_score": test_score,
        "ga_runtime": runtime,
        "ga_generations": len(history),
    }


def _reuse_default(data):
    path = NOISE_RUNS_PATH if data["noise"] else RUNS_PATH
    if not os.path.exists(path):
        return None
    source = pd.read_csv(path)
    rows = source[
        (source["dataset"] == data["dataset"])
        & (source["method"] == "GA")
        & (pd.to_numeric(source["seed"]) == data["seed"])
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    selected = json.loads(row["selected_attributes"])
    mask = np.isin(data["names"], selected)
    if int(mask.sum()) != int(row["n_groups"]):
        return None

    fitness_cv, independent_cv, _ = _sensitivity_scores(data, mask)
    return {
        "selected": selected,
        "fitness_cv": fitness_cv,
        "independent_cv": independent_cv,
        "test_score": float(row["balanced_accuracy"]),
        "ga_runtime": float(row["ga_runtime"]),
        "ga_generations": int(row["ga_generations"]),
    }


def _sensitivity_result(data, spec, cache):
    key = _config_key(data["dataset"], data["noise"], data["seed"], spec)
    if key in cache:
        return cache[key]
    result = None
    # the default config already ran in run_all / run_noise
    default = _spec("alpha", 0.01, 1.0 / len(data["names"]))
    if key == _config_key(data["dataset"], data["noise"], data["seed"], default):
        result = _reuse_default(data)
    if result is None:
        result = _sensitivity_ga(data, spec)
    cache[key] = result
    return result


def _sensitivity_record(data, spec, result):
    mask = np.isin(data["names"], result["selected"])
    noise_kept = _noise_kept(mask, data["names"]) if data["noise"] else 0
    return {
        "dataset": data["dataset"],
        "noise": bool(data["noise"]),
        "seed": data["seed"],
        "parameter": spec["parameter"],
        "value": float(spec["value"]),
        "alpha": float(spec["alpha"]),
        "mutation_rate": float(spec["mutation_rate"]),
        "pop_size": int(spec["pop_size"]),
        "n_groups_selected": int(mask.sum()),
        "noise_groups_kept": noise_kept,
        "real_groups_kept": int(mask.sum() - noise_kept),
        "ga_fitness_cv": float(result["fitness_cv"]),
        "independent_train_cv": float(result["independent_cv"]),
        "test_balanced_accuracy": float(result["test_score"]),
        "ga_runtime": float(result["ga_runtime"]),
        "ga_generations": int(result["ga_generations"]),
        "selected_attributes": json.dumps(data["names"][mask].tolist()),
    }


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
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    _save_csv(summary.reset_index(), os.path.join(RESULT_DIR, "sensitivity_summary.csv"))


def _load_sensitivity_state():
    runs = _load_runs(SENSITIVITY_RUNS_PATH)
    state = {"runs": runs, "done": set(), "cache": {}}
    for _, row in runs.iterrows():
        args = (row["dataset"], row["noise"], row["seed"], row)
        state["done"].add(_row_id(*args))
        state["cache"][_config_key(*args)] = {
            "selected": json.loads(row["selected_attributes"]),
            "fitness_cv": float(row["ga_fitness_cv"]),
            "independent_cv": float(row["independent_train_cv"]),
            "test_score": float(row["test_balanced_accuracy"]),
            "ga_runtime": float(row["ga_runtime"]),
            "ga_generations": int(row["ga_generations"]),
        }
    return state


def _sensitivity_seed(dataset, noise, seed, specs, state):
    todo = [s for s in specs if _row_id(dataset, noise, seed, s) not in state["done"]]
    if not todo:
        return
    data = _sensitivity_data(dataset, noise, seed)
    for spec in todo:
        result = _sensitivity_result(data, spec, state["cache"])
        row = _sensitivity_record(data, spec, result)
        state["runs"] = pd.concat([state["runs"], pd.DataFrame([row])], ignore_index=True)
        state["done"].add(_row_id(dataset, noise, seed, spec))
        _save_csv(state["runs"], SENSITIVITY_RUNS_PATH)
        _write_sensitivity_summary(state["runs"])
        print(
            f"sens {len(state['done'])}/{state['total']}: {dataset} noise={noise}"
            f" seed={seed} {spec['parameter']}={spec['value']:g}",
            flush=True,
        )


def run_sensitivity():
    os.makedirs(RESULT_DIR, exist_ok=True)
    state = _load_sensitivity_state()
    plan = []
    for noise in (False, True):
        for dataset in DATASETS:
            n_groups = len(_load_data(dataset, noise)[3])
            plan.append((dataset, noise, _sensitivity_specs(noise, n_groups)))
    state["total"] = sum(len(specs) for *_, specs in plan) * len(SENSITIVITY_SEEDS)
    for dataset, noise, specs in plan:
        for seed in SENSITIVITY_SEEDS:
            _sensitivity_seed(dataset, noise, seed, specs, state)
    _write_sensitivity_summary(state["runs"])
