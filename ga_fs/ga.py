import numpy as np

from ga_fs.ann import cv_score
from ga_fs.data import expand_group_mask


def tournament(pop, fitness_scores, rng):
    ids = rng.integers(0, len(pop), size=3)
    return pop[ids[np.argmax(fitness_scores[ids])]].copy()


def crossover(parent_a, parent_b, rng, prob=0.8):
    if rng.random() >= prob:
        return parent_a.copy(), parent_b.copy()

    swap = rng.random(len(parent_a)) < 0.5
    child_a = np.where(swap, parent_a, parent_b)
    child_b = np.where(swap, parent_b, parent_a)
    return child_a, child_b


def mutate(child, rng, rate=None):
    child = child.copy()
    if rate is None:
        rate = 1.0 / len(child)
    flips = rng.random(len(child)) < rate
    child[flips] = 1 - child[flips]
    if not child.any():
        child[rng.integers(0, len(child))] = 1
    return child


def run_ga(
    X_tr,
    y_tr,
    groups,
    seed,
    pop_size=30,
    n_gen=40,
    alpha=0.01,
    metric="balanced_accuracy",
    mutation_rate=None,
):
    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    if n_gen < 1:
        raise ValueError("n_gen must be at least 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if mutation_rate is not None and not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if metric not in ("balanced_accuracy", "accuracy"):
        raise ValueError(f"unsupported metric: {metric}")

    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X_tr.shape[1]:
        raise ValueError("groups must have one group id per feature column")

    rng = np.random.default_rng(seed)
    group_ids = np.unique(groups)
    n_groups = len(group_ids)
    if n_groups == 0:
        raise ValueError("groups must not be empty")
    if mutation_rate is None:
        mutation_rate = 1.0 / n_groups

    pop = rng.integers(0, 2, size=(pop_size, n_groups), dtype=np.int8)
    for i in np.flatnonzero(~pop.any(axis=1)):
        pop[i, rng.integers(0, n_groups)] = 1

    cache = {}
    history = []

    best_mask = None
    best_score = -np.inf
    best_metric = 0.0
    no_improve = 0
    for _ in range(n_gen):
        fit = np.empty(pop_size)
        metric_scores = np.empty(pop_size)
        n_new_evals = 0
        n_cache_hits = 0
        for i, mask in enumerate(pop):
            key = mask.tobytes()
            if key in cache:
                n_cache_hits += 1
                metric_scores[i] = cache[key][1]
                fit[i] = cache[key][0]
                continue

            column_mask = expand_group_mask(mask, groups)
            score = cv_score(
                X_tr, y_tr, column_mask, seed, metric=metric, fast=True
            )
            value = score - alpha * (mask.sum() / n_groups)
            cache[key] = (value, score)
            n_new_evals += 1
            metric_scores[i] = score
            fit[i] = value

        order = np.argsort(fit)[::-1]
        best_idx = order[0]
        if fit[best_idx] > best_score:
            best_score = float(fit[best_idx])
            best_mask = pop[best_idx].copy()
            best_metric = float(metric_scores[best_idx])
            no_improve = 0
        else:
            no_improve += 1

        stop = no_improve >= 10 or len(history) + 1 == n_gen
        if not stop:
            new_pop = [pop[i].copy() for i in order[:2]]
            while len(new_pop) < pop_size:
                parent_a = tournament(pop, fit, rng)
                parent_b = tournament(pop, fit, rng)
                children = crossover(parent_a, parent_b, rng)
                for child in children:
                    new_pop.append(
                        mutate(child, rng, mutation_rate).astype(
                            np.int8, copy=False
                        )
                    )
                    if len(new_pop) == pop_size:
                        break
            pop = np.asarray(new_pop)

        history.append(
            {
                "best_fitness": best_score,
                "mean_fitness": float(np.mean(fit)),
                "best_metric_score": best_metric,
                "n_groups_selected": int(best_mask.sum()),
                "n_new_evals": n_new_evals,
                "n_cache_hits": n_cache_hits,
            }
        )
        if stop:
            break

    return best_mask.astype(bool), history