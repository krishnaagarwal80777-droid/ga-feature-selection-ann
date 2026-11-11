import numpy as np

from ga_fs.ann import cv_score


def run_ga(
    X,
    y,
    seed,
    pop_size=30,
    n_gen=30,
    alpha=0.01,
    metric="balanced_accuracy",
    folds=3,
):
    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    if n_gen < 1:
        raise ValueError("n_gen must be at least 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if metric not in ("balanced_accuracy", "accuracy"):
        raise ValueError(f"unsupported metric: {metric}")

    rng = np.random.default_rng(seed)
    n_feats = X.shape[1]
    pop = rng.integers(0, 2, size=(pop_size, n_feats), dtype=np.int8)
    for i in np.flatnonzero(~pop.any(axis=1)):
        pop[i, rng.integers(0, n_feats)] = 1
    cache = {}
    history = []

    def fitness(mask):
        key = mask.tobytes()
        if key not in cache:
            score = cv_score(
                X, y, mask, seed, folds=folds, metric=metric, fast=True
            )
            cache[key] = score - alpha * (mask.sum() / n_feats)
        return cache[key]

    def tournament():
        ids = rng.integers(0, pop_size, size=3)
        return pop[ids[np.argmax([fitness(pop[i]) for i in ids])]].copy()

    best_mask = None
    best_score = -np.inf
    for gen in range(n_gen):
        fit = np.asarray([fitness(mask) for mask in pop])
        best_idx = int(np.argmax(fit))
        if fit[best_idx] > best_score:
            best_score = fit[best_idx]
            best_mask = pop[best_idx].copy()
        history.append(float(best_score))

        if gen == n_gen - 1:
            break

        new_pop = [pop[best_idx].copy()]
        while len(new_pop) < pop_size:
            parent_a = tournament()
            parent_b = tournament()
            swap = rng.random(n_feats) < 0.5
            children = (
                np.where(swap, parent_a, parent_b),
                np.where(swap, parent_b, parent_a),
            )
            for child in children:
                mutate = rng.random(n_feats) < (1.0 / n_feats)
                child[mutate] = 1 - child[mutate]
                if not child.any():
                    child[rng.integers(0, n_feats)] = 1
                new_pop.append(child.astype(np.int8, copy=False))
                if len(new_pop) == pop_size:
                    break
        pop = np.asarray(new_pop)

    return best_mask.astype(bool), history