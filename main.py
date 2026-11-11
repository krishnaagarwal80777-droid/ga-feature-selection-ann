import numpy as np

from ga_fs.ann import evaluate
from ga_fs.data import get_splits, load_dataset


def main():
    metric_names = (
        "accuracy",
        "precision",
        "macro_f1",
        "minority_recall",
        "balanced_accuracy",
        "roc_auc",
    )
    results = {}

    for name in ("breast_cancer", "german_credit"):
        X, y, _ = load_dataset(name)
        scores = {metric: [] for metric in metric_names}
        mask = np.ones(X.shape[1], dtype=bool)

        for seed in range(5):
            X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)
            metrics = evaluate(
                X_tr, y_tr, X_te, y_te, mask, seed, return_diagnostics=True
            )
            print(
                f"{name} seed={seed}: n_iter={metrics['n_iter']}, "
                f"converged={metrics['converged']}"
            )
            for metric in metric_names:
                scores[metric].append(metrics[metric])

        results[name] = scores

    print(
        "dataset         accuracy          precision         macro_f1          "
        "minority_recall   balanced_accuracy  roc_auc"
    )
    for name, scores in results.items():
        summary = "  ".join(
            f"{np.mean(scores[key]):.4f} +/- {np.std(scores[key]):.4f}"
            for key in metric_names
        )
        print(f"{name:<15} {summary}")


if __name__ == "__main__":
    main()
