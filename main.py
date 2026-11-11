import numpy as np

from ga_fs.ann import evaluate
from ga_fs.data import get_splits, load_dataset


def main():
    for name in ("breast_cancer", "german_credit"):
        X, y, _ = load_dataset(name)
        accuracy = []
        f1 = []
        mask = np.ones(X.shape[1], dtype=bool)

        for seed in range(5):
            X_tr, X_te, y_tr, y_te = get_splits(X, y, seed)
            metrics = evaluate(X_tr, y_tr, X_te, y_te, mask, seed)
            accuracy.append(metrics["accuracy"])
            f1.append(metrics["f1"])

        print(
            f"{name}: accuracy={np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}, "
            f"f1={np.mean(f1):.4f} ± {np.std(f1):.4f}"
        )


if __name__ == "__main__":
    main()
