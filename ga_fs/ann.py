import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier


FINAL_ANN_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "early_stopping": False,
    "max_iter": 1000,
    "alpha": 1e-4,
}


def build_ann(seed, fast=False):
    if fast:
        return MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            early_stopping=True,
            max_iter=200,
            random_state=seed,
        )

    return MLPClassifier(**FINAL_ANN_CONFIG, random_state=seed)


def cv_score(X, y, mask, seed, folds=3):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0

    model = build_ann(seed, fast=True)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X[:, mask], y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def evaluate(X_tr, y_tr, X_te, y_te, mask, seed, return_diagnostics=False):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("mask must select at least one feature")

    model = build_ann(seed)
    model.fit(X_tr[:, mask], y_tr)
    y_pred = model.predict(X_te[:, mask])
    y_prob = model.predict_proba(X_te[:, mask])
    positive_idx = np.flatnonzero(model.classes_ == 1)[0]
    classes, counts = np.unique(y_tr, return_counts=True)
    minority_class = classes[np.argmin(counts)]

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "minority_recall": recall_score(
            y_te, y_pred, labels=[minority_class], average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_prob[:, positive_idx]),
    }
    if return_diagnostics:
        metrics["n_iter"] = model.n_iter_
        metrics["converged"] = model.n_iter_ < model.max_iter
    return metrics