import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _top_groups(column_scores, groups, k):
    group_ids = np.unique(groups)
    if not 1 <= k <= len(group_ids):
        raise ValueError("k must be between 1 and the number of groups")

    scores = np.full(len(group_ids), -np.inf)
    for i, group in enumerate(group_ids):
        scores[i] = np.max(column_scores[groups == group])
    selected = np.argsort(-scores, kind="stable")[:k]
    mask = np.zeros(len(group_ids), dtype=bool)
    mask[selected] = True
    return mask


def select_mi(X, y, groups, k, seed=0):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X.shape[1]:
        raise ValueError("groups must have one group id per feature column")
    scores = mutual_info_classif(X, y, random_state=seed)
    return _top_groups(scores, groups, k)


def select_rfe(X, y, groups, k, seed=0):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X.shape[1]:
        raise ValueError("groups must have one group id per feature column")
    X_scaled = StandardScaler().fit_transform(X)
    selector = RFE(
        LogisticRegression(max_iter=1000, random_state=seed),
        n_features_to_select=1,
    )
    selector.fit(X_scaled, y)
    scores = 1.0 / selector.ranking_
    return _top_groups(scores, groups, k)