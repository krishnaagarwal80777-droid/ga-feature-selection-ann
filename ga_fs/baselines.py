import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def random_mask(n_groups, k, seed):
    if not 1 <= k <= n_groups:
        raise ValueError("k must be between 1 and n_groups")

    rng = np.random.default_rng(seed)
    mask = np.zeros(n_groups, dtype=bool)
    mask[rng.choice(n_groups, size=k, replace=False)] = True
    return mask


def _top_groups(column_scores, groups, k):
    group_ids = np.unique(groups)
    if not 1 <= k <= len(group_ids):
        raise ValueError("k must be between 1 and the number of groups")

    group_scores = np.asarray(
        [np.max(column_scores[groups == group]) for group in group_ids]
    )
    selected = np.argsort(-group_scores, kind="stable")[:k]
    mask = np.zeros(len(group_ids), dtype=bool)
    mask[selected] = True
    return mask


def mutual_info_mask(X_tr, y_tr, groups, k, seed=0):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X_tr.shape[1]:
        raise ValueError("groups must have one group id per feature column")

    scores = mutual_info_classif(X_tr, y_tr, random_state=seed)
    return _top_groups(scores, groups, k)


def rfe_mask(X_tr, y_tr, groups, k, seed=0):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X_tr.shape[1]:
        raise ValueError("groups must have one group id per feature column")

    X_scaled = StandardScaler().fit_transform(X_tr)
    selector = RFE(
        LogisticRegression(max_iter=1000, random_state=seed),
        n_features_to_select=1,
    )
    selector.fit(X_scaled, y_tr)
    group_ids = np.unique(groups)
    group_ranks = np.asarray(
        [np.min(selector.ranking_[groups == group]) for group in group_ids]
    )
    selected = np.argsort(group_ranks, kind="stable")[:k]
    mask = np.zeros(len(group_ids), dtype=bool)
    mask[selected] = True
    return mask


def pca_transform(X_tr, X_te, k, seed=0):
    if not 1 <= k <= min(X_tr.shape[0], X_tr.shape[1]):
        raise ValueError("k must be between 1 and the train data dimensions")

    pca = PCA(n_components=k, random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    return X_tr_pca, X_te_pca
