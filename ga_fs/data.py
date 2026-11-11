import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(name):
    if name == "breast_cancer":
        dataset = load_breast_cancer()
        return (
            dataset.data.astype(float),
            dataset.target.astype(int),
            np.asarray(dataset.feature_names),
            np.arange(dataset.data.shape[1]),
            np.asarray(dataset.feature_names),
        )

    if name == "german_credit":
        dataset = fetch_openml(data_id=31, as_frame=True)
        columns = []
        feature_names = []
        groups = []
        group_names = np.asarray(dataset.data.columns)
        for group, name in enumerate(group_names):
            values = dataset.data[name]
            if (
                isinstance(values.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(values.dtype)
                or pd.api.types.is_string_dtype(values.dtype)
            ):
                encoded = pd.get_dummies(values, dtype=float)
                columns.append(encoded.to_numpy(dtype=float))
                feature_names.extend([name] * encoded.shape[1])
                groups.extend([group] * encoded.shape[1])
            else:
                columns.append(pd.to_numeric(values).to_numpy(dtype=float)[:, None])
                feature_names.append(name)
                groups.append(group)

        y = LabelEncoder().fit_transform(dataset.target)
        return (
            np.column_stack(columns),
            y,
            np.asarray(feature_names),
            np.asarray(groups, dtype=int),
            group_names,
        )

    raise ValueError(f"unknown dataset: {name}")


def expand_group_mask(group_mask, groups):
    group_mask = np.asarray(group_mask, dtype=bool)
    group_ids = np.unique(groups)
    if group_mask.ndim != 1 or len(group_mask) != len(group_ids):
        raise ValueError("group mask length must match the number of groups")
    return group_mask[np.searchsorted(group_ids, groups)]


def get_group_names(feature_names, groups):
    group_ids = np.unique(groups)
    names = []
    for group in group_ids:
        group_names = np.unique(np.asarray(feature_names)[groups == group])
        if len(group_names) != 1:
            raise ValueError("each group must map to exactly one attribute name")
        names.append(group_names[0])
    return np.asarray(names)


def get_splits(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
