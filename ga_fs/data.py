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
        )

    if name == "german_credit":
        dataset = fetch_openml(data_id=31, as_frame=True)
        X = pd.get_dummies(dataset.data, dtype=float)
        y = LabelEncoder().fit_transform(dataset.target)
        return X.to_numpy(dtype=float), y, X.columns.to_numpy()

    raise ValueError(f"unknown dataset: {name}")


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
