from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"

# Числовые признаки — стандартизируем
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Бинарные признаки — уже 0/1 после маппинга, передаём как есть
BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Категориальные признаки — One-Hot Encoding
CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

_BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}


def load_and_clean(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Убираем пробелы в строковых колонках
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges: 11 пустых значений у новых клиентов (tenure=0) → 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Удаляем ID — не несёт информации
    df = df.drop(columns=["customerID"])

    # Кодируем таргет и бинарные признаки
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(_BINARY_MAP)

    return df


def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_loaders(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    test_size: float = 0.2,
    val_size: float = 0.2,
    batch_size: int = 64,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Возвращает train/val/test DataLoader-ы и размер входного вектора."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values.astype(np.float32)

    # Сначала отщепляем тест
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Из оставшегося отщепляем val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )

    # Фитим препроцессор только на трейне
    X_train_arr = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_arr = preprocessor.transform(X_val).astype(np.float32)
    X_test_arr = preprocessor.transform(X_test).astype(np.float32)

    input_dim = X_train_arr.shape[1]

    def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train_arr, y_train, shuffle=True)
    val_loader = _make_loader(X_val_arr, y_val, shuffle=False)
    test_loader = _make_loader(X_test_arr, y_test, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(path: Path) -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)
