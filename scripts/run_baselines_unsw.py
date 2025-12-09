#!/usr/bin/env python
"""
Treina baselines em UNSW-NB15:
- Isolation Forest (unsupervised, normal-only)
- RNN (GRU) supervisionada
- Transformer NIDS supervisionado

Salva modelos e métricas em artifacts/benchmark/.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --------------------------
# CONFIGURAÇÕES BÁSICAS
# --------------------------

DEFAULT_TRAIN_CSV = "data/raw/unsw-nb15/csv/UNSW_NB15_training-set.csv"
DEFAULT_TEST_CSV = "data/raw/unsw-nb15/csv/UNSW_NB15_testing-set.csv"
OUT_DIR = Path("artifacts/benchmark")


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self):
        return asdict(self)


# --------------------------
# CARREGAMENTO E FEATURES
# --------------------------


def load_unsw_data(
    train_csv: str = DEFAULT_TRAIN_CSV,
    test_csv: str = DEFAULT_TEST_CSV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Carrega UNSW-NB15 (treino/teste) e retorna X_train, y_train, X_test, y_test.

    Assumimos:
    - coluna 'label' já binária (0=normal, 1=ataque)
    - colunas de id/label/attack_cat são removidas do vetor de features
    """
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Ajuste aqui se os nomes no CSV forem diferentes
    label_col = "label"
    drop_cols = ["id", "attack_cat", label_col]
    drop_cols = [c for c in drop_cols if c in df_train.columns]

    y_train = df_train[label_col].astype(int).values
    y_test = df_test[label_col].astype(int).values

    X_train = df_train.drop(columns=drop_cols)
    X_test = df_test.drop(columns=drop_cols)

    # Apenas features numéricas (se ainda houver alguma categórica, encode antes)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


# --------------------------
# MÉTRICAS AUXILIARES
# --------------------------


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return Metrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
    )


# --------------------------
# 1) ISOLATION FOREST
# --------------------------


def train_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Metrics:
    """
    Treina Isolation Forest apenas em flows normais (y=0)
    e avalia em X_test, y_test.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_normal = X_train[y_train == 0]

    if_clf = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    if_clf.fit(X_train_normal)

    # sklearn: 1 = normal, -1 = anomalia
    pred_raw = if_clf.predict(X_test)
    y_pred = np.where(pred_raw == -1, 1, 0).astype(int)

    metrics = compute_binary_metrics(y_test, y_pred)

    # salvar modelo e métricas
    dump(if_clf, out_dir / "iforest_model.joblib")
    pd.DataFrame([metrics.to_dict()]).to_csv(out_dir / "metrics_isolation_forest.csv", index=False)

    print("[IFOREST]", metrics)
    return metrics


# --------------------------
# DATASET PARA RNN/TRANSFORMER
# --------------------------


class FlowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# --------------------------
# 2) RNN (GRU) BASELINE
# --------------------------


class SimpleGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        # out: (B, 1, H) -> (B, H)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits


def train_supervised_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch}/{max_epochs}] train", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # validação rápida
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(1, total)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

    return model


def run_rnn_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Metrics:
    """
    Treina um baseline RNN (GRU) supervisionado per-flow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split simples treino/val (ex.: 90/10)
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    ds_train = FlowsDataset(X_train[train_idx], y_train[train_idx])
    ds_val = FlowsDataset(X_train[val_idx], y_train[val_idx])
    ds_test = FlowsDataset(X_test, y_test)

    train_loader = DataLoader(ds_train, batch_size=256, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=512, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=512, shuffle=False)

    model = SimpleGRUClassifier(input_dim=X_train.shape[1], hidden_dim=128, num_layers=1)
    model = train_supervised_model(model, train_loader, val_loader, device, max_epochs=10, lr=1e-3, weight_decay=1e-5)

    # avaliação final
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_binary_metrics(y_true, y_pred)

    # salvar modelo e métricas
    torch.save(model.state_dict(), out_dir / "rnn_model.pt")
    pd.DataFrame([metrics.to_dict()]).to_csv(out_dir / "metrics_rnn.csv", index=False)

    print("[RNN]", metrics)
    return metrics


# --------------------------
# 3) TRANSFORMER NIDS
# --------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        enc = self.encoder(x)  # (B, 1, D)
        out = enc[:, -1, :]
        logits = self.fc(out)
        return logits


def run_transformer_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Metrics:
    """
    Treina um baseline Transformer supervisionado per-flow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split simples treino/val (ex.: 90/10)
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    ds_train = FlowsDataset(X_train[train_idx], y_train[train_idx])
    ds_val = FlowsDataset(X_train[val_idx], y_train[val_idx])
    ds_test = FlowsDataset(X_test, y_test)

    train_loader = DataLoader(ds_train, batch_size=256, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=512, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=512, shuffle=False)

    model = SimpleTransformerClassifier(input_dim=X_train.shape[1], d_model=128, nhead=4, num_layers=2)
    model = train_supervised_model(model, train_loader, val_loader, device, max_epochs=10, lr=1e-3, weight_decay=1e-5)

    # avaliação final
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_binary_metrics(y_true, y_pred)

    # salvar modelo e métricas
    torch.save(model.state_dict(), out_dir / "transformer_model.pt")
    pd.DataFrame([metrics.to_dict()]).to_csv(out_dir / "metrics_transformer.csv", index=False)

    print("[TRANSFORMER]", metrics)
    return metrics


# --------------------------
# MAIN
# --------------------------


def main(
    train_csv: str = DEFAULT_TRAIN_CSV,
    test_csv: str = DEFAULT_TEST_CSV,
    out_dir: Path = OUT_DIR,
):
    print(f"[DATA] train_csv={train_csv}")
    print(f"[DATA] test_csv={test_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, scaler = load_unsw_data(train_csv, test_csv)
    # salvar scaler para reutilizar em análises futuras
    dump(scaler, out_dir / "scaler.joblib")

    # 1) Isolation Forest
    train_isolation_forest(X_train, y_train, X_test, y_test, out_dir)

    # 2) RNN baseline
    run_rnn_baseline(X_train, y_train, X_test, y_test, out_dir)

    # 3) Transformer baseline
    run_transformer_baseline(X_train, y_train, X_test, y_test, out_dir)


if __name__ == "__main__":
    main()
