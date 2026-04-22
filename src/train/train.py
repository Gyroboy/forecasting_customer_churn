from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Добавляем src в путь, чтобы импортировать наши модули
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.preprocessing import (
    build_preprocessor,
    load_and_clean,
    prepare_loaders,
    save_preprocessor,
)
from models.mlp import ChurnMLP

MODELS_DIR = ROOT / "artifacts" / "models"

# ── Гиперпараметры ────────────────────────────────────────────────────────────
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
DROPOUT = 0.3
PATIENCE = 15        # early stopping: сколько эпох без улучшения val AUC терпим
WEIGHT_DECAY = 1e-4  # L2-регуляризация в Adam


# ── Утилиты ───────────────────────────────────────────────────────────────────

def compute_pos_weight(train_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """pos_weight для BCEWithLogitsLoss — компенсирует дисбаланс классов."""
    labels = np.concatenate([y.numpy() for _, y in train_loader])
    neg, pos = (labels == 0).sum(), (labels == 1).sum()
    return torch.tensor([neg / pos], dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Возвращает (loss, ROC-AUC, F1)."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)

    return avg_loss, auc, f1


# ── Основной пайплайн ─────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Данные
    df = load_and_clean()
    preprocessor = build_preprocessor()
    train_loader, val_loader, test_loader, input_dim = prepare_loaders(
        df, preprocessor, batch_size=BATCH_SIZE
    )
    print(f"Input dim after preprocessing: {input_dim}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}\n")

    # Компенсация дисбаланса классов через pos_weight
    pos_weight = compute_pos_weight(train_loader).to(device)
    print(f"pos_weight (neg/pos ratio): {pos_weight.item():.3f}\n")

    # Модель, оптимизатор, loss, scheduler
    model = ChurnMLP(input_dim=input_dim, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    # Цикл обучения
    best_val_auc = 0.0
    best_state: dict | None = None
    no_improve = 0

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val AUC':>7} | {'Val F1':>6} | {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_auc:>7.4f} | {val_f1:>6.4f} | {current_lr:>8.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping на эпохе {epoch}. Лучший val AUC: {best_val_auc:.4f}")
                break

    # Загружаем лучшие веса
    model.load_state_dict(best_state)

    # Оценка на тесте
    _, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"Test ROC-AUC : {test_auc:.4f}")
    print(f"Test F1      : {test_f1:.4f}")

    # Полный classification report
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=["No Churn", "Churn"]))

    # Сохранение модели и препроцессора
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "churn_mlp.pt"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "dropout": DROPOUT,
            "val_auc": best_val_auc,
            "test_auc": test_auc,
        },
        model_path,
    )
    save_preprocessor(preprocessor, preprocessor_path)

    print(f"\nМодель сохранена:       {model_path}")
    print(f"Препроцессор сохранён:  {preprocessor_path}")


if __name__ == "__main__":
    main()
