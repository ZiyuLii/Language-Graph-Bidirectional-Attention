import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, Dataset, random_split

from lgba.model.multimodal_regression_model import MultiModalRegressionModel


DATA_PATH = "data/regression.csv"
SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "pIC50"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4
PATIENCE = 5

SPE_VOCAB_PATH = "models/vocab-spe.pkl"
AWD_VOCAB_PATH = "models/vocab-awd.pkl"
AWD_MODEL_PATH = "models/smiles_encoder.pth"
DMPNN_MODEL_PATH = "models/best_encoder.pth"

SAVE_DIR = "regression_checkpoints"
DEVICE = torch.device("cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


class SmilesRegressionDataset(Dataset):
    def __init__(self, csv_path, smiles_col="SMILES", target_col="pIC50"):
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].astype(str).tolist()
        self.targets = df[target_col].astype(float).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], torch.tensor(self.targets[idx], dtype=torch.float32)


def compute_metrics(preds, targets):
    preds = preds.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def train():
    print("Loading dataset...")
    dataset = SmilesRegressionDataset(DATA_PATH, SMILES_COLUMN, TARGET_COLUMN)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print("Building model...")
    model = MultiModalRegressionModel(
        spe_vocab_path=SPE_VOCAB_PATH,
        awd_vocab_path=AWD_VOCAB_PATH,
        awd_model_path=AWD_MODEL_PATH,
        dmpnn_model_path=DMPNN_MODEL_PATH,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve_epochs = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for smiles, target in train_loader:
            smiles = list(smiles)
            target = target.to(DEVICE).unsqueeze(1)
            pred = model(smiles)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for smiles, target in val_loader:
                smiles = list(smiles)
                target = target.to(DEVICE).unsqueeze(1)
                pred = model(smiles)
                val_preds.append(pred)
                val_targets.append(target)

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_loss = loss_fn(val_preds, val_targets).item()
        mae, rmse, r2 = compute_metrics(val_preds, val_targets)

        train_loss = sum(train_losses) / len(train_losses)
        print(
            f"[Epoch {epoch:02d}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print("Early stopping.")
                break

    print("Training complete.")


if __name__ == "__main__":
    train()

