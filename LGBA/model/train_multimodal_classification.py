import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from lgba.model.multimodal_regression_model import MultiModalRegressionModel


RDLogger.DisableLog("rdApp.*")

DATA_PATH = "data/classification.csv"
SMILES_COLUMN = "SMILES"
LABEL_COLUMN = "OUTCOME"

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 2e-4
PATIENCE = 5

SPE_VOCAB_PATH = "models/vocab-spe.pkl"
AWD_VOCAB_PATH = "models/vocab-awd.pkl"
AWD_MODEL_PATH = "models/smiles_encoder.pth"
DMPNN_MODEL_PATH = "models/best_encoder.pth"

SAVE_DIR = "classification_checkpoints"
DEVICE = torch.device("cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


class SmilesClassificationDataset(Dataset):
    def __init__(self, csv_path, smiles_col="SMILES", label_col="Label"):
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def get_pos_weight(subset):
    labels = [
        subset.dataset.labels[i]
        if isinstance(subset.dataset.labels[i], (int, float))
        else subset.dataset.labels[i].item()
        for i in subset.indices
    ]

    counter = Counter(labels)
    neg, pos = counter.get(0, 0), counter.get(1, 0)

    if pos == 0:
        print("Warning: no positive samples in training set. pos_weight set to 1.0")
        pos_weight = 1.0
    else:
        pos_weight = neg / pos

    return torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)


class MultiModalClassifier(MultiModalRegressionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(kwargs.get("fusion_dim", 256), kwargs.get("mlp_hidden_dim", 128)),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(kwargs.get("mlp_hidden_dim", 128), 1),
        )

    def forward(self, smiles_list):
        return super().forward(smiles_list)


def compute_metrics(y_true, y_pred):
    y_pred_cls = (y_pred > 0.5).astype(int)
    report = classification_report(y_true, y_pred_cls, digits=4, output_dict=False)
    print(report)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_cls))
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("Warning: ROC-AUC could not be computed for this validation set.")


def train():
    print("Loading dataset...")
    dataset = SmilesClassificationDataset(DATA_PATH, SMILES_COLUMN, LABEL_COLUMN)

    labels = [label for _, label in dataset]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(split.split(X=labels, y=labels))
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    pos_weight = get_pos_weight(train_set)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print("Building model...")
    model = MultiModalClassifier(
        spe_vocab_path=SPE_VOCAB_PATH,
        awd_vocab_path=AWD_VOCAB_PATH,
        awd_model_path=AWD_MODEL_PATH,
        dmpnn_model_path=DMPNN_MODEL_PATH,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for smiles, label in train_loader:
            smiles = list(smiles)
            label = label.to(DEVICE).unsqueeze(1)
            logits = model(smiles)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for smiles, label in val_loader:
                smiles = list(smiles)
                label = label.to(DEVICE).unsqueeze(1)
                logits = model(smiles)
                val_logits.append(torch.sigmoid(logits))
                val_labels.append(label)

        val_preds = torch.cat(val_logits, dim=0).flatten().cpu().numpy()
        val_targets = torch.cat(val_labels, dim=0).flatten().cpu().numpy()
        val_loss = loss_fn(torch.tensor(val_preds), torch.tensor(val_targets)).item()

        train_loss = sum(train_losses) / len(train_losses)
        print(f"[Epoch {epoch:02d}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}")
        compute_metrics(val_targets, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_classifier.pth"))
            print("Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print("Early stopping.")
                break

    print("Training complete.")


if __name__ == "__main__":
    train()

