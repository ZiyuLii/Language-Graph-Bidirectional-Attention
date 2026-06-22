import pandas as pd
import torch

from lgba.model.multimodal_regression_model import MultiModalRegressionModel
from lgba.model.train_multimodal_classification import MultiModalClassifier


TASK_TYPE = "regression"
# TASK_TYPE = "classification"

INPUT_CSV = "data/input.csv"
SMILES_COLUMN = "SMILES"
OUTPUT_CSV = "results/predictions.csv"

SPE_VOCAB_PATH = "models/vocab-spe.pkl"
AWD_VOCAB_PATH = "models/vocab-awd.pkl"
AWD_MODEL_PATH = "models/smiles_encoder.pth"
DMPNN_MODEL_PATH = "models/best_encoder.pth"

CLASSIFIER_CKPT = "classification_checkpoints/best_classifier.pth"
REGRESSOR_CKPT = "regression_checkpoints/Lipophilicity_best_model.pth"

BATCH_SIZE = 64
DEVICE = torch.device("cpu")


def load_model(task_type):
    if task_type == "classification":
        model = MultiModalClassifier(
            spe_vocab_path=SPE_VOCAB_PATH,
            awd_vocab_path=AWD_VOCAB_PATH,
            awd_model_path=AWD_MODEL_PATH,
            dmpnn_model_path=DMPNN_MODEL_PATH,
            device=DEVICE,
        ).to(DEVICE)
        ckpt = CLASSIFIER_CKPT
    elif task_type == "regression":
        model = MultiModalRegressionModel(
            spe_vocab_path=SPE_VOCAB_PATH,
            awd_vocab_path=AWD_VOCAB_PATH,
            awd_model_path=AWD_MODEL_PATH,
            dmpnn_model_path=DMPNN_MODEL_PATH,
            device=DEVICE,
        ).to(DEVICE)
        ckpt = REGRESSOR_CKPT
    else:
        raise ValueError("TASK_TYPE must be either 'classification' or 'regression'")

    state_dict = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict(model, smiles_list, task_type):
    preds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), BATCH_SIZE):
            batch = smiles_list[i:i + BATCH_SIZE]
            output = model(batch).squeeze(1)
            if task_type == "classification":
                prob = torch.sigmoid(output).cpu().numpy()
                pred = (prob >= 0.5).astype(int)
                preds.extend(pred.tolist())
            else:
                preds.extend(output.cpu().numpy().tolist())
    return preds


def main():
    df = pd.read_csv(INPUT_CSV)
    if SMILES_COLUMN not in df.columns:
        raise KeyError(f"Input CSV does not contain column: {SMILES_COLUMN}")

    model = load_model(TASK_TYPE)
    smiles_list = df[SMILES_COLUMN].astype(str).tolist()
    predictions = predict(model, smiles_list, TASK_TYPE)

    if TASK_TYPE == "classification":
        df["prediction"] = predictions
    else:
        df["predicted_value"] = predictions

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Prediction complete. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

