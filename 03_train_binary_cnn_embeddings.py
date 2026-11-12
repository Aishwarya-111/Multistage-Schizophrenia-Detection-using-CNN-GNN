import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# STEP 3: Binary CNN Embedding Extractor
# Trains Healthy vs Schizophrenia, then saves embeddings
# for final 4-class GNN.
# ============================================================

METADATA_PATH = "Outputs/npy_dataset/npy_metadata.csv"
OUT_DIR = "Outputs/npy_dataset"

MODEL_PATH = os.path.join(OUT_DIR, "best_binary_cnn_for_embeddings.pth")
FEATURE_PATH = os.path.join(OUT_DIR, "cnn_embeddings_4class.csv")
RESULT_PATH = os.path.join(OUT_DIR, "binary_cnn_embedding_results.txt")

RANDOM_STATE = 42

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8

SLICES_PER_AXIS = 15
EMBED_DIM = 512


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(RANDOM_STATE)


def get_binary_label(stage):
    if stage == "Healthy":
        return "Healthy"
    return "Schizophrenia"


def get_slice_indices(length, n=SLICES_PER_AXIS):
    start = int(length * 0.25)
    end = int(length * 0.75)

    if end <= start:
        start = 0
        end = length - 1

    idxs = np.linspace(start, end, n, dtype=int)
    return np.clip(idxs, 0, length - 1).tolist()


def volume_to_slices(volume):
    volume = volume[0]
    slices = []

    for axis in range(3):
        idxs = get_slice_indices(volume.shape[axis], SLICES_PER_AXIS)

        for idx in idxs:
            if axis == 0:
                sl = volume[idx, :, :]
            elif axis == 1:
                sl = volume[:, idx, :]
            else:
                sl = volume[:, :, idx]

            sl = np.clip(sl, 0, 1).astype(np.float32)
            slices.append(sl)

    return slices


class MRISliceDataset(Dataset):
    def __init__(self, df, label_encoder, augment=False):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.augment = augment
        self.items = []

        print("Indexing MRI slices...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            volume = np.load(row["npy_path"]).astype(np.float32)
            slices = volume_to_slices(volume)

            for slice_idx in range(len(slices)):
                self.items.append({
                    "npy_path": row["npy_path"],
                    "participant_id": row["participant_id"],
                    "stage": row["stage"],
                    "binary_label": row["binary_label"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "slice_idx": slice_idx,
                })

        print(f"Total slices indexed: {len(self.items)}")

        self.train_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomRotation(5),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.eval_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        volume = np.load(item["npy_path"]).astype(np.float32)
        slices = volume_to_slices(volume)
        sl = slices[item["slice_idx"]]

        if self.augment:
            x = self.train_tf(sl)
        else:
            x = self.eval_tf(sl)

        x = x.repeat(3, 1, 1)

        y = self.le.transform([item["binary_label"]])[0]

        return x, torch.tensor(y).long(), item["participant_id"]


class BinaryMRIModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone.layer2.parameters():
            p.requires_grad = True

        for p in self.backbone.layer3.parameters():
            p.requires_grad = True

        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 2),
        )

    def forward(self, x, return_embedding=False):
        emb = self.backbone(x)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb

        return logits


def evaluate_subject_level(model, loader, device):
    model.eval()

    subject_probs = defaultdict(list)
    subject_true = {}

    with torch.no_grad():
        for x, y, pids in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for pid, prob, true_label in zip(pids, probs, y.numpy()):
                subject_probs[pid].append(prob)
                subject_true[pid] = true_label

    y_true = []
    y_pred = []

    for pid, probs in subject_probs.items():
        mean_prob = np.mean(np.vstack(probs), axis=0)
        pred = int(np.argmax(mean_prob))

        y_pred.append(pred)
        y_true.append(subject_true[pid])

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def extract_embeddings(model, df, le, device):
    dataset = MRISliceDataset(df, le, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()

    subject_embs = defaultdict(list)
    subject_probs = defaultdict(list)

    with torch.no_grad():
        for x, y, pids in tqdm(loader, desc="Extracting embeddings"):
            x = x.to(device)
            logits, emb = model(x, return_embedding=True)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            emb = emb.cpu().numpy()

            for pid, e, p in zip(pids, emb, probs):
                subject_embs[pid].append(e)
                subject_probs[pid].append(p)

    records = []

    for _, row in df.iterrows():
        pid = row["participant_id"]

        if pid not in subject_embs:
            continue

        mean_emb = np.mean(np.vstack(subject_embs[pid]), axis=0)
        mean_prob = np.mean(np.vstack(subject_probs[pid]), axis=0)

        healthy_idx = le.transform(["Healthy"])[0]
        schz_idx = le.transform(["Schizophrenia"])[0]

        pred_idx = int(np.argmax(mean_prob))
        pred_label = le.inverse_transform([pred_idx])[0]

        records.append({
            "participant_id": pid,
            "dataset": row["dataset"],
            "stage": row["stage"],
            "binary_label": row["binary_label"],
            "split": row["split"],
            "age": row.get("age"),
            "sex": row.get("sex"),
            "raw_score": row.get("raw_score"),
            "norm_score": row.get("norm_score"),
            "npy_path": row.get("npy_path"),
            "embedding": mean_emb.tolist(),
            "prob_healthy": float(mean_prob[healthy_idx]),
            "prob_schizophrenia": float(mean_prob[schz_idx]),
            "binary_pred": pred_label,
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("STEP 3 — Binary CNN Embedding Extractor")
    print("=" * 60)

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Run 02_preprocess_npy_3d.py first.")

    df = pd.read_csv(METADATA_PATH)
    df["binary_label"] = df["stage"].apply(get_binary_label)

    print(f"Subjects loaded: {len(df)}")

    print("\n4-class split counts:")
    print(df.groupby(["split", "stage"]).size())

    print("\nBinary label counts:")
    print(df["binary_label"].value_counts())

    le = LabelEncoder()
    le.fit(["Healthy", "Schizophrenia"])

    train_full = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_df, val_df = train_test_split(
        train_full,
        test_size=0.20,
        stratify=train_full["binary_label"],
        random_state=RANDOM_STATE,
    )

    print("\nTrain binary:")
    print(train_df["binary_label"].value_counts())

    print("\nValidation binary:")
    print(val_df["binary_label"].value_counts())

    print("\nTest binary:")
    print(test_df["binary_label"].value_counts())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_ds = MRISliceDataset(train_df, le, augment=True)
    val_ds = MRISliceDataset(val_df, le, augment=False)
    test_ds = MRISliceDataset(test_df, le, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BinaryMRIModel().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_val = 0.0
    best_epoch = 0
    no_improve = 0

    print("\nTraining binary CNN for MRI embeddings...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        slice_true = []
        slice_pred = []

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            preds = logits.argmax(1)

            slice_true.extend(y.cpu().numpy())
            slice_pred.extend(preds.cpu().numpy())

        scheduler.step()

        train_acc = accuracy_score(slice_true, slice_pred)
        val_acc, _, _ = evaluate_subject_level(model, val_loader, device)

        marker = ""

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ◀ BEST"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {running_loss / len(train_ds):.4f} | "
            f"Train slice: {train_acc * 100:.1f}% | "
            f"Val subject: {val_acc * 100:.1f}%{marker}"
        )

        if no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest validation binary accuracy: {best_val * 100:.1f}% at epoch {best_epoch}")

    print("\nLoading best binary CNN...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_acc, y_true, y_pred = evaluate_subject_level(model, test_loader, device)

    print(f"\nFinal TEST binary accuracy: {test_acc * 100:.1f}%")

    report = classification_report(
        y_true,
        y_pred,
        target_names=le.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best Validation Binary Accuracy: {best_val * 100:.2f}%\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Final Test Binary Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nSaved binary CNN results → {RESULT_PATH}")

    print("\nExtracting CNN embeddings for all 4-class subjects...")
    features_df = extract_embeddings(model, df, le, device)
    features_df.to_csv(FEATURE_PATH, index=False)

    print(f"Saved embeddings → {FEATURE_PATH}")
    print(f"Subjects saved: {len(features_df)}")
    print(f"Embedding dimension: {EMBED_DIM}")


if __name__ == "__main__":
    main()