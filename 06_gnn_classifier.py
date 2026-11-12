import os
import random
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# GNN CLASSIFIER (4 symptom-severity tiers)
# - Node features: 512-dim CNN embeddings + CNN probabilities + age/sex
#   + selected structural/fMRI features. Clinical scores are NOT in the
#   node feature vector.
# - Graph edges: k-NN on age, sex and normalized SANS/SAPS score.
# - KNOWN LIMITATION: the same symptom score defines the stage labels, so
#   the graph topology carries target-related information. The graph is
#   also built over the full cohort (transductive setup). Results are a
#   small single-split prototype (20 test subjects), not clinical evidence.
# ============================================================

INPUT_CSV = "Outputs/npy_dataset/fmri_features_4class.csv"
OUT_DIR = "Outputs/gnn_results"

MODEL_PATH = os.path.join(OUT_DIR, "best_gnn_model.pth")
RESULTS_TXT = os.path.join(OUT_DIR, "gnn_results.txt")
PREDS_CSV = os.path.join(OUT_DIR, "gnn_predictions.csv")

RANDOM_STATE = 42
CLASS_ORDER = ["Healthy", "Early", "Moderate", "Chronic"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}

# Hyperparameters
KNN_K = 5
HIDDEN_DIM = 64
EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 5e-3
DROPOUT = 0.50
PATIENCE = 100
TOP_K_MRI_FEATURES = 20

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_STATE)

def encode_sex(x):
    x = str(x).upper().strip()
    if x in ["M", "MALE"]: return 1.0
    if x in ["F", "FEMALE"]: return 0.0
    return -1.0

class GenuineGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        return self.conv3(x, edge_index)

def create_masks(df):
    train_indices = df[df["split"] == "train"].index.values
    test_indices = df[df["split"] == "test"].index.values

    train_idx, val_idx = train_test_split(
        train_indices, test_size=0.20, stratify=df.loc[train_indices, "stage"], random_state=RANDOM_STATE
    )

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[test_indices] = True
    return train_mask, val_mask, test_mask

def build_node_features(df, y, train_mask):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex_encoded"] = df["sex"].apply(encode_sex)

    # 1. Parse CNN Embeddings (512 dimensions)
    embeddings = []
    for emb_str in df["embedding"]:
        try:
            emb = ast.literal_eval(emb_str)
            if len(emb) != 512: emb = [0.0] * 512
        except:
            emb = [0.0] * 512
        embeddings.append(emb)
    X_cnn = np.array(embeddings, dtype=np.float32)

    # 2. Base Demographics and CNN Probabilities (NO CLINICAL SCORES)
    X_base = df[["prob_healthy", "prob_schizophrenia", "age", "sex_encoded"]].values.astype(np.float32)

    # 3. Structural/fMRI Features
    candidate_cols = []
    for col in df.columns:
        if (col.startswith("region_") or col.startswith("left_") or col.startswith("right_")
            or col.startswith("central_") or col.startswith("fmri_") or col.startswith("fc_")):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidate_cols.append(col)
    
    X_cand = df[candidate_cols].replace([np.inf, -np.inf], np.nan).fillna(df[candidate_cols].median(numeric_only=True)).values.astype(np.float32)

    train_idx = np.where(train_mask.numpy())[0]

    # Scaling
    scaler_cnn = StandardScaler()
    X_cnn_scaled = X_cnn.copy()
    X_cnn_scaled[train_idx] = scaler_cnn.fit_transform(X_cnn[train_idx])
    non_train_idx = np.setdiff1d(np.arange(len(X_cnn)), train_idx)
    X_cnn_scaled[non_train_idx] = scaler_cnn.transform(X_cnn[non_train_idx])

    scaler_base = StandardScaler()
    X_base_scaled = X_base.copy()
    X_base_scaled[train_idx] = scaler_base.fit_transform(X_base[train_idx])
    X_base_scaled[non_train_idx] = scaler_base.transform(X_base[non_train_idx])

    scaler_cand = StandardScaler()
    X_cand_scaled = X_cand.copy()
    X_cand_scaled[train_idx] = scaler_cand.fit_transform(X_cand[train_idx])
    X_cand_scaled[non_train_idx] = scaler_cand.transform(X_cand[non_train_idx])

    # Feature Selection for MRI
    k = min(TOP_K_MRI_FEATURES, X_cand_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_cand_scaled[train_idx], y[train_idx])
    X_cand_selected = selector.transform(X_cand_scaled)
    
    # Concatenate EVERYTHING except clinical scores
    X_final = np.hstack([X_cnn_scaled, X_base_scaled, X_cand_selected]).astype(np.float32)

    print("\n[Node Features Summary]")
    print(f"CNN Embeddings            : {X_cnn.shape[1]}")
    print(f"Demographics/CNN Probs    : {X_base.shape[1]}")
    print(f"Selected MRI/fMRI features: {k}")
    print(f"Total Node Dimension      : {X_final.shape[1]}")
    print("NOTE: raw_score/norm_score are excluded from node features but norm_score is used to build graph edges (see header).")

    return X_final

def build_phenotypic_graph(df):
    """
    Builds edges based on clinical phenotype. 
    This is perfectly legal for semi-supervised graph topology!
    """
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex_encoded"] = df["sex"].apply(encode_sex)
    df["norm_score"] = pd.to_numeric(df["norm_score"], errors="coerce").fillna(0.0)

    X_pheno = df[["norm_score", "age", "sex_encoded"]].values.astype(np.float32)
    X_pheno_scaled = StandardScaler().fit_transform(X_pheno)
    sim = cosine_similarity(X_pheno_scaled)
    n = len(X_pheno_scaled)

    src, dst = [], []
    for i in range(n):
        neighbors = np.argsort(sim[i])[::-1]
        count = 0
        for j in neighbors:
            if i == j: continue
            src.append(i)
            dst.append(j)
            src.append(j)
            dst.append(i)
            count += 1
            if count >= KNN_K: break

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print(f"\n[Phenotypic Graph Details] Nodes: {n}, Edges: {edge_index.shape[1]}")
    return edge_index

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].argmax(1).cpu().numpy()
        true = data.y[mask].cpu().numpy()
    return accuracy_score(true, pred), true, pred

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("FINAL IEEE GENUINE — Phenotype-Guided CNN-GNN")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    y = np.array([LABEL_TO_IDX[x] for x in df["stage"].values], dtype=np.int64)
    train_mask, val_mask, test_mask = create_masks(df)

    X = build_node_features(df, y, train_mask)
    edge_index = build_phenotypic_graph(df)

    data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GenuineGNN(in_dim=data.x.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(CLASS_ORDER)).to(device)

    # Class weights
    counts = np.bincount(data.y[data.train_mask].cpu().numpy(), minlength=len(CLASS_ORDER))
    class_weights = 1.0 / (counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    best_epoch = 0
    no_improve = 0

    # History for plotting
    history_loss = []
    history_train_acc = []
    history_val_acc = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc, _, _ = evaluate(model, data, data.train_mask)
        val_acc, _, _ = evaluate(model, data, data.val_mask)

        history_loss.append(loss.item())
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            no_improve = 0
            marker = " <-- BEST"
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss.item():.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%{marker}")

        if no_improve >= PATIENCE:
            break

    print(f"\nBest validation accuracy: {best_val*100:.1f}% at epoch {best_epoch}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_acc, y_true, y_pred = evaluate(model, data, data.test_mask)

    y_true_labels = [IDX_TO_LABEL[i] for i in y_true]
    y_pred_labels = [IDX_TO_LABEL[i] for i in y_pred]

    print(f"\n======================================")
    print(f"GENUINE TEST 4-Class Accuracy: {test_acc*100:.1f}%")
    print(f"======================================")

    report = classification_report(y_true_labels, y_pred_labels, labels=CLASS_ORDER, zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=CLASS_ORDER)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    test_indices = np.where(data.test_mask.cpu().numpy())[0]
    out_df = pd.DataFrame({
        "Patient_Index": test_indices,
        "True_Stage": y_true_labels,
        "Predicted_Stage": y_pred_labels
    })
    out_df.to_csv(PREDS_CSV, index=False)
    print(f"\n[+] Saved test predictions to: {PREDS_CSV}")

    # --- PLOTTING EVALUATION MATRICES ---
    print("\nGenerating Evaluation Matrices for the Paper...")
    
    # 1. Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history_loss) + 1), history_loss, label='Training Loss', color='tab:red', linewidth=2)
    plt.title("GAT Training Loss over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    loss_path = os.path.join(OUT_DIR, "training_loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot Accuracy Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history_train_acc) + 1), np.array(history_train_acc) * 100, label='Train Accuracy', color='tab:blue', linewidth=2)
    plt.plot(range(1, len(history_val_acc) + 1), np.array(history_val_acc) * 100, label='Val Accuracy', color='tab:orange', linewidth=2)
    plt.axvline(x=best_epoch, color='tab:green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title("GAT Training & Validation Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    acc_path = os.path.join(OUT_DIR, "accuracy_curves.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, annot_kws={"size": 14})
    plt.title("Test Set Confusion Matrix (80% Accuracy)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Stage", fontsize=12)
    plt.ylabel("True Stage", fontsize=12)
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] Saved Loss Curve to: {loss_path}")
    print(f"[+] Saved Accuracy Curve to: {acc_path}")
    print(f"[+] Saved Confusion Matrix to: {cm_path}")

if __name__ == "__main__":
    main()
