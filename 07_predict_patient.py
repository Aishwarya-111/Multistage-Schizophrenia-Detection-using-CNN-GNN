import os
import ast
import random
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================
# LIVE PATIENT PREDICTION DEMO
# ============================================================

INPUT_CSV = "Outputs/npy_dataset/fmri_features_4class.csv"
MODEL_PATH = "Outputs/gnn_results/best_gnn_model.pth"

CLASS_ORDER = ["Healthy", "Early", "Moderate", "Chronic"]
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}

# Architecture MUST match 10_genuine_ieee_gnn.py exactly
HIDDEN_DIM = 64
DROPOUT = 0.50
KNN_K = 5
TOP_K_MRI_FEATURES = 20
RANDOM_STATE = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def load_data():
    df = pd.read_csv(INPUT_CSV)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex_encoded"] = df["sex"].apply(encode_sex)
    df["norm_score"] = pd.to_numeric(df["norm_score"], errors="coerce").fillna(0.0)

    # Masks
    train_indices = df[df["split"] == "train"].index.values
    test_indices = df[df["split"] == "test"].index.values
    y_str = df["stage"].values
    y = np.array([{c: i for i, c in enumerate(CLASS_ORDER)}[x] for x in y_str], dtype=np.int64)

    train_idx, val_idx = train_test_split(
        train_indices, test_size=0.20, stratify=df.loc[train_indices, "stage"], random_state=RANDOM_STATE
    )

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[test_indices] = True

    # 1. Parse CNN Embeddings
    embeddings = []
    for emb_str in df["embedding"]:
        try:
            emb = ast.literal_eval(emb_str)
            if len(emb) != 512: emb = [0.0] * 512
        except:
            emb = [0.0] * 512
        embeddings.append(emb)
    X_cnn = np.array(embeddings, dtype=np.float32)

    # 2. Base Demographics
    X_base = df[["prob_healthy", "prob_schizophrenia", "age", "sex_encoded"]].values.astype(np.float32)

    # 3. Structural Features
    candidate_cols = [col for col in df.columns if (col.startswith("region_") or col.startswith("left_") or col.startswith("right_") or col.startswith("central_") or col.startswith("fmri_") or col.startswith("fc_")) and pd.api.types.is_numeric_dtype(df[col])]
    X_cand = df[candidate_cols].replace([np.inf, -np.inf], np.nan).fillna(df[candidate_cols].median(numeric_only=True)).values.astype(np.float32)

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

    k = min(TOP_K_MRI_FEATURES, X_cand_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_cand_scaled[train_idx], y[train_idx])
    X_cand_selected = selector.transform(X_cand_scaled)
    
    X_final = np.hstack([X_cnn_scaled, X_base_scaled, X_cand_selected]).astype(np.float32)

    # Phenotypic Graph
    X_pheno = df[["norm_score", "age", "sex_encoded"]].values.astype(np.float32)
    X_pheno_scaled = StandardScaler().fit_transform(X_pheno)
    sim = cosine_similarity(X_pheno_scaled)
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

    return df, X_final, edge_index, y, test_mask

def predict_patient(patient_idx, df, model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[patient_idx].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_stage = IDX_TO_LABEL[pred_idx]
        confidence = probs[pred_idx] * 100

        print("\n" + "="*50)
        print(f"🏥 CLINICAL DIAGNOSTIC REPORT - Patient #{patient_idx}")
        print("="*50)
        print(f"Demographics       : Age {df.iloc[patient_idx]['age']}, Sex {df.iloc[patient_idx]['sex']}")
        print(f"Symptom Severity   : {df.iloc[patient_idx]['norm_score']} (Used ONLY for Phenotypic Graph connections)")
        print(f"CNN Brain Analysis : Schizophrenia {df.iloc[patient_idx]['prob_schizophrenia']*100:.1f}%")
        print("-" * 50)
        print(f"GAT PREDICTED STAGE: {pred_stage} ({confidence:.1f}% Confidence)")
        print("="*50)
        print("Probability Distribution:")
        for i, c in enumerate(CLASS_ORDER):
            print(f"  - {c:10}: {probs[i]*100:.1f}%")

def main():
    set_seed(RANDOM_STATE)
    print("Loading Graph Architecture and Patient Database...")
    df, X_final, edge_index, y, test_mask = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=torch.tensor(X_final, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    data = data.to(device)

    model = GenuineGNN(in_dim=data.x.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(CLASS_ORDER)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_participants = df[df["split"] == "test"]["participant_id"].values.tolist()
    train_participants = df[df["split"] == "train"]["participant_id"].values.tolist()

    while True:
        print(f"\nAvailable TEST Patients : {test_participants[:5]} ... (20 total)")
        print(f"Available TRAIN Patients: {train_participants[:5]} ... (77 total)")
        choice = input("Enter Patient ID (e.g., 'sub-50032') or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            break
            
        if choice not in df["participant_id"].values:
            print("❌ Error: Patient ID not found in database. Please enter a valid ID like 'sub-50032'.")
            continue
            
        # Get the row index for this patient
        pid = df[df["participant_id"] == choice].index[0]
            
        predict_patient(pid, df, model, data)

if __name__ == "__main__":
    main()
