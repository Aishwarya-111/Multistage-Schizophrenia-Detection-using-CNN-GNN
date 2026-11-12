import os
import ast
import uuid
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from scipy.stats import skew, kurtosis
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18

from torch_geometric.nn import GATConv

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB upload limit

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_SLICES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'slices')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_SLICES_FOLDER, exist_ok=True)

# ============================================================
# PATHS
# ============================================================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_CSV         = os.path.join(BASE_DIR, "Outputs", "npy_dataset", "fmri_features_4class.csv")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "Outputs", "npy_dataset", "best_binary_cnn_for_embeddings.pth")
GNN_MODEL_PATH = os.path.join(BASE_DIR, "Outputs", "gnn_results", "best_gnn_model.pth")

# ============================================================
# CONSTANTS
# ============================================================
CLASS_ORDER    = ["Healthy", "Early", "Moderate", "Chronic"]
IDX_TO_LABEL   = {i: c for i, c in enumerate(CLASS_ORDER)}
KNN_K          = 5
HIDDEN_DIM     = 64
TOP_K_MRI_FEATURES = 20
SLICES_PER_AXIS    = 8

# ============================================================
# MODEL ARCHITECTURES  (must be defined BEFORE loading)
# ============================================================
class BinaryMRIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
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


class GenuineGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.50, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.50, training=self.training)
        return self.conv3(x, edge_index)

# ============================================================
# LOAD CNN ONCE AT STARTUP  (now safe — class is defined above)
# ============================================================
for _p in (DB_CSV, CNN_MODEL_PATH, GNN_MODEL_PATH):
    if not os.path.exists(_p):
        print(f"WARNING: required file missing: {_p}")

device = torch.device("cpu")
print("Loading CNN model...")
cnn_model = BinaryMRIModel().to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.eval()
print("CNN model loaded successfully!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def crop_nonzero_brain(volume, margin=4):
    threshold = np.percentile(volume, 15)
    coords = np.argwhere(volume > threshold)
    if coords.size == 0:
        return volume
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin, np.array(volume.shape) - 1)
    return volume[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]


def normalize_volume(volume):
    volume = volume.astype(np.float32)
    p1, p99 = np.percentile(volume, 1), np.percentile(volume, 99)
    volume = np.clip(volume, p1, p99)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume.astype(np.float32)


def resize_volume(volume):
    factors = (64.0/volume.shape[0], 64.0/volume.shape[1], 64.0/volume.shape[2])
    return zoom(volume, factors, order=1).astype(np.float32)


def get_slice_indices(length, n=SLICES_PER_AXIS):
    start = int(length * 0.25)
    end   = int(length * 0.75)
    if end <= start:
        start, end = 0, length - 1
    idxs = np.linspace(start, end, n, dtype=int)
    return np.clip(idxs, 0, length - 1).tolist()


def volume_to_slices(volume):
    vol, slices = volume[0], []
    for axis in range(3):
        for idx in get_slice_indices(vol.shape[axis], SLICES_PER_AXIS):
            if axis == 0:   sl = vol[idx, :, :]
            elif axis == 1: sl = vol[:, idx, :]
            else:           sl = vol[:, :, idx]
            slices.append(np.clip(sl, 0, 1).astype(np.float32))
    return slices


def safe_stats(arr, prefix):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["mean","std","min","max","median","p10","p90","skew","kurtosis"]}
    return {
        f"{prefix}_mean":     float(np.mean(arr)),
        f"{prefix}_std":      float(np.std(arr)),
        f"{prefix}_min":      float(np.min(arr)),
        f"{prefix}_max":      float(np.max(arr)),
        f"{prefix}_median":   float(np.median(arr)),
        f"{prefix}_p10":      float(np.percentile(arr, 10)),
        f"{prefix}_p90":      float(np.percentile(arr, 90)),
        f"{prefix}_skew":     float(skew(arr))     if len(arr) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(arr)) if len(arr) > 2 else 0.0,
    }


def region_features(volume, mask, region_slice, name):
    block      = volume[region_slice]
    block_mask = mask[region_slice]
    voxels     = block[block_mask]
    feats      = {f"{name}_density": float(block_mask.mean())}
    feats.update(safe_stats(voxels, name))
    return feats


def extract_structural_features(volume_64):
    features   = {}
    brain_mask = volume_64 > 0.05
    brain_voxels = volume_64[brain_mask]

    features["brain_voxel_count"]  = int(brain_mask.sum())
    features["brain_volume_ratio"] = float(brain_mask.mean())
    features.update(safe_stats(brain_voxels, "whole_brain"))

    low  = (volume_64 > 0.05) & (volume_64 <= 0.33)
    mid  = (volume_64 > 0.33) & (volume_64 <= 0.66)
    high =  volume_64 > 0.66
    features["low_intensity_ratio"]  = float(low.mean())
    features["mid_intensity_ratio"]  = float(mid.mean())
    features["high_intensity_ratio"] = float(high.mean())

    x, y, z   = volume_64.shape
    xm, ym, zm = x//2, y//2, z//2
    left, right = volume_64[:xm], volume_64[xm:]
    features["left_right_mean_diff"]   = float(abs(left.mean() - right.mean()))
    features["left_right_volume_diff"] = float(abs((left>0.05).mean() - (right>0.05).mean()))

    regions = {
        "left_frontal":   (slice(0,xm),  slice(0,y//3),       slice(z//3,z)),
        "right_frontal":  (slice(xm,x),  slice(0,y//3),       slice(z//3,z)),
        "left_temporal":  (slice(0,xm),  slice(y//3,2*y//3),  slice(0,z//2)),
        "right_temporal": (slice(xm,x),  slice(y//3,2*y//3),  slice(0,z//2)),
        "left_parietal":  (slice(0,xm),  slice(y//3,2*y//3),  slice(z//2,z)),
        "right_parietal": (slice(xm,x),  slice(y//3,2*y//3),  slice(z//2,z)),
        "left_occipital": (slice(0,xm),  slice(2*y//3,y),     slice(z//3,z)),
        "right_occipital":(slice(xm,x),  slice(2*y//3,y),     slice(z//3,z)),
        "central_deep":   (slice(x//4,3*x//4), slice(y//4,3*y//4), slice(z//4,3*z//4)),
        "superior":       (slice(0,x),   slice(0,y),          slice(zm,z)),
        "inferior":       (slice(0,x),   slice(0,y),          slice(0,zm)),
        "anterior":       (slice(0,x),   slice(0,ym),         slice(0,z)),
        "posterior":      (slice(0,x),   slice(ym,y),         slice(0,z)),
    }
    for name, slc in regions.items():
        features.update(region_features(volume_64, brain_mask, slc, name))

    hippocampus_regions = {
        "left_hippocampus_proxy":  (slice(x//4,xm),    slice(y//3,2*y//3), slice(z//4,z//2)),
        "right_hippocampus_proxy": (slice(xm,3*x//4),  slice(y//3,2*y//3), slice(z//4,z//2)),
    }
    for name, slc in hippocampus_regions.items():
        features.update(region_features(volume_64, brain_mask, slc, name))

    features["hippocampus_proxy_asymmetry"] = abs(
        features["left_hippocampus_proxy_mean"] - features["right_hippocampus_proxy_mean"]
    )

    center = volume_64[x//3:2*x//3, y//3:2*y//3, z//3:2*z//3]
    features["central_low_intensity_ratio"]  = float(((center>0.05)&(center<=0.30)).mean())
    features["central_high_intensity_ratio"] = float((center>0.66).mean())
    features.update(safe_stats(center, "central_cube"))

    gx = gy = gz = 4
    sx, sy, sz = x//gx, y//gy, z//gz
    idx = 0
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                block      = volume_64[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz]
                block_mask = brain_mask[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz]
                voxels     = block[block_mask]
                features[f"region_{idx:02d}_density"] = float(block_mask.mean())
                features.update(safe_stats(voxels, f"region_{idx:02d}"))
                idx += 1

    return features


def encode_sex(x):
    x = str(x).upper().strip()
    if x in ["M","MALE","1.0","1"]:   return 1.0
    if x in ["F","FEMALE","0.0","0"]: return 0.0
    return -1.0


# ============================================================
# API ENDPOINTS
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/cohort-stats', methods=['GET'])
def get_cohort_stats():
    try:
        df = pd.read_csv(DB_CSV)
        stages    = {str(k): int(v)   for k,v in df['stage'].value_counts().items()}
        datasets  = {str(k): int(v)   for k,v in df['dataset'].value_counts().items()}
        avg_scores= {str(k): float(v) for k,v in df.groupby('stage')['norm_score'].mean().items()}
        ages      = [float(x) for x in df['age'].dropna().tolist()]
        scz_probs = {str(k): float(v) for k,v in df.groupby('stage')['prob_schizophrenia'].mean().items()}
        sex_counts= {str(k): int(v)   for k,v in df['sex'].apply(
            lambda s: 'Male' if str(s).upper().strip() in ['M','MALE'] else 'Female'
        ).value_counts().items()}
        return jsonify({
            "stages": stages, "datasets": datasets,
            "avg_symptom_scores": avg_scores, "avg_scz_probabilities": scz_probs,
            "sex_counts": sex_counts, "age_list": ages, "total_patients": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cohort-graph', methods=['GET'])
def get_cohort_graph():
    try:
        df = pd.read_csv(DB_CSV)
        df["sex_encoded"] = df["sex"].apply(encode_sex)
        df["age"]         = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["norm_score"]  = pd.to_numeric(df["norm_score"], errors="coerce").fillna(0.0)

        nodes = []
        for i, row in df.iterrows():
            nodes.append({
                "id": str(row["participant_id"]), "index": i,
                "dataset": str(row["dataset"]),   "stage": str(row["stage"]),
                "age": float(row["age"]),
                "sex": "Male" if row["sex_encoded"]==1.0 else "Female",
                "norm_score": float(row["norm_score"]),
                "prob_schizophrenia": float(row["prob_schizophrenia"]),
                "is_new": False
            })

        X_pheno        = df[["norm_score","age","sex_encoded"]].values.astype(np.float32)
        X_pheno_scaled = StandardScaler().fit_transform(X_pheno)
        sim = cosine_similarity(X_pheno_scaled)
        n   = len(X_pheno_scaled)

        edges = []
        for i in range(n):
            neighbors = np.argsort(sim[i])[::-1]
            count = 0
            for j in neighbors:
                if i == j: continue
                edges.append({
                    "source": str(df.iloc[i]["participant_id"]),
                    "target": str(df.iloc[j]["participant_id"]),
                    "similarity": float(sim[i,j])
                })
                count += 1
                if count >= 3: break

        return jsonify({"nodes": nodes, "links": edges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    session_id       = str(uuid.uuid4())
    temp_nii_path    = os.path.join(UPLOAD_FOLDER, f"{session_id}.nii.gz")
    session_slice_dir= os.path.join(STATIC_SLICES_FOLDER, session_id)

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        age  = float(request.form.get('age', 40.0))
        sex  = request.form.get('sex', 'M').upper()

        sans_blunt     = float(request.form.get('sans_blunt',     0.0))
        sans_alogia    = float(request.form.get('sans_alogia',    0.0))
        sans_avolition = float(request.form.get('sans_avolition', 0.0))
        sans_anhedonia = float(request.form.get('sans_anhedonia', 0.0))
        sans_attention = float(request.form.get('sans_attention', 0.0))
        saps_halluc    = float(request.form.get('saps_halluc',    0.0))
        saps_delus     = float(request.form.get('saps_delus',     0.0))
        saps_behav     = float(request.form.get('saps_behav',     0.0))
        saps_thought   = float(request.form.get('saps_thought',   0.0))
        saps_inapp     = float(request.form.get('saps_inapp',     0.0))

        raw_score  = (sans_blunt + sans_alogia + sans_avolition + sans_anhedonia + sans_attention +
                      saps_halluc + saps_delus + saps_behav + saps_thought + saps_inapp)
        norm_score = (raw_score / 50.0) * 100.0

        file.save(temp_nii_path)

        # ── STEP 1: Preprocess MRI ───────────────────────────────────
        print(f"[{session_id}] Preprocessing MRI...")
        img          = nib.load(temp_nii_path)
        volume_raw   = img.get_fdata()
        volume_cropped = crop_nonzero_brain(volume_raw)
        volume_norm    = normalize_volume(volume_cropped)
        volume_resized = resize_volume(volume_norm)
        del volume_raw, volume_cropped, volume_norm
        gc.collect()

        # Save slices for viewer
        os.makedirs(session_slice_dir, exist_ok=True)
        for i in range(0, 64, 4):
            for axis_name, sl in [
                ("axial",    volume_resized[:, :, i]),
                ("coronal",  volume_resized[:, i, :]),
                ("sagittal", volume_resized[i, :, :]),
            ]:
                img_out = Image.fromarray((sl * 255).astype(np.uint8))
                img_out.save(os.path.join(session_slice_dir, f"{axis_name}_{i}.png"))
        gc.collect()

        volume_expanded = np.expand_dims(volume_resized, axis=0)

        # ── STEP 2: CNN Embeddings ───────────────────────────────────
        print(f"[{session_id}] Running CNN...")
        slices  = volume_to_slices(volume_expanded)
        eval_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        slice_embs, slice_probs = [], []
        with torch.no_grad():
            for sl in slices:
                x      = eval_tf(sl).repeat(3,1,1).unsqueeze(0).to(device)
                logits, emb = cnn_model(x, return_embedding=True)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                slice_embs.append(emb.cpu().numpy()[0])
                slice_probs.append(probs)

        mean_emb          = np.mean(np.vstack(slice_embs),  axis=0)
        mean_prob         = np.mean(np.vstack(slice_probs), axis=0)
        prob_healthy      = float(mean_prob[0])
        prob_schizophrenia= float(mean_prob[1])
        del slices, slice_embs, slice_probs
        gc.collect()

        # ── STEP 3: Structural Features ──────────────────────────────
        print(f"[{session_id}] Extracting structural features...")
        struct_feats = extract_structural_features(volume_resized)
        del volume_resized, volume_expanded
        gc.collect()

        # ── STEP 4: GNN Inference ────────────────────────────────────
        print(f"[{session_id}] Running GNN...")
        df_db = pd.read_csv(DB_CSV)

        new_row = {
            "participant_id": "new_patient",
            "dataset":        "clinical_upload",
            "stage":          "Early",
            "binary_label":   "Schizophrenia" if prob_schizophrenia >= 0.5 else "Healthy",
            "split":          "test",
            "age":            age,
            "sex":            sex,
            "raw_score":      raw_score,
            "norm_score":     norm_score,
            "npy_path":       f"uploads/{session_id}.npy",
            "embedding":      str(mean_emb.tolist()),
            "prob_healthy":   prob_healthy,
            "prob_schizophrenia": prob_schizophrenia,
            "binary_pred":    "Schizophrenia" if prob_schizophrenia >= 0.5 else "Healthy",
        }
        new_row.update(struct_feats)

        fmri_cols = [c for c in df_db.columns if c.startswith("fmri_") or c.startswith("fc_")]
        for col in fmri_cols:
            new_row[col] = float(df_db[col].median(numeric_only=True))

        df_merged = pd.concat([df_db, pd.DataFrame([new_row])], ignore_index=True)
        df_merged["age"]         = pd.to_numeric(df_merged["age"], errors="coerce").fillna(df_merged["age"].median())
        df_merged["sex_encoded"] = df_merged["sex"].apply(encode_sex)
        df_merged["norm_score"]  = pd.to_numeric(df_merged["norm_score"], errors="coerce").fillna(0.0)

        train_idx = df_merged[df_merged["split"] == "train"].index.values

        embeddings = []
        for emb_str in df_merged["embedding"]:
            try:
                emb = ast.literal_eval(emb_str)
                if len(emb) != 512: emb = [0.0] * 512
            except:
                emb = [0.0] * 512
            embeddings.append(emb)
        X_cnn  = np.array(embeddings, dtype=np.float32)
        X_base = df_merged[["prob_healthy","prob_schizophrenia","age","sex_encoded"]].values.astype(np.float32)

        candidate_cols = [
            col for col in df_merged.columns
            if (col.startswith("region_") or col.startswith("left_") or
                col.startswith("right_") or col.startswith("central_") or
                col.startswith("fmri_")  or col.startswith("fc_"))
            and pd.api.types.is_numeric_dtype(df_merged[col])
        ]
        X_cand = (df_merged[candidate_cols]
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(df_merged[candidate_cols].median(numeric_only=True))
                  .values.astype(np.float32))

        non_train_idx = np.setdiff1d(np.arange(len(X_cnn)), train_idx)

        def scale(X):
            sc = StandardScaler()
            out = X.copy()
            out[train_idx]     = sc.fit_transform(X[train_idx])
            out[non_train_idx] = sc.transform(X[non_train_idx])
            return out

        X_cnn_scaled  = scale(X_cnn)
        X_base_scaled = scale(X_base)
        X_cand_scaled = scale(X_cand)

        y_stages = np.array(
            [CLASS_ORDER.index(s) for s in df_merged["stage"].values], dtype=np.int64
        )
        k        = min(TOP_K_MRI_FEATURES, X_cand_scaled.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_cand_scaled[train_idx], y_stages[train_idx])
        X_cand_selected = selector.transform(X_cand_scaled)

        X_final = np.hstack([X_cnn_scaled, X_base_scaled, X_cand_selected]).astype(np.float32)

        X_pheno        = df_merged[["norm_score","age","sex_encoded"]].values.astype(np.float32)
        X_pheno_scaled = StandardScaler().fit_transform(X_pheno)
        sim = cosine_similarity(X_pheno_scaled)
        n   = len(X_pheno_scaled)

        src, dst, new_patient_neighbors = [], [], []
        for i in range(n):
            neighbors = np.argsort(sim[i])[::-1]
            count = 0
            for j in neighbors:
                if i == j: continue
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)
                if i == n - 1:
                    new_patient_neighbors.append({
                        "participant_id": str(df_merged.iloc[j]["participant_id"]),
                        "dataset":        str(df_merged.iloc[j]["dataset"]),
                        "stage":          str(df_merged.iloc[j]["stage"]),
                        "age":            float(df_merged.iloc[j]["age"]),
                        "sex":            "Male" if df_merged.iloc[j]["sex_encoded"]==1.0 else "Female",
                        "norm_score":     float(df_merged.iloc[j]["norm_score"]),
                        "similarity":     float(sim[i,j]) * 100.0
                    })
                count += 1
                if count >= KNN_K: break

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        gnn_model = GenuineGNN(in_dim=X_final.shape[1], hidden_dim=HIDDEN_DIM, num_classes=4).to(device)
        gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
        gnn_model.eval()

        with torch.no_grad():
            x_tensor  = torch.tensor(X_final, dtype=torch.float32).to(device)
            logits    = gnn_model(x_tensor, edge_index.to(device))
            probs     = F.softmax(logits, dim=1)[n-1].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_stage = CLASS_ORDER[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0
        del gnn_model, x_tensor, logits
        gc.collect()

        # Build D3 graph data
        d3_nodes = []
        for i, row in df_merged.iterrows():
            d3_nodes.append({
                "id":      str(row["participant_id"]),
                "dataset": str(row["dataset"]),
                "stage":   pred_stage if i == n-1 else str(row["stage"]),
                "age":     float(row["age"]),
                "sex":     "Male" if row["sex_encoded"]==1.0 else "Female",
                "norm_score":         float(row["norm_score"]),
                "prob_schizophrenia": float(row["prob_schizophrenia"]),
                "is_new": bool(i == n-1)
            })

        d3_links = []
        for i in range(n):
            neighbors = np.argsort(sim[i])[::-1]
            count = 0
            for j in neighbors:
                if i == j: continue
                d3_links.append({
                    "source":     str(df_merged.iloc[i]["participant_id"]),
                    "target":     str(df_merged.iloc[j]["participant_id"]),
                    "similarity": float(sim[i,j]),
                    "is_new_link": bool(i == n-1 or j == n-1)
                })
                count += 1
                if count >= 3: break

        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)

        return jsonify({
            "session_id": session_id,
            "prediction": {
                "predicted_stage": pred_stage,
                "confidence":      confidence,
                "probabilities": {
                    "Healthy":  float(probs[0]) * 100.0,
                    "Early":    float(probs[1]) * 100.0,
                    "Moderate": float(probs[2]) * 100.0,
                    "Chronic":  float(probs[3]) * 100.0
                }
            },
            "patient_info": {
                "age":  age,
                "sex":  "Male" if sex == 'M' else "Female",
                "raw_symptom_score":        raw_score,
                "normalized_symptom_score": norm_score,
                "cnn_binary_healthy_prob":  prob_healthy       * 100.0,
                "cnn_binary_schz_prob":     prob_schizophrenia * 100.0
            },
            "key_structural_features": {
                "brain_voxel_count":    struct_feats["brain_voxel_count"],
                "brain_volume_ratio":   struct_feats["brain_volume_ratio"]          * 100.0,
                "left_right_mean_diff": struct_feats["left_right_mean_diff"]         * 100.0,
                "left_right_volume_diff": struct_feats["left_right_volume_diff"]     * 100.0,
                "hippocampus_asymmetry":  struct_feats["hippocampus_proxy_asymmetry"]* 100.0,
                "central_ventricle_ratio":struct_feats["central_low_intensity_ratio"]* 100.0
            },
            "phenotypic_neighbors": new_patient_neighbors,
            "graph_data": {"nodes": d3_nodes, "links": d3_links}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)
        return jsonify({"error": str(e)}), 500


@app.route('/static/slices/<session_id>/<filename>')
def serve_slices(session_id, filename):
    return send_from_directory(os.path.join(STATIC_SLICES_FOLDER, session_id), filename)


# ============================================================
# START
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)