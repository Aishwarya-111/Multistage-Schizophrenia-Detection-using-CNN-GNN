import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.ndimage import zoom
from scipy.signal import detrend
from scipy.stats import skew, kurtosis

INPUT_CSV = "Outputs/npy_dataset/structural_features_4class.csv"
OUTPUT_CSV = "Outputs/npy_dataset/fmri_features_4class.csv"

DATASET_115 = "dataset_115"
DATASET_UCLA = "dataset_ucla"

TARGET_SHAPE = (32, 32, 32)
GRID_SIZE = 4
MAX_TIMEPOINTS = 120


def get_func_file(participant_id, dataset):
    if dataset == "ds115":
        pattern = os.path.join(DATASET_115, participant_id, "func", "*letter2backtask_bold.nii.gz")
    elif dataset == "ucla":
        pattern = os.path.join(DATASET_UCLA, participant_id, "func", "*task-rest_bold.nii.gz")
    else:
        return None

    files = glob.glob(pattern)
    return files[0] if files else None


def safe_stats(arr, prefix):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "min", "max", "median", "p10", "p90", "skew", "kurtosis"]}

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_skew": float(skew(arr)) if len(arr) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(arr)) if len(arr) > 2 else 0.0,
    }


def resize_3d(vol):
    factors = (
        TARGET_SHAPE[0] / vol.shape[0],
        TARGET_SHAPE[1] / vol.shape[1],
        TARGET_SHAPE[2] / vol.shape[2],
    )
    return zoom(vol, factors, order=1).astype(np.float32)


def zscore_ts(ts):
    ts = np.asarray(ts, dtype=np.float32)
    ts = detrend(ts, type="linear")

    std = np.std(ts)
    if std < 1e-8:
        return np.zeros_like(ts)

    return (ts - np.mean(ts)) / (std + 1e-8)


def extract_region_timeseries(func_path):
    img = nib.load(func_path)
    data = img.get_fdata().astype(np.float32)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI, got {data.shape}")

    tmax = min(data.shape[3], MAX_TIMEPOINTS)
    data = data[:, :, :, :tmax]

    resized = []
    for t in range(data.shape[-1]):
        vol = data[:, :, :, t]

        p1, p99 = np.percentile(vol, [1, 99])
        vol = np.clip(vol, p1, p99)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

        resized.append(resize_3d(vol))

    data = np.stack(resized, axis=-1)

    mean_img = np.mean(data, axis=-1)
    brain_mask = mean_img > 0.05

    gx = gy = gz = GRID_SIZE
    sx = TARGET_SHAPE[0] // gx
    sy = TARGET_SHAPE[1] // gy
    sz = TARGET_SHAPE[2] // gz

    region_ts = []

    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                block = data[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz, :]
                mask = brain_mask[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz]

                if mask.sum() < 5:
                    ts = np.zeros(data.shape[-1], dtype=np.float32)
                else:
                    ts = block[mask].mean(axis=0)

                region_ts.append(zscore_ts(ts))

    return np.array(region_ts, dtype=np.float32)


def extract_fmri_features(func_path):
    features = {}

    region_ts = extract_region_timeseries(func_path)

    corr = np.corrcoef(region_ts)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    upper = corr[np.triu_indices_from(corr, k=1)]

    features.update(safe_stats(upper, "fc"))
    features["fc_abs_mean"] = float(np.mean(np.abs(upper)))
    features["fc_positive_ratio"] = float(np.mean(upper > 0))
    features["fc_negative_ratio"] = float(np.mean(upper < 0))
    features["fc_strong_positive_ratio"] = float(np.mean(upper > 0.5))
    features["fc_strong_negative_ratio"] = float(np.mean(upper < -0.5))

    mean_conn = corr.mean(axis=1)
    abs_conn = np.abs(corr).mean(axis=1)

    for i in range(64):
        features[f"fmri_region_{i:02d}_mean_conn"] = float(mean_conn[i])
        features[f"fmri_region_{i:02d}_abs_conn"] = float(abs_conn[i])

    # Approximate network group summaries
    groups = {
        "fmri_frontal_proxy": list(range(0, 16)),
        "fmri_central_proxy": list(range(16, 48)),
        "fmri_posterior_proxy": list(range(48, 64)),
        "fmri_left_proxy": [i for i in range(64) if i // 16 < 2],
        "fmri_right_proxy": [i for i in range(64) if i // 16 >= 2],
    }

    for name, idxs in groups.items():
        subcorr = corr[np.ix_(idxs, idxs)]
        vals = subcorr[np.triu_indices_from(subcorr, k=1)]
        features.update(safe_stats(vals, name))

    # Keep selected edge features
    max_edges = min(500, len(upper))
    for i in range(max_edges):
        features[f"fc_edge_{i:03d}"] = float(upper[i])

    return features


def main():
    print("=" * 60)
    print("STEP 5 — Enhanced fMRI Connectivity Feature Extraction")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Run 04_extract_structural_features.py first.")

    df = pd.read_csv(INPUT_CSV)

    records = []
    missing = 0
    errors = 0

    for _, row in df.iterrows():
        pid = row["participant_id"]
        dataset = row["dataset"]

        func_path = get_func_file(pid, dataset)

        if func_path is None:
            print(f"[MISSING] {pid} ({dataset})")
            missing += 1
            continue

        try:
            feats = extract_fmri_features(func_path)
            base = row.to_dict()
            base["func_path"] = func_path
            base.update(feats)
            records.append(base)
            print(f"[OK] {pid} ({dataset})")

        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            errors += 1

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Subjects with fMRI processed: {len(out_df)}")
    print(f"Missing fMRI subjects       : {missing}")
    print(f"Errors                      : {errors}")
    print(f"Saved → {OUTPUT_CSV}")
    print(f"Total columns/features      : {out_df.shape[1]}")


if __name__ == "__main__":
    main()