import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# STEP 2: 4-Class MRI Preprocessing
# NIfTI (.nii.gz) → normalized 3D NumPy volume (.npy)
# ============================================================

METADATA_CSV = "Outputs/metadata.csv"

BASE_DIRS = {
    "ds115": "dataset_115",
    "ucla": "dataset_ucla",
}

OUT_DIR = "Outputs/npy_dataset"

TARGET_SHAPE = (64, 64, 64)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def get_nifti_path(pid, dataset):
    return os.path.join(
        BASE_DIRS[dataset],
        pid,
        "anat",
        f"{pid}_T1w.nii.gz"
    )


def crop_nonzero_brain(volume, margin=4):
    threshold = np.percentile(volume, 15)
    coords = np.argwhere(volume > threshold)

    if coords.size == 0:
        return volume

    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin, np.array(volume.shape) - 1)

    return volume[
        lo[0]:hi[0] + 1,
        lo[1]:hi[1] + 1,
        lo[2]:hi[2] + 1,
    ]


def normalize_volume(volume):
    volume = volume.astype(np.float32)

    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)

    volume = np.clip(volume, p1, p99)

    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    return volume.astype(np.float32)


def resize_volume(volume):
    factors = (
        TARGET_SHAPE[0] / volume.shape[0],
        TARGET_SHAPE[1] / volume.shape[1],
        TARGET_SHAPE[2] / volume.shape[2],
    )

    return zoom(volume, factors, order=1).astype(np.float32)


def preprocess_mri(nifti_path):
    img = nib.load(nifti_path)
    volume = img.get_fdata()

    volume = crop_nonzero_brain(volume)
    volume = normalize_volume(volume)
    volume = resize_volume(volume)

    # Shape for CNN/GNN pipeline: (channel, depth, height, width)
    volume = np.expand_dims(volume, axis=0)

    return volume.astype(np.float32)


def save_subject(row, split_name):
    pid = row["participant_id"]
    dataset = row["dataset"]
    stage = row["stage"]
    nifti_path = row["nifti_path"]

    try:
        volume = preprocess_mri(nifti_path)

        save_dir = os.path.join(OUT_DIR, split_name, stage)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{pid}.npy")
        np.save(save_path, volume)

        return {
            "participant_id": pid,
            "dataset": dataset,
            "stage": stage,
            "split": split_name,
            "npy_path": save_path,
            "shape": str(volume.shape),
            "age": row.get("age"),
            "sex": row.get("sex"),
            "raw_score": row.get("raw_score"),
            "norm_score": row.get("norm_score"),
        }

    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        return None


def main():
    print("=" * 60)
    print("STEP 2 — 4-Class MRI Preprocessing to .npy")
    print("=" * 60)

    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError("Missing Outputs/metadata.csv. Run Step 1 first.")

    df = pd.read_csv(METADATA_CSV)
    print(f"Subjects loaded from metadata: {len(df)}")

    df["nifti_path"] = df.apply(
        lambda row: get_nifti_path(row["participant_id"], row["dataset"]),
        axis=1,
    )

    df["mri_exists"] = df["nifti_path"].apply(os.path.exists)

    missing = df[df["mri_exists"] == False]

    if len(missing) > 0:
        print("\nMissing MRI subjects:")
        print(missing[["participant_id", "dataset", "stage", "nifti_path"]].to_string(index=False))
    else:
        print("\nNo missing MRI files found.")

    available = df[df["mri_exists"] == True].copy()

    print("\nAvailable subjects:")
    print(available["stage"].value_counts())

    train_df, test_df = train_test_split(
        available,
        test_size=TEST_SIZE,
        stratify=available["stage"],
        random_state=RANDOM_STATE,
    )

    print("\nTrain split:")
    print(train_df["stage"].value_counts())

    print("\nTest split:")
    print(test_df["stage"].value_counts())

    # Only delete train/test data folders, preserve model files
    for split_dir in [os.path.join(OUT_DIR, "train"), os.path.join(OUT_DIR, "test")]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    os.makedirs(OUT_DIR, exist_ok=True)

    records = []

    print("\nProcessing train subjects...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        rec = save_subject(row, "train")
        if rec is not None:
            records.append(rec)

    print("\nProcessing test subjects...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        rec = save_subject(row, "test")
        if rec is not None:
            records.append(rec)

    out_meta = os.path.join(OUT_DIR, "npy_metadata.csv")
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_meta, index=False)

    print("\nDone.")
    print(f"Processed subjects: {len(out_df)}")
    print(f"Saved metadata: {out_meta}")

    print("\nFinal processed counts:")
    print(out_df.groupby(["split", "stage"]).size())

    print("\nSaved volume shape:")
    print("(1, 64, 64, 64)")


if __name__ == "__main__":
    main()