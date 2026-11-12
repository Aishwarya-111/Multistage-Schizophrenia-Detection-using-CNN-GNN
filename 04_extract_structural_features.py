import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

INPUT_CSV = "Outputs/npy_dataset/cnn_embeddings_4class.csv"
OUTPUT_CSV = "Outputs/npy_dataset/structural_features_4class.csv"


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


def region_features(volume, mask, region_slice, name):
    block = volume[region_slice]
    block_mask = mask[region_slice]
    voxels = block[block_mask]

    feats = {}
    feats[f"{name}_density"] = float(block_mask.mean())
    feats.update(safe_stats(voxels, name))
    return feats


def extract_features(npy_path):
    volume = np.load(npy_path).astype(np.float32)[0]  # (64,64,64)

    features = {}

    brain_mask = volume > 0.05
    brain_voxels = volume[brain_mask]

    features["brain_voxel_count"] = int(brain_mask.sum())
    features["brain_volume_ratio"] = float(brain_mask.mean())
    features.update(safe_stats(brain_voxels, "whole_brain"))

    # Tissue/intensity proxies
    low = (volume > 0.05) & (volume <= 0.33)
    mid = (volume > 0.33) & (volume <= 0.66)
    high = volume > 0.66

    features["low_intensity_ratio"] = float(low.mean())
    features["mid_intensity_ratio"] = float(mid.mean())
    features["high_intensity_ratio"] = float(high.mean())

    x, y, z = volume.shape
    xm, ym, zm = x // 2, y // 2, z // 2

    # Hemispheric asymmetry
    left = volume[:xm]
    right = volume[xm:]
    features["left_right_mean_diff"] = float(abs(left.mean() - right.mean()))
    features["left_right_volume_diff"] = float(abs((left > 0.05).mean() - (right > 0.05).mean()))

    # Approximate lobar/clinical regions
    regions = {
        "left_frontal": (slice(0, xm), slice(0, y // 3), slice(z // 3, z)),
        "right_frontal": (slice(xm, x), slice(0, y // 3), slice(z // 3, z)),

        "left_temporal": (slice(0, xm), slice(y // 3, 2 * y // 3), slice(0, z // 2)),
        "right_temporal": (slice(xm, x), slice(y // 3, 2 * y // 3), slice(0, z // 2)),

        "left_parietal": (slice(0, xm), slice(y // 3, 2 * y // 3), slice(z // 2, z)),
        "right_parietal": (slice(xm, x), slice(y // 3, 2 * y // 3), slice(z // 2, z)),

        "left_occipital": (slice(0, xm), slice(2 * y // 3, y), slice(z // 3, z)),
        "right_occipital": (slice(xm, x), slice(2 * y // 3, y), slice(z // 3, z)),

        "central_deep": (slice(x // 4, 3 * x // 4), slice(y // 4, 3 * y // 4), slice(z // 4, 3 * z // 4)),
        "superior": (slice(0, x), slice(0, y), slice(zm, z)),
        "inferior": (slice(0, x), slice(0, y), slice(0, zm)),
        "anterior": (slice(0, x), slice(0, ym), slice(0, z)),
        "posterior": (slice(0, x), slice(ym, y), slice(0, z)),
    }

    for name, slc in regions.items():
        features.update(region_features(volume, brain_mask, slc, name))

    # Hippocampus-like rough medial-inferior temporal proxies
    hippocampus_regions = {
        "left_hippocampus_proxy": (slice(x // 4, xm), slice(y // 3, 2 * y // 3), slice(z // 4, z // 2)),
        "right_hippocampus_proxy": (slice(xm, 3 * x // 4), slice(y // 3, 2 * y // 3), slice(z // 4, z // 2)),
    }

    for name, slc in hippocampus_regions.items():
        features.update(region_features(volume, brain_mask, slc, name))

    features["hippocampus_proxy_asymmetry"] = abs(
        features["left_hippocampus_proxy_mean"] - features["right_hippocampus_proxy_mean"]
    )

    # Ventricle/CSF-like central low/high intensity proxies
    center = volume[x // 3:2 * x // 3, y // 3:2 * y // 3, z // 3:2 * z // 3]
    features["central_low_intensity_ratio"] = float(((center > 0.05) & (center <= 0.30)).mean())
    features["central_high_intensity_ratio"] = float((center > 0.66).mean())
    features.update(safe_stats(center, "central_cube"))

    # Dense regional grid features: 4x4x4
    gx = gy = gz = 4
    sx, sy, sz = x // gx, y // gy, z // gz

    idx = 0
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                block = volume[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz]
                block_mask = brain_mask[i*sx:(i+1)*sx, j*sy:(j+1)*sy, k*sz:(k+1)*sz]
                voxels = block[block_mask]

                features[f"region_{idx:02d}_density"] = float(block_mask.mean())
                features.update(safe_stats(voxels, f"region_{idx:02d}"))
                idx += 1

    return features


def main():
    print("=" * 60)
    print("STEP 4 — Enhanced Regional Structural Feature Extraction")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Run 03_train_binary_cnn_embeddings.py first.")

    df = pd.read_csv(INPUT_CSV)
    records = []

    for idx, row in df.iterrows():
        npy_path = row["npy_path"]

        if not os.path.exists(npy_path):
            print(f"[SKIP] Missing npy: {npy_path}")
            continue

        feats = extract_features(npy_path)
        base = row.to_dict()
        base.update(feats)
        records.append(base)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} subjects")

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Subjects processed: {len(out_df)}")
    print(f"Saved → {OUTPUT_CSV}")
    print(f"Total columns/features: {out_df.shape[1]}")


if __name__ == "__main__":
    main()