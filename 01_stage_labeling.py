import os
import pandas as pd
import numpy as np

# ============================================================
# STEP 1: 4-Class Clinical Stage Labeling
# Healthy / Early / Moderate / Chronic
# ============================================================

DIR_DS115 = "dataset_115"
DIR_UCLA = "dataset_ucla"
OUTPUT_CSV = "Outputs/metadata.csv"

RANDOM_STATE = 42
PREFER_UCLA_HEALTHY = True

DS115_SAPS = ["saps7", "saps20", "saps25", "saps34"]
DS115_SANS = ["sans8", "sans13", "sans17", "sans22", "sans25"]
DS115_MAX = (len(DS115_SAPS) + len(DS115_SANS)) * 5

UCLA_SANS_G = [
    "global_bluntaffect",
    "global_alogia",
    "global_anhedonia",
    "global_avolition",
    "global_attention",
]

UCLA_SAPS_G = [
    "global_hallucinations",
    "global_delusions",
    "global_posformalthought",
    "global_bizarrebehav",
    "global_inappaffect",
]

UCLA_MAX = (len(UCLA_SANS_G) + len(UCLA_SAPS_G)) * 5


def get_nifti_path(pid, dataset):
    if dataset == "ds115":
        return os.path.join(DIR_DS115, pid, "anat", f"{pid}_T1w.nii.gz")
    if dataset == "ucla":
        return os.path.join(DIR_UCLA, pid, "anat", f"{pid}_T1w.nii.gz")
    return ""


def safe_sum(row, cols):
    total = 0.0
    for col in cols:
        value = pd.to_numeric(row.get(col, 0), errors="coerce")
        if pd.isna(value):
            value = 0
        total += value
    return total


def assign_stage(score, p33, p66):
    if score <= p33:
        return "Early"
    elif score <= p66:
        return "Moderate"
    else:
        return "Chronic"


def load_ds115():
    path = os.path.join(DIR_DS115, "participants1.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path, sep="\t")
    records = []

    for _, row in df.iterrows():
        pid = row["participant_id"]
        cond = str(row.get("condit", "")).upper().strip()
        age = pd.to_numeric(row.get("age"), errors="coerce")
        sex = str(row.get("gender", "")).upper().strip()

        if cond in ["CON", "CON-SIB"]:
            records.append({
                "participant_id": pid,
                "dataset": "ds115",
                "age": age,
                "sex": sex,
                "stage": "Healthy",
                "raw_score": np.nan,
                "norm_score": np.nan,
            })

        elif cond == "SCZ":
            raw = safe_sum(row, DS115_SAPS + DS115_SANS)
            norm = (raw / DS115_MAX) * 100

            records.append({
                "participant_id": pid,
                "dataset": "ds115",
                "age": age,
                "sex": sex,
                "stage": "__patient__",
                "raw_score": raw,
                "norm_score": round(norm, 2),
            })

    return pd.DataFrame(records)


def load_ucla():
    participants_path = os.path.join(DIR_UCLA, "participants.tsv")
    sans_path = os.path.join(DIR_UCLA, "phenotype", "sans.tsv")
    saps_path = os.path.join(DIR_UCLA, "phenotype", "saps.tsv")

    for path in [participants_path, sans_path, saps_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    parts = pd.read_csv(participants_path, sep="\t")
    sans_df = pd.read_csv(sans_path, sep="\t")
    saps_df = pd.read_csv(saps_path, sep="\t")

    records = []

    for _, row in parts.iterrows():
        pid = row["participant_id"]
        diagnosis = str(row.get("diagnosis", "")).upper().strip()
        age = pd.to_numeric(row.get("age"), errors="coerce")
        sex = str(row.get("gender", "")).upper().strip()

        if diagnosis == "CONTROL":
            records.append({
                "participant_id": pid,
                "dataset": "ucla",
                "age": age,
                "sex": sex,
                "stage": "Healthy",
                "raw_score": np.nan,
                "norm_score": np.nan,
            })

        elif diagnosis == "SCHZ":
            sans_row = sans_df[sans_df["participant_id"] == pid]
            saps_row = saps_df[saps_df["participant_id"] == pid]

            sans_raw = (
                float(sans_row[UCLA_SANS_G].fillna(0).values.sum())
                if not sans_row.empty else 0.0
            )
            saps_raw = (
                float(saps_row[UCLA_SAPS_G].fillna(0).values.sum())
                if not saps_row.empty else 0.0
            )

            raw = sans_raw + saps_raw
            norm = (raw / UCLA_MAX) * 100

            records.append({
                "participant_id": pid,
                "dataset": "ucla",
                "age": age,
                "sex": sex,
                "stage": "__patient__",
                "raw_score": raw,
                "norm_score": round(norm, 2),
            })

    return pd.DataFrame(records)


def main():
    os.makedirs("Outputs", exist_ok=True)

    print("=" * 60)
    print("STEP 1 — 4-Class Clinical Stage Labeling")
    print("=" * 60)

    df115 = load_ds115()
    df_ucla = load_ucla()
    combined = pd.concat([df115, df_ucla], ignore_index=True)

    combined["nifti_path"] = combined.apply(
        lambda row: get_nifti_path(row["participant_id"], row["dataset"]),
        axis=1,
    )
    combined["mri_exists"] = combined["nifti_path"].apply(os.path.exists)

    available = combined[combined["mri_exists"] == True].copy()

    print("\nMRI-available subjects before staging:")
    print(available["stage"].value_counts())

    patient_scores = available.loc[
        available["stage"] == "__patient__", "norm_score"
    ].dropna()

    p33 = float(np.percentile(patient_scores, 33))
    p66 = float(np.percentile(patient_scores, 66))

    print(f"\nPatients used for staging: {len(patient_scores)}")
    print(f"p33 threshold: {p33:.2f}")
    print(f"p66 threshold: {p66:.2f}")

    mask = available["stage"] == "__patient__"
    available.loc[mask, "stage"] = available.loc[mask, "norm_score"].apply(
        lambda x: assign_stage(x, p33, p66)
    )

    print("\nAvailable subjects after staging:")
    print(available["stage"].value_counts())

    patient_df = available[available["stage"] != "Healthy"].copy()
    healthy_df = available[available["stage"] == "Healthy"].copy()

    patient_counts = patient_df["stage"].value_counts()
    target_healthy = int(patient_counts.median())

    if PREFER_UCLA_HEALTHY:
        ucla_healthy = healthy_df[healthy_df["dataset"] == "ucla"].copy()
        other_healthy = healthy_df[healthy_df["dataset"] != "ucla"].copy()

        if len(ucla_healthy) >= target_healthy:
            selected_healthy = ucla_healthy.sample(n=target_healthy, random_state=RANDOM_STATE)
        else:
            needed = target_healthy - len(ucla_healthy)
            selected_healthy = pd.concat([
                ucla_healthy,
                other_healthy.sample(n=min(needed, len(other_healthy)), random_state=RANDOM_STATE),
            ])
    else:
        selected_healthy = healthy_df.sample(n=target_healthy, random_state=RANDOM_STATE)

    final_df = pd.concat([selected_healthy, patient_df], ignore_index=True)
    final_df = final_df.drop(columns=["mri_exists"], errors="ignore")

    print("\nFinal 4-class MRI-available dataset:")
    print(final_df["stage"].value_counts())

    print("\nHealthy selected from:")
    print(final_df[final_df["stage"] == "Healthy"]["dataset"].value_counts())

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Total subjects: {len(final_df)}")

    print("\nClinical staging explanation:")
    print("Healthy  = MRI-available controls")
    print("Early    = lower SANS/SAPS symptom burden")
    print("Moderate = middle SANS/SAPS symptom burden")
    print("Chronic  = higher SANS/SAPS symptom burden")
    print("These are clinically informed proxy stages.")


if __name__ == "__main__":
    main()