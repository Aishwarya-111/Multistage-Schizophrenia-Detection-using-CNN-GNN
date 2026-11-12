import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score

INPUT_CSV = "Outputs/npy_dataset/fmri_features_4class.csv"
OUT_DIR = "Outputs/cnn_results"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading CNN outputs from database...")
    df = pd.read_csv(INPUT_CSV)
    
    # We only evaluate on the unseen TEST set to ensure genuine results
    test_df = df[df["split"] == "test"]
    
    y_true = test_df["binary_label"].values
    y_prob = test_df["prob_schizophrenia"].values
    y_pred = np.where(y_prob > 0.5, 'Schizophrenia', 'Healthy')
    
    # Map for ROC curve
    y_true_binary = np.where(y_true == 'Schizophrenia', 1, 0)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[+] CNN Test Accuracy: {acc * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Healthy", "Schizophrenia"]))
    
    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=["Healthy", "Schizophrenia"], 
                yticklabels=["Healthy", "Schizophrenia"],
                annot_kws={"size": 16})
    plt.title("CNN Binary Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("CNN Predicted", fontsize=12)
    plt.ylabel("True Diagnosis", fontsize=12)
    cm_path = os.path.join(OUT_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('CNN ROC Curve (Healthy vs. Schizophrenia)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    roc_path = os.path.join(OUT_DIR, "cnn_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot Probability Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=test_df, x="prob_schizophrenia", hue="binary_label", 
                 kde=True, bins=15, palette=["green", "red"])
    plt.title("CNN Probability Distribution by Class", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Probability of Schizophrenia", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(["Schizophrenia (1)", "Healthy (0)"])
    plt.grid(True, linestyle='--', alpha=0.7)
    dist_path = os.path.join(OUT_DIR, "cnn_probability_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Plot F1-Scores
    classes = ["Healthy", "Schizophrenia"]
    f1_vals = f1_score(y_true, y_pred, labels=classes, average=None)
    plt.figure(figsize=(6, 5))
    plt.bar(classes, f1_vals, color=['#55A868', '#C44E52'], alpha=0.8)
    plt.ylim(0, 1.1)
    plt.title("CNN F1-Scores (Healthy vs. Schizophrenia)", fontsize=14, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(f1_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    f1_path = os.path.join(OUT_DIR, "cnn_f1_scores.png")
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("[DONE] GENERATED CNN EVALUATION MATRICES")
    print("="*50)
    print(f"[+] Confusion Matrix : {cm_path}")
    print(f"[+] ROC AUC Curve    : {roc_path}")
    print(f"[+] Prob Distribution: {dist_path}")
    print(f"[+] F1-Score Chart   : {f1_path}")

if __name__ == "__main__":
    main()
