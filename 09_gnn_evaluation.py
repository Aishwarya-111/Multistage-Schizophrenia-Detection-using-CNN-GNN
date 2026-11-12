import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PREDS_CSV = "Outputs/gnn_results/gnn_predictions.csv"
OUT_DIR = "Outputs/gnn_results"
CLASS_ORDER = ["Healthy", "Early", "Moderate", "Chronic"]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(PREDS_CSV):
        print(f"❌ Error: Predictions file {PREDS_CSV} not found. Please run 06_gnn_classifier.py first.")
        return

    print("Loading GNN predictions for evaluation...")
    df = pd.read_csv(PREDS_CSV)
    
    y_true = df["True_Stage"].values
    y_pred = df["Predicted_Stage"].values
    
    # 1. Print Classification Report
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    print("\n[+] GNN Classification Report:")
    print(report)
    
    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_ORDER, 
                yticklabels=CLASS_ORDER,
                annot_kws={"size": 14})
    plt.title("GAT 4-Class Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Stage", fontsize=12)
    plt.ylabel("True Stage", fontsize=12)
    cm_path = os.path.join(OUT_DIR, "gnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot F1-Scores per Class
    f1_vals = f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    plt.figure(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    plt.bar(CLASS_ORDER, f1_vals, color=colors, alpha=0.8)
    plt.ylim(0, 1.1)
    plt.title("GNN F1-Scores per Disease Stage", fontsize=14, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=12)
    plt.xlabel("Schizophrenia Stage", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for i, v in enumerate(f1_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    f1_path = os.path.join(OUT_DIR, "gnn_f1_scores.png")
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("[DONE] GENERATED GNN EVALUATION MATRICES")
    print("="*50)
    print(f"[+] Confusion Matrix : {cm_path}")
    print(f"[+] F1-Score Chart   : {f1_path}")

if __name__ == "__main__":
    main()
