"""
Evaluation Module for Customer Churn Prediction.
Calculates Precision, Recall, F1, Confusion Matrix, and generates comparison plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_single_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluates a single trained model on test data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "cm_array": cm.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist() if y_proba is not None else [],
    }
    return metrics


def plot_confusion_matrices(results: dict, output_path: str = "reports/confusion_matrices.png"):
    """
    Plots a 2x2 grid of confusion matrices for all evaluated models.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (name, res) in enumerate(results.items()):
        cm = np.array(res["cm_array"])
        tn, fp, fn, tp = cm.ravel()
        labels = [
            [f"True Neg (Stay)\n{tn}\n({tn/cm.sum():.1%})", f"False Pos (Churn)\n{fp}\n({fp/cm.sum():.1%})"],
            [f"False Neg (Stay)\n{fn}\n({fn/cm.sum():.1%})", f"True Pos (Churn)\n{tp}\n({tp/cm.sum():.1%})"],
        ]

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 11, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{name}\nPrecision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1_score']:.3f}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("Actual Label", fontsize=10)
        axes[idx].set_xticklabels(["No Churn (0)", "Churn (1)"])
        axes[idx].set_yticklabels(["No Churn (0)", "Churn (1)"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices plot to {output_path}")


def plot_roc_curves(results: dict, y_test, output_path: str = "reports/roc_curves.png"):
    """
    Plots comparative ROC-AUC curves for all models.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 7))

    for name, res in results.items():
        if res["y_proba"]:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison - Customer Churn Prediction", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC curves plot to {output_path}")


def plot_metrics_comparison(results: dict, output_path: str = "reports/model_comparison.png"):
    """
    Plots side-by-side bar chart of Precision, Recall, F1, Accuracy across all models.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    records = []
    for name, res in results.items():
        records.append(
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1 Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"],
            }
        )
    df_metrics = pd.DataFrame(records)

    df_melted = pd.melt(df_metrics, id_vars=["Model"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Model Performance Metrics Comparison", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved metrics comparison plot to {output_path}")


def plot_feature_importance(model, feature_names: list, model_name: str, output_path: str = "reports/feature_importance.png"):
    """
    Plots top 15 most important features for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f"Top 15 Feature Importances ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved feature importance plot to {output_path}")
