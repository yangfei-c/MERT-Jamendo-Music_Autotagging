import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

# =============== Settings =====================
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FONT_SCALE = 1.4
COLOR_PALETTE = sns.color_palette("bright", 10)
RESULTS_SUBDIR_PATTERN = "*"
TEST_RESULTS_FILENAME = "results.pt"
SUMMARY_CSV_FILENAME = "all_tasks_summary.csv"
COMBINED_ROC_FILENAME = "all_tasks_roc.png"
COMBINED_PR_FILENAME = "all_tasks_pr.png"
INDIVIDUAL_ROC_SUFFIX = "_roc.png"
INDIVIDUAL_PR_SUFFIX = "_pr.png"
DPI = 300
MODEL_IDENTIFIERS = ["Linear", "GCN"]  # Can extend with more models


def load_pt(file_path):
    print(f"Loading: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    predictions = data['predictions'].numpy()
    targets = data['targets'].numpy()
    return predictions, targets


def plot_individual_roc_and_pr(
        predictions: np.ndarray,
        targets: np.ndarray,
        task_name: str,
        save_dir: str
) -> None:
    """Plot and save individual ROC and PR curves for a task."""

    # ROC Curve (Micro-averaged)
    fpr, tpr, _ = roc_curve(targets.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curve - {task_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}{INDIVIDUAL_ROC_SUFFIX}"), dpi=DPI)
    plt.close()

    # PR Curve (Micro-averaged)
    precision, recall, _ = precision_recall_curve(targets.ravel(), predictions.ravel())
    pr_auc = average_precision_score(targets, predictions, average="micro")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve - {task_name}')
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}{INDIVIDUAL_PR_SUFFIX}"), dpi=DPI)
    plt.close()


def compute_metrics(
        predictions: np.ndarray,
        targets: np.ndarray
) -> dict[str, float]:
    """Compute detailed metrics: Macro/Micro AUC and AP."""
    return {
        "macro_auc": roc_auc_score(targets, predictions, average='macro'),
        "macro_ap": average_precision_score(targets, predictions, average='macro'),
        "micro_auc": roc_auc_score(targets, predictions, average='micro'),
        "micro_ap": average_precision_score(targets, predictions, average='micro')
    }


def plot_combined_roc_and_pr(
        results_list: list[dict],
        save_root: str
) -> None:
    """Plot combined ROC and PR curves for all tasks."""

    # Combined ROC
    plt.figure(figsize=(10, 8))
    for idx, item in enumerate(results_list):
        task = item['task']
        fpr, tpr, _ = roc_curve(item['targets'].ravel(), item['preds'].ravel())
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], lw=2, alpha=0.8,
                 label=f'{task} (AUC={roc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Performance Comparison (ROC Curves)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, COMBINED_ROC_FILENAME), dpi=DPI)
    plt.close()

    # Combined PR
    plt.figure(figsize=(10, 8))
    for idx, item in enumerate(results_list):
        task = item['task']
        precision, recall, _ = precision_recall_curve(item['targets'].ravel(), item['preds'].ravel())
        pr_val = average_precision_score(item['targets'], item['preds'], average="micro")
        plt.plot(recall, precision, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], lw=2, alpha=0.8,
                 label=f'{task} (mAP={pr_val:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Performance Comparison (Precision-Recall Curves)')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, COMBINED_PR_FILENAME), dpi=DPI)
    plt.close()


def clean_task_name(folder_name: str) -> str:
    """Clean task name by removing model suffixes and prefixes."""
    cleaned = folder_name
    for model in MODEL_IDENTIFIERS:
        cleaned = cleaned.replace(f"_{model}", "")
    cleaned = cleaned.replace("MusicTAG_", "")
    return cleaned


def identify_model(folder_name: str) -> str:
    """Identify the model type from folder name."""
    for model in MODEL_IDENTIFIERS:
        if model in folder_name:
            return model
    return "Unknown"


def main(results_root: str = None) -> None:
    """Main function to process results, with optional root directory."""
    if results_root is None:
        # Default to ../results relative to script location
        results_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    # Set global plotting style
    plt.style.use(PLOT_STYLE)
    sns.set_context("paper", font_scale=FONT_SCALE)

    # Find all test results files
    search_pattern = os.path.join(results_root, RESULTS_SUBDIR_PATTERN, TEST_RESULTS_FILENAME)
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"❌ No result files found! Check path: {results_root}")
        return

    print(f"🔍 Found {len(result_files)} task results, starting processing...\n")

    summary_data = []
    results_for_plots = []

    for file_path in result_files:
        folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(folder_path)

        task_name = clean_task_name(folder_name)
        model_type = identify_model(folder_name)

        # Load data
        preds, targets = load_pt(file_path)

        # Plot individual curves
        print(f" [Plotting] {task_name} ({model_type}) -> {folder_path}")
        plot_individual_roc_and_pr(preds, targets, task_name, folder_path)

        # Compute metrics
        metrics = compute_metrics(preds, targets)
        metrics["task"] = task_name
        metrics["model"] = model_type
        summary_data.append(metrics)

        # Collect for combined plots (optional filter: e.g., skip large tasks)
        # if "Autotagging" in task_name or "All" in task_name: continue
        results_for_plots.append({
            "task": f"{task_name} ({model_type})",
            "preds": preds,
            "targets": targets
        })

    # Generate summary CSV
    df = pd.DataFrame(summary_data)
    column_order = ["task", "model", "macro_auc", "macro_ap", "micro_auc", "micro_ap"]
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order] if column_order else df

    summary_csv_path = os.path.join(results_root, SUMMARY_CSV_FILENAME)
    df.to_csv(summary_csv_path, index=False)

    # Generate combined plots if data available
    if results_for_plots:
        print("\n [Summary] Generating combined plots...")
        plot_combined_roc_and_pr(results_for_plots, results_root)

    print("\n" + "=" * 50)
    print("✅ Processing complete!")
    print(f"📊 Summary table: {summary_csv_path}")
    print(f"📈 Combined plots saved in: {results_root}")
    print("=" * 50)
    print(df)


if __name__ == "__main__":
    main()