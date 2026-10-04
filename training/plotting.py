from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from training.data import TRAIT_DISPLAY

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 13

CULTIVAR_COLORS = {
    "Salanova": "tab:blue",
    "Lugano": "tab:orange",
    "Satine": "tab:green",
    "Aphylion": "tab:red",
}


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(y_true, y_pred, cultivars, trait: str, unit: str, model: str, input_type: str,
                  r2: float, rmse: float, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for cultivar, color in CULTIVAR_COLORS.items():
        mask = np.array(cultivars) == cultivar
        if mask.sum() == 0:
            continue
        ax.scatter(np.array(y_true)[mask], np.array(y_pred)[mask], color=color, label=cultivar,
                   alpha=0.75, edgecolor="k", linewidth=0.3, s=45)

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.5, zorder=0)

    trait_label = TRAIT_DISPLAY.get(trait, trait)
    ax.set_xlabel(f"Actual {trait_label} ({unit})", fontsize=14)
    ax.set_ylabel(f"Predicted {trait_label} ({unit})", fontsize=14)
    ax.set_title(f"{model} - {input_type} - {trait_label}", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc="lower right")
    ax.text(0.03, 0.97, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
            fontsize=13, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    _save(fig, out_dir, f"{model}_{input_type}_{trait}")


def confusion_matrix_plot(cm: np.ndarray, class_names: list, out_dir: Path,
                           model: str = "", input_type: str = ""):
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax, cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted Variety", fontsize=13)
    ax.set_ylabel("True Variety", fontsize=13)
    title_suffix = f" ({model}, {input_type})" if model else ""
    ax.set_title(f"Confusion Matrix - Variety Classification{title_suffix}", fontsize=14)
    ax.tick_params(labelsize=13)
    name = f"confusion_matrix_{model}_{input_type}" if model else "confusion_matrix"
    _save(fig, out_dir, name)


def training_curve_plot(history_df, model: str, input_type: str, best_epoch: int,
                         early_stop_epoch: int | None, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="tab:blue")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="tab:orange")

    if early_stop_epoch is not None:
        ax.axvline(early_stop_epoch, linestyle="--", color="gray", label="Early Stop")

    best_row = history_df[history_df["epoch"] == best_epoch]
    if len(best_row):
        ax.scatter(best_row["epoch"], best_row["val_loss"], marker="*", s=250, color="red",
                   zorder=5, label="_nolegend_")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss (MSE)", fontsize=14)
    ax.set_title(f"{model} - {input_type} Training Curve", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)
    _save(fig, out_dir, f"{model}_{input_type}_loss_curve")


def kfold_comparison_plot(summary_df, out_dir: Path):
    traits = summary_df["trait"].unique().tolist()
    archs = summary_df["model"].unique().tolist()
    x = np.arange(len(archs))
    width = 0.8 / len(traits)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, trait in enumerate(traits):
        sub = summary_df[summary_df["trait"] == trait].set_index("model").reindex(archs)
        bars = ax.bar(x + i * width, sub["R2_mean"], width, yerr=sub["R2_std"], capsize=4,
                      label=TRAIT_DISPLAY.get(trait, trait))
        for b, v in zip(bars, sub["R2_mean"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylim(0, 1.15)

    ax.set_xticks(x + width * (len(traits) - 1) / 2)
    ax.set_xticklabels(archs, fontsize=13)
    ax.set_xlabel("Architecture", fontsize=14)
    ax.set_ylabel("R² (mean ± std)", fontsize=14)
    ax.set_title("5-Fold Cross Validation Results", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)
    _save(fig, out_dir, "kfold_comparison")


def ml_vs_cnn_comparison_plot(combined_df, out_dir: Path):
    traits = ["LeafArea", "Diameter", "FreshWeightShoot", "DryWeightShoot"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cnn_cmap = plt.cm.Blues
    ml_cmap = plt.cm.Oranges

    for ax, trait in zip(axes, traits):
        sub = combined_df[combined_df["trait"] == trait].sort_values("R2", ascending=False)
        colors = []
        n_cnn = (sub["source"] == "CNN").sum()
        n_ml = (sub["source"] == "ML").sum()
        ci = 0
        mi = 0
        for s in sub["source"]:
            if s == "CNN":
                ci += 1
                colors.append(cnn_cmap(0.4 + 0.5 * ci / max(n_cnn, 1)))
            else:
                mi += 1
                colors.append(ml_cmap(0.4 + 0.5 * mi / max(n_ml, 1)))

        bars = ax.bar(sub["model"], sub["R2"], color=colors)
        for b, v in zip(bars, sub["R2"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(TRAIT_DISPLAY.get(trait, trait), fontsize=14)
        ax.set_ylabel("R²", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.tick_params(axis="y", labelsize=13)

    fig.suptitle("CNN vs Traditional ML Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "ml_vs_cnn_comparison")
