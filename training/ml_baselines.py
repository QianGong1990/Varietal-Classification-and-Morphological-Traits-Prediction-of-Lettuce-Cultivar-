"""Traditional ML baselines on hand-crafted color/vegetation-index features
extracted from the ORIGINAL (non-augmented) removed_background images."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from training.data import REGRESSION_TARGETS, DATASET_ROOT, load_split_df, load_ground_truth
from training.metrics import regression_metrics
from training.plotting import ml_vs_cnn_comparison_plot
from training.seed import set_seed

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEG_DIR = DATASET_ROOT / "removed_background_224"

SEED = 42


def extract_features(image_path: Path) -> np.ndarray:
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    non_black = (r + g + b) > 10
    total = r.size
    lpr = non_black.sum() / total

    if non_black.sum() == 0:
        r_fg, g_fg, b_fg = r.flatten(), g.flatten(), b.flatten()
    else:
        r_fg, g_fg, b_fg = r[non_black], g[non_black], b[non_black]

    mean_r, mean_g, mean_b = r_fg.mean(), g_fg.mean(), b_fg.mean()
    std_r, std_g, std_b = r_fg.std(), g_fg.std(), b_fg.std()

    exg = (2 * g_fg - r_fg - b_fg).mean()
    exr = (1.4 * r_fg - g_fg).mean()
    denom = (g_fg + r_fg - b_fg)
    vari = np.where(denom != 0, (g_fg - r_fg) / np.where(denom == 0, 1, denom), 0).mean()

    return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b, lpr, exg, exr, vari])


FEATURE_NAMES = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "LPR", "ExG", "ExR", "VARI"]


def build_feature_table() -> pd.DataFrame:
    split_df = load_split_df()
    gt = load_ground_truth()
    rows = []
    for _, r in split_df.iterrows():
        if r["split"] not in ("train", "test"):
            continue
        image_id = int(r["id"])
        gt_row = gt[f"Image{image_id}"]
        feats = extract_features(SEG_DIR / r["filename"])
        row = {"filename": r["filename"], "variety": r["variety"], "split": r["split"]}
        row.update(dict(zip(FEATURE_NAMES, feats)))
        for t in REGRESSION_TARGETS:
            row[t] = gt_row[t]
        rows.append(row)
    return pd.DataFrame(rows)


def make_models():
    return {
        "LR": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0),
    }


def main():
    set_seed(SEED)
    out_dir = RESULTS_DIR / "ml_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_table()
    df.to_csv(out_dir / "features.csv", index=False)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train_raw = train_df[FEATURE_NAMES].values
    X_test_raw = test_df[FEATURE_NAMES].values

    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=0.95, random_state=SEED).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA: {X_train_raw.shape[1]} features -> {X_train_pca.shape[1]} components (95% variance)")

    results = []
    for trait in REGRESSION_TARGETS:
        y_train = train_df[trait].values
        y_test = test_df[trait].values

        for name, model_fn in make_models().items():
            model = model_fn
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            m = regression_metrics(y_test, pred)
            results.append({"model": name, "trait": trait, **m})

        for name, model_fn in make_models().items():
            model = model_fn
            model.fit(X_train_pca, y_train)
            pred = model.predict(X_test_pca)
            m = regression_metrics(y_test, pred)
            results.append({"model": f"PCA+{name}", "trait": trait, **m})

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "ml_results.csv", index=False)
    print(results_df.to_string(index=False))

    with open(out_dir / "pca_info.json", "w") as f:
        json.dump({
            "n_features_in": int(X_train_raw.shape[1]),
            "n_components": int(X_train_pca.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }, f, indent=2)

    return results_df


def build_ml_vs_cnn_plot():
    """Combines ml_results.csv with the best CNN result per trait/model from
    results/regression/all_results.csv into results/figures/ml_comparison/."""
    ml_path = RESULTS_DIR / "ml_baselines" / "ml_results.csv"
    cnn_path = RESULTS_DIR / "regression" / "all_results.csv"
    if not ml_path.exists() or not cnn_path.exists():
        print("Skipping ml-vs-cnn plot: missing ml_results.csv or all_results.csv")
        return

    ml_df = pd.read_csv(ml_path)
    ml_df["source"] = "ML"

    cnn_df = pd.read_csv(cnn_path)
    # Keep only the best input_type per (architecture, trait) so the plot
    # stays readable: 5 CNN bars + 6 ML bars per subplot, not 15 + 6.
    cnn_df = cnn_df.loc[cnn_df.groupby(["model", "trait"])["R2"].idxmax()]
    cnn_df["model_label"] = cnn_df["model"] + " (" + cnn_df["input_type"] + ")"
    cnn_df = cnn_df.rename(columns={"model": "orig_model"})
    cnn_df["model"] = cnn_df["model_label"]
    cnn_df["source"] = "CNN"

    combined = pd.concat([
        ml_df[["model", "trait", "R2", "source"]],
        cnn_df[["model", "trait", "R2", "source"]],
    ], ignore_index=True)

    ml_vs_cnn_comparison_plot(combined, RESULTS_DIR / "figures" / "ml_comparison")


if __name__ == "__main__":
    main()
    build_ml_vs_cnn_plot()
