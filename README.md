# Varietal Classification and Morphological Traits Prediction of Lettuce Cultivars

Dataset, training pipeline, data split, model architectures and best trained models
for (1) classifying four lettuce cultivars and (2) predicting morphological traits
(leaf area, diameter, fresh and dry shoot weight) from top-view RGB images.

## Original dataset
The images and ground-truth measurements are **secondary data**, publicly available in the
4TU.ResearchData repository:

> Hemming, S., de Zwart, H.F., Elings, A., Bijlaard, M., van Marrewijk, B., & Petropoulou, A.
> (2021). *3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images* (Version 1)
> [Data set]. 4TU.ResearchData. https://doi.org/10.4121/15023088.v1

Available at https://data.4tu.nl/articles/_/15023088/1. The data were generated for the 3rd
International Autonomous Greenhouse Challenge at the Wageningen University & Research
Greenhouse Horticulture Business Unit (Bleiswijk, The Netherlands, 2021). They are
redistributed here under the original
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. Please cite the original dataset when using it.

Differences from the original release:
- `RGBImages/` and `GroundTruth_All_388_Images.json` are taken from the original dataset.
- The 388 depth images of the original release are not included.
- `GroundTruth_All_388_Images.csv`, `binary_masks/` and `split/` were created in this study.

```bibtex
@misc{hemming2021lettuce,
  author    = {Hemming, Silke and de Zwart, H. F. and Elings, Anne and Bijlaard, Monique
               and van Marrewijk, Bart and Petropoulou, Anna},
  title     = {3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images},
  year      = {2021},
  publisher = {4TU.ResearchData},
  version   = {1},
  doi       = {10.4121/15023088.v1},
  url       = {https://data.4tu.nl/articles/_/15023088/1}
}
```

## Repository contents
| Path | Contents |
|---|---|
| `dataset/RGBImages/` | 388 raw top-view RGB images (`RGB_<id>.png`, 1920×1080), from the original dataset |
| `dataset/binary_masks/` | Full-resolution plant/background masks for each image |
| `dataset/GroundTruth_All_388_Images.json` | Ground-truth file from the original dataset (read by the pipeline) |
| `dataset/GroundTruth_All_388_Images.csv` | Same measurements as a table, one row per image |
| `dataset/split/dataset_split.xlsx` / `.csv` | Train/valid/test assignment of every image |
| `remove_background.py`, `resize_dataset.py`, `run_augmentation.py`, `preview_augmentation.py` | Preprocessing |
| `training/` | Training and evaluation pipeline (PyTorch) |
| `training/models.py` | The five architectures used |
| `results/normalization_params.json` | Pixel and target normalization used in training |
| `results/*/model_weights/`, `results/onnx/` | Best trained models (Git LFS) |

Run `git lfs install` before cloning so the model files are downloaded.

## Dataset
Ground-truth columns (CSV): `Image_id`, `RGB_Image`, `Variety`, `FreshWeightShoot_g`,
`DryWeightShoot_g`, `Height_cm`, `Diameter_cm`, `LeafArea_cm2`.

## Data split
Stratified by cultivar, 70 / 15 / 15 %:

| Variety | Train | Valid | Test | Total |
|---|---|---|---|---|
| Aphylion | 64 | 14 | 14 | 92 |
| Lugano | 67 | 14 | 15 | 96 |
| Salanova | 72 | 15 | 15 | 102 |
| Satine | 69 | 15 | 14 | 98 |
| **Total** | 272 | 58 | 58 | 388 |

## Models
`training/models.py` defines SimpleCNN (trained from scratch, 224 px), MobileNetV2 and
EfficientNetB0 (ImageNet-pretrained, 224 px), Xception and InceptionResNetV2
(ImageNet-pretrained, 299 px). Each was trained on three inputs: raw RGB,
background-removed (segmented) RGB and binary mask.

## Best models
| Task | Trait | Model | Input | Test result | File |
|---|---|---|---|---|---|
| Regression | Diameter | InceptionResNetV2 | RGB | R² 0.892, RMSE 2.167 cm | `results/regression/model_weights/InceptionResNetV2_rgb_best.pth` |
| Regression | Fresh weight | InceptionResNetV2 | RGB | R² 0.935, RMSE 30.647 g | `results/regression/model_weights/InceptionResNetV2_rgb_best.pth` |
| Regression | Leaf area | InceptionResNetV2 | segmented | R² 0.898, RMSE 520.734 cm² | `results/regression/model_weights/InceptionResNetV2_segmented_best.pth` |
| Regression | Dry weight | SimpleCNN | RGB | R² 0.958, RMSE 1.020 g | `results/regression/model_weights/SimpleCNN_rgb_best.pth` |
| Classification | Variety | SimpleCNN | segmented | 98.28 % accuracy | `results/classification/model_weights/SimpleCNN_segmented_best.pth` |

`results/onnx/simplecnn_rgb_regression.onnx` is an ONNX export of the SimpleCNN RGB
regression model.

## Reproducing the pipeline
```bash
pip install -r requirements.txt
python remove_background.py        # dataset/RGBImages -> dataset/removed_backgound
python resize_dataset.py           # 1080 px crop centred on the plant -> 224 and 299 px
python run_augmentation.py         # augments the training split -> dataset/augmented
python -m training.run_all --epochs 50 --seed 42   # 15 regression runs
python -m training.run_classification_all          # cultivar classification
python -m training.kfold                           # 5-fold cross-validation
python -m training.ml_baselines                    # classical ML baselines
```
`training/submit_*.sh` are the SLURM scripts used on an H100 GPU node; edit the
e-mail and conda environment before use.

## Using a best model
```python
import json, numpy as np, torch
from PIL import Image
from training.models import build_model

norm = json.load(open("results/normalization_params.json"))
model = build_model("SimpleCNN", out_dim=4)
model.load_state_dict(torch.load("results/regression/model_weights/SimpleCNN_rgb_best.pth",
                                 map_location="cpu", weights_only=True))
model.eval()

# 1080x1080 crop centred on the plant (as in resize_dataset.py), then resize
img = Image.open("dataset/RGBImages/RGB_1.png").convert("RGB")
ys, xs = np.nonzero(np.asarray(Image.open("dataset/binary_masks/RGB_1.png").convert("L")))
w, h = img.size
x0 = min(max(int(xs.mean()) - 540, 0), w - 1080)
y0 = min(max(int(ys.mean()) - 540, 0), h - 1080)
img = img.crop((x0, y0, x0 + 1080, y0 + 1080))
x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
p = norm["pixel_norm"]["rgb_224"]   # SimpleCNN; pretrained models use ImageNet mean/std
x = (torch.from_numpy(x).permute(2, 0, 1) - torch.tensor(p["mean"])[:, None, None]) \
    / torch.tensor(p["std"])[:, None, None]

with torch.no_grad():
    out = model(x[None])[0]
for i, t in enumerate(["LeafArea", "Diameter", "FreshWeightShoot", "DryWeightShoot"]):
    print(t, float(out[i]) * norm["targets"][t]["std"] + norm["targets"][t]["mean"])
```
Regression models output the four traits in the order above, standardized with the
training-set mean/std in `normalization_params.json`. The classifier outputs logits
for `["Aphylion", "Lugano", "Salanova", "Satine"]`.
