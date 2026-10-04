# Varietal Classification and Morphological Traits Prediction of Lettuce Cultivars

Raw image dataset, ground-truth measurements and the best trained models for
classifying four lettuce cultivars and predicting their morphological traits.

## Dataset
- `dataset/RGBImages/`: 388 raw top-view RGB images (`RGB_<id>.png`) of lettuce grown
  in a controlled greenhouse, destructively measured at 7-day intervals.
- `dataset/GroundTruth_All_388_Images.csv`: one row per image.

| Column | Unit |
|---|---|
| Image_id, RGB_Image | – |
| Variety (Aphylion, Lugano, Salanova, Satine) | – |
| FreshWeightShoot_g, DryWeightShoot_g | gram/plant |
| Height_cm, Diameter_cm | cm |
| LeafArea_cm2 | cm² |

Images per variety: Aphylion 92, Lugano 96, Salanova 102, Satine 98.

## Best models (PyTorch `.pth`, stored with Git LFS)
| Task | Trait | Model | Input | Test result | File |
|---|---|---|---|---|---|
| Regression | Diameter | InceptionResNetV2 | RGB | R² 0.892, RMSE 2.167 | `results/regression/model_weights/InceptionResNetV2_rgb_best.pth` |
| Regression | FreshWeightShoot | InceptionResNetV2 | RGB | R² 0.935, RMSE 30.647 | `results/regression/model_weights/InceptionResNetV2_rgb_best.pth` |
| Regression | LeafArea | InceptionResNetV2 | segmented | R² 0.898, RMSE 520.734 | `results/regression/model_weights/InceptionResNetV2_segmented_best.pth` |
| Regression | DryWeightShoot | SimpleCNN | RGB | R² 0.958, RMSE 1.020 | `results/regression/model_weights/SimpleCNN_rgb_best.pth` |
| Classification | Variety | SimpleCNN | segmented | 98.28 % accuracy | `results/classification/model_weights/SimpleCNN_segmented_best.pth` |

`results/onnx/simplecnn_rgb_regression.onnx` is an ONNX export of the SimpleCNN RGB
regression model.

Run `git lfs install` before cloning so the model files are downloaded.
