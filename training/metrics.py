import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    rng = y_true.max() - y_true.min()
    nrmse = rmse / rng if rng > 0 else float("nan")
    return {"R2": float(r2), "MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "NRMSE": float(nrmse)}
