import numpy as np

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    return float(np.mean(np.abs(y_pred - y_true)))
    
    