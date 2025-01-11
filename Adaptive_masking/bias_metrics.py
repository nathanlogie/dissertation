import numpy as np


def example_bias_metric(y_true: np.ndarray, y_pred_shadow: np.ndarray, sensitive_attr: np.ndarray) -> float:
    group_0_mean = y_pred_shadow[sensitive_attr == 0].mean()
    group_1_mean = y_pred_shadow[sensitive_attr == 1].mean()
    return abs(group_0_mean - group_1_mean)