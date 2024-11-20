from typing import Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class AdaptiveMasking:

    def __init__(self, model:Union[LogisticRegression, RandomForestClassifier], bias_metric:Callable, threshold:int, mask:int=0) -> None:
        self.model = model
        self.bias_metric = bias_metric
        self.threshold = threshold
        self.mask = mask
        self.is_masking = False

    def predict(self, x_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate_bias(self, y_true: Union[np.ndarray, pd.Series],
                      y_pred: np.ndarray,
                      sensitive_attribute: Union[np.ndarray, pd.Series]) -> float:
        return self.bias_metric(y_true, y_pred, sensitive_attribute)

    def train(self, x_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.DataFrame], sensitive_attribute):
        pass

    def masking(self, x_test: Union[np.ndarray, pd.DataFrame],
                y_test: Union[np.ndarray, pd.Series],
                sensitive_attribute: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.DataFrame]:        pass