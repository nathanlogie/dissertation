from typing import Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class AdaptiveMasking:

    def __init__(self, model:Union[LogisticRegression, RandomForestClassifier], bias_metric:Callable, threshold:float,sensitive_attribute:str, mask:int=-1) -> None:
        self.model = model
        self.bias_metric = bias_metric
        self.threshold = threshold
        self.mask = mask
        self.sensitive_attribute = sensitive_attribute
        self.is_masking = False

    def predict(self, x_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate_bias(self, y_true: Union[np.ndarray, pd.Series],
                      y_pred: np.ndarray) -> float:

        return self.bias_metric(y_true, y_pred, self.sensitive_attribute)

    def train(self, x_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series],
              sensitive_attribute) -> None:
        sensitive_attribute = np.array(sensitive_attribute)

        for epoch in range(1, 15):
            print(f"Epoch {epoch}")

            if self.is_masking:
                x_train[self.sensitive_attribute] = self.mask
                print(f"Epoch {epoch}: Sensitive attribute masked.")

            self.model.fit(x_train, y_train)

            y_pred = self.model.predict(x_train)

            bias_score = self.bias_metric(y_train, y_pred, sensitive_attribute)
            print(f"Epoch {epoch}: Bias Score = {bias_score}")

            if bias_score > self.threshold:
                self.is_masking = True
            else:
                self.is_masking = False

            if self.is_masking:
                x_train[self.sensitive_attribute] = self.mask

    def masking(self, x_data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:

        if isinstance(x_data, pd.DataFrame):
            x_data[self.sensitive_attribute] = self.mask
        elif isinstance(x_data, np.ndarray):
            sensitive_idx = list(x_data.columns).index(self.sensitive_attribute)
            x_data[:, sensitive_idx] = self.mask
        else:
            raise ValueError("Unsupported data type for x_data. Use pandas DataFrame or numpy array.")

        return x_data