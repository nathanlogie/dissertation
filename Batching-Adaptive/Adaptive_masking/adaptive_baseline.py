from typing import Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class AdaptiveMasking:

    def __init__(self, model: Union[LogisticRegression, RandomForestClassifier], bias_metric: Callable,
                 threshold: float, sensitive_attribute: str, batching: Callable[[pd.DataFrame, str, str, int], list],
                 retrain_full: bool = False, mask: int = -1) -> None:
        self.model = model
        self.bias_metric = bias_metric
        self.threshold = threshold
        self.mask = mask
        self.sensitive_attribute = sensitive_attribute
        self.is_masking = False
        self.retrain_full = retrain_full
        self.batching = batching

    def predict(self, x_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate_bias(self, y_true: Union[np.ndarray, pd.Series],
                      y_pred: np.ndarray) -> float:

        return self.bias_metric(y_true, y_pred, self.sensitive_attribute)

    def train(self, x_train: Union[np.ndarray, pd.DataFrame],
              y_train: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model using batched training data and adapt masking based on bias evaluation.
        """
        if isinstance(x_train, np.ndarray):
            x_train = pd.DataFrame(x_train)
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train, name='target')

        combined_training = pd.concat([x_train, y_train], axis=1)
        target_name = y_train.name if isinstance(y_train, pd.Series) else 'target'

        batches = self.batching(combined_training, target_name,
                                self.sensitive_attribute, 5)

        for batch_idx, batch in enumerate(batches):
            print(f"Batch {batch_idx + 1}/{len(batches)}")

            x_batch = batch.drop(columns=y_train.name)
            y_batch = batch[y_train.name]
            x_batch_train, x_batch_val, y_batch_train, y_batch_val = train_test_split(
                x_batch, y_batch, test_size=0.2, random_state=42
            )

            if self.is_masking:
                x_batch_train = self.masking(x_batch_train)

            self.model.fit(x_batch_train, y_batch_train)

            y_pred_val = self.model.predict(x_batch_val)
            bias_score = self.evaluate_bias(y_batch_val, y_pred_val)

            if bias_score > self.threshold:
                self.is_masking = True
            else:
                self.is_masking = False

    def masking(self, x_data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:

        if isinstance(x_data, pd.DataFrame):
            x_data[self.sensitive_attribute] = self.mask
        elif isinstance(x_data, np.ndarray):
            sensitive_idx = list(x_data.columns).index(self.sensitive_attribute)
            x_data[:, sensitive_idx] = self.mask
        else:
            raise ValueError("Unsupported data type for x_data. Use pandas DataFrame or numpy array.")

        return x_data
