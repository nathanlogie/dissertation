from typing import Union, Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from bias_metrics import example_bias_metric
from batching_strats import batch_by_similarity

class AdaptiveBaseline:

    def __init__(self, model: Union[LogisticRegression, RandomForestClassifier], bias_metric: Callable,
                 threshold: float, sensitive_attribute: str, batching: Callable[[pd.DataFrame, str, str, int], list],
                 mask: int = 0) -> None:

        self.model = model
        self.bias_metric = bias_metric
        self.threshold = threshold
        self.mask = mask
        self.sensitive_attribute = sensitive_attribute
        self.is_masking = False
        self.batching = batching

    def predict(self, x_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate_bias(self, y_true: Union[np.ndarray, pd.Series],
                      y_pred: np.ndarray) -> float:
        """
        Evaluate bias using the provided bias metric.
        """
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
        target_name = y_train.name

        # Split into train, validation, and test sets
        train_data, test_data = train_test_split(combined_training, test_size=0.1, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2222, random_state=42)  # 0.2222 ensures 20% of total

        x_val = val_data.drop(columns=target_name)
        y_val = val_data[target_name]

        # Batch the remaining 70% training data
        batches = self.batching(train_data, target_name, self.sensitive_attribute, 5)

        for batch_idx, batch in enumerate(batches):
            print(f"Batch {batch_idx + 1}/{len(batches)}")

            x_batch = batch.drop(columns=target_name)
            y_batch = batch[target_name]

            # Apply masking if necessary
            if self.is_masking:
                x_batch = self.masking(x_batch)

            self.model.fit(x_batch, y_batch)

            y_pred_val = self.model.predict(x_val)
            bias_score = self.evaluate_bias(y_val, y_pred_val)
            print(f"Bias Score for Batch {batch_idx + 1}: {bias_score}")

            self.is_masking = bias_score > self.threshold

    def masking(self, x_data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(x_data, pd.DataFrame):
            x_data[self.sensitive_attribute] = self.mask
        else:
            raise ValueError("Unsupported data type for x_data. Use pandas DataFrame.")
        return x_data

def main():
    print("Adaptive Baseline")

    adaptiveMasking = AdaptiveBaseline(model=LogisticRegression(), bias_metric=example_bias_metric, threshold=0.1, sensitive_attribute="race", batching=batch_by_similarity, mask=1)