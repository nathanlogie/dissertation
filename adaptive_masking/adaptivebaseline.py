from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AdaptiveBaseline:

    def __init__(self, model: Union[LogisticRegression, RandomForestClassifier], bias_metric: Callable,
                 threshold: float, sensitive_attribute: str, batching: Callable[[pd.DataFrame, str, str, int], list],
                 mask: int = 0, num_batches: int = 10) -> None:

        """
            Parameters:
            - model: Classifier model (e.g., LogisticRegression, RandomForestClassifier)
            - bias_metric: Function to compute a bias metric given true and predicted values.
            - threshold: Threshold for determining when to apply masking.
            - sensitive_attribute: Name of the sensitive attribute in the dataset.
            - batching: Function that splits the dataset into batches for training.
              Expects input as (dataframe, target column, sensitive attribute, number of batches).
            - mask: Value to use for masking sensitive attributes.
            - batch_number: Number of batches to split training data into.
            """

        self.model = make_pipeline(StandardScaler(), model)
        self.bias_metric = bias_metric
        self.threshold = threshold
        self.mask = mask
        self.sensitive_attribute = sensitive_attribute
        self.is_masking = False
        self.batching = batching
        self.num_batches = num_batches

    def predict(self, x_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate_bias(self, y_true, y_pred, sensitive_attr):
        bias_score = self.bias_metric(y_true, y_pred, sensitive_attr)
        if np.isnan(bias_score):
            print("Warning: Bias metric returned NaN.")
            return 0
        return bias_score

    def train(self, x_train: Union[np.ndarray, pd.DataFrame],
              y_train: Union[np.ndarray, pd.Series],
              x_val: pd.DataFrame, y_val: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
              show_plots: bool) -> None:

        """
           Train the model adaptively, applying masking if bias exceeds threshold.

           Parameters:
           - x_train, y_train: Training features and labels.
           - x_val, y_val: Validation features and labels for bias evaluation.
           - x_test, y_test: Test set for final evaluation.
           - show_plots: Display bias score plots if True.
           """

        if isinstance(x_train, np.ndarray):
            x_train = pd.DataFrame(x_train)
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train, name='target')

        self.model.fit(x_train, y_train)

        combined_training = pd.concat([x_train, y_train], axis=1)
        target_name = y_train.name

        batches = self.batching(combined_training, target_name, self.sensitive_attribute, self.num_batches)

        train_sizes = []
        val_bias_scores = []
        test_bias_scores = []

        x_cumulative_training = pd.DataFrame()
        y_cumulative_training = pd.Series(dtype=y_train.dtype)

        for batch_idx, batch in enumerate(batches):
            # print(f"Batch {batch_idx + 1}/{len(batches)}")

            x_batch = batch.drop(columns=target_name)
            y_batch = batch[target_name]

            # print(f"Batch {batch_idx + 1} size: {len(x_batch)}")
            # print(f"Sensitive attribute distribution:\n{x_batch[self.sensitive_attribute].value_counts()}")

            if x_batch.empty:
                print(f"Skipping empty batch {batch_idx + 1}")
                continue

            # Apply masking if necessary
            if self.is_masking:
                x_batch = self.masking(x_batch)

            x_cumulative_training = pd.concat([x_cumulative_training, x_batch], ignore_index=True)
            y_cumulative_training = pd.concat([y_cumulative_training, y_batch], ignore_index=True)

            # Train the model
            self.model.fit(x_cumulative_training, y_cumulative_training)

            # Update training size
            current_train_size = len(x_cumulative_training)
            train_sizes.append(current_train_size)

            # Evaluate bias on the validation set
            y_pred_val = self.model.predict(x_val)
            val_bias_score = self.evaluate_bias(
                y_val,
                y_pred_val,
                x_val[self.sensitive_attribute]
            )

            if np.isnan(val_bias_score):
                print(f"Skipping Batch {batch_idx + 1} due to NaN bias score")
                continue

            val_bias_scores.append(val_bias_score)

            # Evaluate bias on the test set
            y_pred_test = self.model.predict(x_test)
            test_bias_score = self.evaluate_bias(y_test, y_pred_test, x_test[self.sensitive_attribute])
            test_bias_scores.append(test_bias_score)

            # print(f"Validation Bias Score for Batch {batch_idx + 1}: {val_bias_score}")
            # print(f"Test Bias Score for Batch {batch_idx + 1}: {test_bias_score}")

            self.is_masking = (val_bias_score > self.threshold) if val_bias_score else False

        if show_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(train_sizes, val_bias_scores, marker='o', label='Validation Bias Score')
            plt.plot(train_sizes, test_bias_scores, marker='x', label='Test Bias Score')
            plt.xlabel('Number of Items in Training Set')
            plt.ylabel('Bias Score')
            plt.title(f'Bias Scores vs Training Size for Sensitive Attribute: {self.sensitive_attribute}')
            plt.legend()
            plt.show()

    def masking(self, x_data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        masked_data = x_data.copy()
        if self.sensitive_attribute in masked_data.columns:
            masked_data[self.sensitive_attribute] = self.mask
        return masked_data

    def main(self, filepath: str, sensitive_attribute: str, target_column: str,
             display_metrics: bool = False, show_plots: bool = False) -> dict:
        df = pd.read_csv(filepath, header=0, skipinitialspace=True)
        df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

        y = df[target_column]
        X = df.drop(columns=[target_column])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        self.train(X_train, y_train, X_val, y_val, X_test, y_test, show_plots)

        X_test_copy = X_test.copy()
        X_test_copy[target_column] = y_test.to_numpy()
        dataset_true = BinaryLabelDataset(df=X_test_copy, label_names=[target_column],
                                          protected_attribute_names=[sensitive_attribute])

        y_pred = self.model.predict(X_test)
        X_test_pred = X_test.copy()
        X_test_pred[target_column] = y_pred
        dataset_predicted = BinaryLabelDataset(df=X_test_pred, label_names=[target_column],
                                               protected_attribute_names=[sensitive_attribute])

        metric = BinaryLabelDatasetMetric(dataset_true, privileged_groups=[{sensitive_attribute: 1}],
                                          unprivileged_groups=[{sensitive_attribute: 0}])
        classification_metric = ClassificationMetric(dataset_true, dataset_predicted,
                                                     privileged_groups=[{sensitive_attribute: 1}],
                                                     unprivileged_groups=[{sensitive_attribute: 0}])

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        disparate_impact = metric.disparate_impact()
        statistical_parity_diff = metric.statistical_parity_difference()
        ppv_privileged = classification_metric.positive_predictive_value(privileged=True)
        ppv_unprivileged = classification_metric.positive_predictive_value(privileged=False)
        ppv_parity = abs(ppv_privileged - ppv_unprivileged)

        fpr_privileged = classification_metric.false_positive_rate(privileged=True)
        fpr_unprivileged = classification_metric.false_positive_rate(privileged=False)
        fpr_parity = abs(fpr_privileged - fpr_unprivileged)

        if display_metrics:
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}')
            print("Bias Metrics")
            print(f'Disparate Impact: {disparate_impact}')
            print(f'Statistical Parity Difference: {statistical_parity_diff}')
            print(f"PPV Parity: {ppv_parity}")
            print(f"FPR Parity: {fpr_parity}")

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Disparate Impact": disparate_impact,
            "Statistical Parity Difference": statistical_parity_diff,
            "PPV Parity": ppv_parity,
            "FPR Parity": fpr_parity
        }
