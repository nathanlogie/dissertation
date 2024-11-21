from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset

from adaptive_runs.adaptivemasking import AdaptiveMasking


def run_adaptive(filepath: str, sensitive_attribute: str, target_column: str,
                 model: Union[LogisticRegression, RandomForestClassifier],
                 display_metrics: bool = False) -> dict:
    # Load and preprocess the dataset
    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    target = df[target_column]
    sensitive_attr_values = df[sensitive_attribute]

    # Encode categorical sensitive attributes
    sensitive_attr_values = LabelEncoder().fit_transform(sensitive_attr_values)
    df[sensitive_attribute] = sensitive_attr_values

    # Separate features and labels
    X = df.drop(columns=[target_column])
    y = target

    # Define a bias metric (example)
    def example_bias_metric(y_true, y_pred, sensitive_attr):
        # Dummy implementation of a bias metric
        return np.abs(y_pred[sensitive_attr == 1].mean() - y_pred[sensitive_attr == 0].mean())

    # Instantiate AdaptiveMasking
    adaptive_model = AdaptiveMasking(
        model=model,
        bias_metric=example_bias_metric,
        threshold=0.1,
        sensitive_attribute=sensitive_attribute
    )

    # Train and test the model
    adaptive_model.train(X, y, sensitive_attr_values)

    y_pred = adaptive_model.predict(X)
    df[target_name] = y_pred

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    aif_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[target_column],
        protected_attribute_names=[sensitive_attribute]
    )

    metrics = BinaryLabelDatasetMetric(aif_data, privileged_groups=[{sensitive_attribute: 1}],
                                       unprivileged_groups=[{sensitive_attribute: 0}])

    disparate_impact = metrics.ratio(metrics.base_rate)
    statistical_parity_diff = metrics.statistical_parity_difference()

    classification_metrics = ClassificationMetric(aif_data, aif_data, privileged_groups=[{sensitive_attribute: 1}],
                                                  unprivileged_groups=[{sensitive_attribute: 0}])

    ppv_privileged = classification_metrics.positive_predictive_value(privileged=True)
    ppv_unprivileged = classification_metrics.positive_predictive_value(privileged=False)
    ppv_parity = abs(ppv_privileged - ppv_unprivileged)

    fpr_privileged = classification_metrics.false_positive_rate(privileged=True)
    fpr_unprivileged = classification_metrics.false_positive_rate(privileged=False)
    fpr_parity = abs(fpr_privileged - fpr_unprivileged)

    if display_metrics:
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print("\nBias Metrics")
        print(f'Disparate Impact: {disparate_impact}')
        print(f'Statistical Parity Difference: {statistical_parity_diff}')
        print(f"PPV Parity: {ppv_parity}")
        print(f"FPR Parity: {fpr_parity}")

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Disparate Impact": disparate_impact,
        "Statistical Parity Difference": statistical_parity_diff,
        "PPV Parity": ppv_parity,
        "FPR Parity": fpr_parity
    }

    return results

if __name__ == "__main__":
    run_adaptive(filepath="../datasets/processed_datasets/german_credit.csv",
                 sensitive_attribute="age",
                 target_column="credit_risk",
                 model=LogisticRegression(solver='liblinear', random_state=1),
                 display_metrics=True)
