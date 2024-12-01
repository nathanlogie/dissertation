from typing import Union, Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from adaptivebaseline import AdaptiveBaseline

def run_adaptive_baseline(filepath: str, sensitive_attribute: str, target_column: str,
                          model: Union[LogisticRegression, RandomForestClassifier], batching: Callable,
                          display_metrics: bool = False) -> dict:
    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    target = df[target_column]
    sensitive_attr_values = df[sensitive_attribute]

    sensitive_attr_values = LabelEncoder().fit_transform(sensitive_attr_values)
    df[sensitive_attribute] = sensitive_attr_values

    X = df.drop(columns=[target_column])
    y = target

    def example_bias_metric(y_true: np.ndarray, y_pred_shadow: np.ndarray, sensitive_attr: np.ndarray) -> float:
        group_0_mean = y_pred_shadow[sensitive_attr == 0].mean()
        group_1_mean = y_pred_shadow[sensitive_attr == 1].mean()
        return abs(group_0_mean - group_1_mean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    adaptive_model = AdaptiveBaseline(
        model=model,
        bias_metric=example_bias_metric,
        threshold=0.1,
        sensitive_attribute=sensitive_attribute,
        batching=batching
    )

    adaptive_model.train(X_train, y_train)

    y_pred = adaptive_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_df[target_column] = y_pred

    aif_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=test_df,
        label_names=[target_column],
        protected_attribute_names=[sensitive_attribute]
    )

    dataset_metrics = BinaryLabelDatasetMetric(
        aif_data,
        privileged_groups=[{sensitive_attribute: 1}],
        unprivileged_groups=[{sensitive_attribute: 0}]
    )

    disparate_impact = dataset_metrics.disparate_impact()
    statistical_parity_diff = dataset_metrics.statistical_parity_difference()

    classification_metrics = ClassificationMetric(
        aif_data,
        aif_data,
        privileged_groups=[{sensitive_attribute: 1}],
        unprivileged_groups=[{sensitive_attribute: 0}]
    )

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
