from typing import Union

import pandas as pd
from aif360.algorithms.preprocessing import LFR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset


def run_with_lfr(filepath: str, sensitive_attribute: str, target_column: str,model:Union[LogisticRegression, RandomForestClassifier], display_metrics: bool = False) -> dict:
    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_dataset = BinaryLabelDataset(
        df=pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
        label_names=[target_column],
        protected_attribute_names=[sensitive_attribute])

    test_dataset = BinaryLabelDataset(
        df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
        label_names=[target_column],
        protected_attribute_names=[sensitive_attribute])

    lfr = LFR(unprivileged_groups=[{sensitive_attribute: 0}], privileged_groups=[{sensitive_attribute: 1}], verbose=0,
              seed=2024)

    lfr.fit(train_dataset)
    train_dataset_transformed = lfr.transform(train_dataset)
    test_dataset_transformed = lfr.transform(test_dataset)

    X_train_transformed = train_dataset_transformed.features
    y_train_transformed = train_dataset_transformed.labels.ravel()

    X_test_transformed = test_dataset_transformed.features
    y_test_transformed = test_dataset_transformed.labels.ravel()

    model = make_pipeline(StandardScaler(), model)
    model.fit(X_train_transformed, y_train_transformed)
    y_pred = model.predict(X_test_transformed)

    y_test_transformed = y_test.to_numpy()

    accuracy = accuracy_score(y_test_transformed, y_pred)
    precision = precision_score(y_test_transformed, y_pred)
    recall = recall_score(y_test_transformed, y_pred)
    f1 = f1_score(y_test_transformed, y_pred)

    X_test_copy = X_test.copy()
    X_test_copy[target_column] = y_test_transformed

    dataset_true = BinaryLabelDataset(df=X_test_copy,
                                      label_names=[target_column],
                                      protected_attribute_names=[sensitive_attribute])

    X_test_pred = X_test.copy()
    X_test_pred[target_column] = y_pred

    dataset_predicted = BinaryLabelDataset(df=X_test_pred,
                                           label_names=[target_column],
                                           protected_attribute_names=[sensitive_attribute])

    metric = BinaryLabelDatasetMetric(dataset_true,
                                      privileged_groups=[{sensitive_attribute: 1}],
                                      unprivileged_groups=[{sensitive_attribute: 0}])

    classification_metric = ClassificationMetric(dataset_true, dataset_predicted,
                                                 privileged_groups=[{sensitive_attribute: 1}],
                                                 unprivileged_groups=[{sensitive_attribute: 0}])

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
