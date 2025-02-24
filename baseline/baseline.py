from typing import Union

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def run_baseline(filepath: str, sensitive_attribute: str, target_column: str,
                 model: Union[LogisticRegression, RandomForestClassifier], display_metrics: bool = False) -> dict:
    """
    Run a baseline machine learning model on the given dataset and compute fairness metrics.

    Parameters:
    - filepath (str): Path to the CSV file containing the dataset.
    - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
    - target_column (str): The column name of the target variable in the dataset.

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, F1 score, and fairness metrics
      such as Disparate Impact, Statistical Parity Difference, PPV Parity, and FPR Parity.

    The function reads the dataset from the given filepath, encodes the sensitive attribute,
    splits the data into training and testing sets, trains a logistic regression model, and
    evaluates its performance. It calculates fairness metrics using the AIF360 library to
    assess bias in the model predictions.
    """
    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    target_name = "predicted_" + target_column
    df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_test[target_column] = y_test
    X_test[target_name] = y_pred

    dataset_true = BinaryLabelDataset(df=X_test.drop(columns=[target_name]),
                                      label_names=[target_column],
                                      protected_attribute_names=[sensitive_attribute])

    X_test[target_column] = y_pred
    dataset_predicted = BinaryLabelDataset(df=X_test.drop(columns=[target_name]),
                                           label_names=[target_column],
                                           protected_attribute_names=[sensitive_attribute])

    metric = BinaryLabelDatasetMetric(dataset_predicted,
                                      privileged_groups=[{sensitive_attribute: 1}],
                                      unprivileged_groups=[{sensitive_attribute: 0}])

    classification_metric = ClassificationMetric(dataset_true, dataset_predicted,
                                                 privileged_groups=[{sensitive_attribute: 1}],
                                                 unprivileged_groups=[{sensitive_attribute: 0}])

    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Bias Metrics
    disparate_impact = metric.disparate_impact()  # Ratio of favorable outcomes
    statistical_parity_diff = metric.statistical_parity_difference()  # Difference in favorable outcome rates
    average_odds_diff = classification_metric.average_odds_difference()  # Average TPR and FPR difference
    equal_opportunity_diff = classification_metric.equal_opportunity_difference()  # Difference in TPR

    if display_metrics:
        print(f'Accuracy: {accuracy}')
        print(f'Balanced Accuracy: {bal_accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print("Bias Metrics")
        print(f'Disparate Impact: {disparate_impact}')
        print(f'Statistical Parity Difference: {statistical_parity_diff}')
        print(f'Average Odds Difference: {average_odds_diff}')
        print(f'Equal Opportunity Difference: {equal_opportunity_diff}')

    return {
        "Accuracy": round(accuracy, 4),
        "Balanced Accuracy": round(bal_accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Disparate Impact": round(disparate_impact, 4),
        "Statistical Parity Difference": round(statistical_parity_diff, 4),
        "Average Odds Difference": round(average_odds_diff, 4),
        "Equal Opportunity Difference": round(equal_opportunity_diff, 4)
    }

