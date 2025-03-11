import pandas as pd
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def run_with_lfr(filepath: str, sensitive_attribute: str, target_column: str, display_metrics: bool = False) -> dict:
    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_dataset = BinaryLabelDataset(
        df=pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
        label_names=[target_column], protected_attribute_names=[sensitive_attribute])

    test_dataset = BinaryLabelDataset(
        df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
        label_names=[target_column], protected_attribute_names=[sensitive_attribute])

    lfr = LFR(unprivileged_groups=[{sensitive_attribute: 0}], privileged_groups=[{sensitive_attribute: 1}], verbose=0,
              seed=2024)

    lfr.fit(train_dataset)
    train_dataset_transformed = lfr.transform(train_dataset)
    test_dataset_transformed = lfr.transform(test_dataset)

    X_train_transformed = train_dataset_transformed.features
    y_train_transformed = train_dataset_transformed.labels.ravel()

    X_test_transformed = test_dataset_transformed.features

    model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
    model.fit(X_train_transformed, y_train_transformed)
    y_pred = model.predict(X_test_transformed)

    y_test_transformed = y_test.to_numpy()

    accuracy = accuracy_score(y_test_transformed, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test_transformed, y_pred)
    precision = precision_score(y_test_transformed, y_pred)
    recall = recall_score(y_test_transformed, y_pred)
    f1 = f1_score(y_test_transformed, y_pred)

    X_test_copy = X_test.copy()
    X_test_copy[target_column] = y_test_transformed

    dataset_true = BinaryLabelDataset(df=X_test_copy, label_names=[target_column],
                                      protected_attribute_names=[sensitive_attribute])

    X_test_pred = X_test.copy()
    X_test_pred[target_column] = y_pred

    dataset_predicted = BinaryLabelDataset(df=X_test_pred, label_names=[target_column],
                                           protected_attribute_names=[sensitive_attribute])

    metric = BinaryLabelDatasetMetric(dataset_true, privileged_groups=[{sensitive_attribute: 1}],
                                      unprivileged_groups=[{sensitive_attribute: 0}])

    classification_metric = ClassificationMetric(dataset_true, dataset_predicted,
                                                 privileged_groups=[{sensitive_attribute: 1}],
                                                 unprivileged_groups=[{sensitive_attribute: 0}])

    disparate_impact = metric.disparate_impact()
    statistical_parity_diff = metric.statistical_parity_difference()

    average_odds_diff = classification_metric.average_odds_difference()
    equal_opportunity_diff = classification_metric.equal_opportunity_difference()

    return {"Accuracy": round(accuracy, 4), "Bal. Acc.": round(bal_accuracy, 4),
        "Precision": round(precision, 4), "Recall": round(recall, 4), "F1 Score": round(f1, 4),
        "Disparate Impact": round(disparate_impact, 4),
        "Statistical Parity Difference": round(statistical_parity_diff, 4),
        "Average Odds Difference": round(average_odds_diff, 4),
        "Equal Opportunity Difference": round(equal_opportunity_diff, 4)}
