import pandas as pd
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset


def run_dsp(filepath: str, sensitive_attribute: str, target_column: str, display_metrics: bool = False) -> dict:

    df = pd.read_csv(filepath, header=0, skipinitialspace=True)
    target_name = "predicted_" + target_column
    df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_dataset = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1),
                                       label_names=[target_column],
                                       protected_attribute_names=[sensitive_attribute])

    di_remover = DisparateImpactRemover(repair_level=1.0)
    train_dataset_repaired = di_remover.fit_transform(train_dataset)

    X_train_repaired = train_dataset_repaired.features
    y_train_repaired = train_dataset_repaired.labels.ravel()

    model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
    model.fit(X_train_repaired, y_train_repaired)
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

    metric = BinaryLabelDatasetMetric(dataset_true,
                                      privileged_groups=[{sensitive_attribute: 1}],
                                      unprivileged_groups=[{sensitive_attribute: 0}])

    classification_metric = ClassificationMetric(dataset_true, dataset_predicted,
                                                 privileged_groups=[{sensitive_attribute: 1}],
                                                 unprivileged_groups=[{sensitive_attribute: 0}])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    disparate_impact = metric.ratio(metric.base_rate)

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
