import pandas as pd
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.sklearn.metrics import average_odds_difference, equal_opportunity_difference
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def prejudice_main(filepath, sensitive_attribute, target_column):
    data = pd.read_csv(filepath)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

    favorable_class = [1]
    train_dataset = StandardDataset(train_data, label_name=target_column,
                                    protected_attribute_names=[sensitive_attribute],
                                    privileged_classes=[[1]], favorable_classes=favorable_class)
    test_dataset = StandardDataset(test_data, label_name=target_column,
                                   protected_attribute_names=[sensitive_attribute],
                                   privileged_classes=[[1]], favorable_classes=favorable_class)

    technique = PrejudiceRemover(sensitive_attr=sensitive_attribute, class_attr=target_column)
    technique.fit(train_dataset)

    y_pred = technique.predict(test_dataset)

    accuracy = accuracy_score(test_dataset.labels, y_pred.labels)
    bal_accuracy = balanced_accuracy_score(test_dataset.labels, y_pred.labels)
    precision = precision_score(test_dataset.labels, y_pred.labels)
    recall = recall_score(test_dataset.labels, y_pred.labels)
    f1 = f1_score(test_dataset.labels, y_pred.labels)

    dataset_predicted = BinaryLabelDataset(df=test_dataset.convert_to_dataframe()[0],
                                           label_names=[target_column],
                                           protected_attribute_names=[sensitive_attribute])

    dataset_predicted.labels = y_pred.labels

    metric_pred = BinaryLabelDatasetMetric(dataset_predicted,
                                           privileged_groups=[{sensitive_attribute: 1}],
                                           unprivileged_groups=[{sensitive_attribute: 0}])

    disparate_impact_pred = metric_pred.disparate_impact()
    statistical_parity_diff_pred = metric_pred.statistical_parity_difference()

    prot_attr_series = pd.Series(test_dataset.protected_attributes.ravel(), name="protected_attribute")

    average_odds = average_odds_difference(
        y_true=test_dataset.labels.ravel(),
        y_pred=y_pred.labels.ravel(),
        prot_attr=prot_attr_series
    )

    equal_opp = equal_opportunity_difference(
        y_true=test_dataset.labels.ravel(),
        y_pred=y_pred.labels.ravel(),
        prot_attr=prot_attr_series
    )
    return {
        "Accuracy": round(accuracy, 4),
        "Bal. Acc.": round(bal_accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Disparate Impact": round(disparate_impact_pred, 4),
        "Statistical Parity Difference": round(statistical_parity_diff_pred, 4),
        "Average Odds Difference": round(average_odds, 4),
        "Equal Opportunity Difference": round(equal_opp, 4)
    }


if __name__ == "__main__":
    datasets = [
        ("../datasets/processed_datasets/adult.csv", "sex", "income", "Income Census"),
        ("../datasets/processed_datasets/german_credit.csv", "age", "credit_risk", "German Credit"),
        ("../datasets/processed_datasets/compass.csv", "race", "two_year_recid", "Recidivism Compass")
    ]
    total = []
    for dataset in datasets:
        print(f"Running on {dataset[3]}")
        curr = prejudice_main(dataset[0], dataset[1], dataset[2])

        curr_df = pd.DataFrame([curr])
        curr_df["Dataset"] = dataset[3]
        curr_df["Run Type"] = "Prejudice Remover"

        total.append(curr_df)

    pd.concat(total).to_csv("prejudice_remover_results.csv", index=False)
