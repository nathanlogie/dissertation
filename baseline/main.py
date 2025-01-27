import pandas as pd
from sklearn.linear_model import LogisticRegression

from baseline.baseline import run_baseline


def baseline_main(datasets: list[dict], model) -> pd.DataFrame:
    """
    Runs baseline experiments on three datasets and prints individual_results.

    Datasets are:

    1. Income Census
    2. German Credit
    3. Recidivism Compass

    Each dataset is run with the provided `sensitive_attribute` and `target_column`.

    The individual_results are stored in a Pandas DataFrame, printed to the console and converted to a csv.
    """

    all_results = []

    for dataset in datasets:
        result = run_baseline(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"],
            model=model
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "Baseline"
    results_df = results_df[
        ["Run Type", "Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    return results_df


if __name__ == "__main__":
    adjusted_datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income"},
        {"name": "German Credit", "filepath": "../datasets/processed_datasets/german_credit.csv",
         "sensitive_attribute": "age",
         "target_column": "credit_risk"},
        {"name": "Recidivism Compass", "filepath": "../datasets/processed_datasets/compass.csv",
         "sensitive_attribute": "race",
         "target_column": "two_year_recid"}
    ]
    baseline_main(
        datasets=adjusted_datasets,
        model=LogisticRegression(solver='liblinear', random_state=1)
    )
