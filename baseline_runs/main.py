import pandas as pd

from baseline_runs.baseline import run_baseline

def main():
    """
    Runs baseline experiments on three datasets and prints results.

    Datasets are:

    1. Income Census
    2. German Credit
    3. Recidivism Compass

    Each dataset is run with the provided `sensitive_attribute` and `target_column`.

    The results are stored in a Pandas DataFrame, printed to the console and converted to a csv.
    """
    datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv", "sensitive_attribute": "sex",
         "target_column": "income"},
        {"name": "German Credit", "filepath": "../datasets/processed_datasets/german_credit.csv", "sensitive_attribute": "age",
         "target_column": "credit_risk"},
        {"name": "Recidivism Compass", "filepath": "../datasets/processed_datasets/compass.csv", "sensitive_attribute": "race",
         "target_column": "two_year_recid"}
    ]

    all_results = []

    for dataset in datasets:
        print(dataset["name"] + "-- START")
        result = run_baseline(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"]
        )
        result["Dataset"] = dataset["name"]
        print(dataset["name"] + "-- DONE")
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["Dataset"] + [col for col in results_df.columns if col != "Dataset"]]
    results_df.to_csv("../results/baseline_results.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    main()
