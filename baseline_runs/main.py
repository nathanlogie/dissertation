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

    The results are stored in a Pandas DataFrame and printed to the console.
    """
    datasets = [
        {"name": "Income Census", "filepath": "../processed_datasets/adult.csv", "sensitive_attribute": "sex",
         "target_column": "income"},
        {"name": "German Credit", "filepath": "../processed_datasets/german_credit.csv", "sensitive_attribute": "age",
         "target_column": "credit_risk"},
        {"name": "Recidivism Compass", "filepath": "../processed_datasets/compass.csv", "sensitive_attribute": "race",
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

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    print(results_df)

if __name__ == "__main__":
    main()
