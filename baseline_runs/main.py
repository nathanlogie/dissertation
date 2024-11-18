import pandas as pd

from baseline_runs.baseline import run_baseline

def baseline_main(datasets : list[dict]) -> pd.DataFrame:
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
            target_column=dataset["target_column"]
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "Baseline"
    results_df = results_df[["Run Type","Dataset"] + [col for col in results_df.columns if col != "Dataset"]]
    results_df.to_csv("individual_results/baseline_results.csv", index=False)

    return results_df

if __name__ == "__main__":
    baseline_main()
