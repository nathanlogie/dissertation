import pandas as pd

from sensitive_removed_runs.sensitive_runs import run_sensitive_removed


def sens_removed_main(datasets : list[dict], model) -> pd.DataFrame:
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
        result = run_sensitive_removed(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"],
            model=model
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "SA Removed"
    results_df = results_df[
        ["Run Type", "Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    model_name = model.__class__.__name__
    results_df.to_csv(f"individual_results/sens_removed_results_{model_name}.csv", index=False)

    return results_df

if __name__ == "__main__":
    sens_removed_main()