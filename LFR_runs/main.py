import pandas as pd

from LFR_runs.lfr import run_with_lfr


def lfr_main(datasets : list[dict]) -> pd.DataFrame:

    all_results = []

    for dataset in datasets:
        print(f"Running LFR on {dataset['name']}")
        result = run_with_lfr(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"]
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "DSP"
    results_df = results_df[["Run Type","Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    results_df.to_csv("individual_results/dsp_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    lfr_main(datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income"},
        {"name": "German Credit", "filepath": "../datasets/processed_datasets/german_credit.csv",
         "sensitive_attribute": "age",
         "target_column": "credit_risk"},
        {"name": "Recidivism Compass", "filepath": "../datasets/processed_datasets/compass.csv",
         "sensitive_attribute": "race",
         "target_column": "two_year_recid"}
    ])
