import pandas as pd

from baseline_runs.main import baseline_main
from dsp_runs.main import dsp_main

def main():
    datasets = [
        {"name": "Income Census", "filepath": "datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income"},
        {"name": "German Credit", "filepath": "datasets/processed_datasets/german_credit.csv",
         "sensitive_attribute": "age",
         "target_column": "credit_risk"},
        {"name": "Recidivism Compass", "filepath": "datasets/processed_datasets/compass.csv",
         "sensitive_attribute": "race",
         "target_column": "two_year_recid"}
    ]

    main_runs = [baseline_main, dsp_main]
    all_combined_results = []

    for run in main_runs:
        print(f"Running {run.__name__} on all datasets")
        result_df = run(datasets)
        all_combined_results.append(result_df)
        print(f"Finished running {run.__name__}")

    combined_results_df = pd.concat(all_combined_results, ignore_index=True)
    combined_results_df.sort_values("Dataset", inplace=True)

    print(combined_results_df)
    combined_results_df.to_csv("combined_results.csv", index=False)

if __name__ == "__main__":
    main()
