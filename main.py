import pandas as pd

from LFR_runs.main import lfr_main
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

    # LFR takes by far the longest to run so if you only wish to see adaptive masking compared to some then comment it out
    # LFR has issues with collapsing all results to 1 in the case of Recidivism Compass so it isn't run on that dataset

    main_runs = [baseline_main, dsp_main, lfr_main]
    all_combined_results = []

    for run in main_runs:
        print(f"Running {run.__name__} on all datasets")
        if run == lfr_main:
            result_df = run(datasets[:-1])
            placeholder_row = pd.DataFrame({
                "Dataset": ["Recidivism Compass"],
                "Accuracy": ["--"],
                "Precision": ["--"],
                "Recall": ["--"],
                "F1 Score": ["--"],
                "Disparate Impact": ["--"],
                "Statistical Parity Difference": ["--"],
                "PPV Parity": ["--"],
                "FPR Parity": ["--"]
            })
            result_df = pd.concat([result_df, placeholder_row], ignore_index=True)
        else:
            result_df = run(datasets)
        all_combined_results.append(result_df)

    combined_results_df = pd.concat(all_combined_results, ignore_index=True)
    combined_results_df.sort_values("Dataset", inplace=True)

    print(combined_results_df)
    combined_results_df.to_csv("combined_results.csv", index=False)

if __name__ == "__main__":
    main()
