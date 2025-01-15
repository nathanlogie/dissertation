import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from adaptive_masking.main import adaptive_baseline_main
from baseline.main import baseline_main


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

    models = [
        ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=1)),
        ("Random Forest", RandomForestClassifier(random_state=1))
    ]

    main_runs = [baseline_main, adaptive_baseline_main]
    all_combined_results = []

    for model_name, model in models:
        print(f"Running experiments with {model_name}")
        for run in main_runs:
            print(f"Running {run.__name__} on all datasets with {model_name}")
            result_df = run(datasets, model)
            result_df["Model"] = model_name
            all_combined_results.append(result_df)

    combined_results_df = pd.concat(all_combined_results, ignore_index=True)
    combined_results_df.sort_values(["Dataset", "Model", "Run Type"], inplace=True)

    print(combined_results_df)
    combined_results_df.to_csv("combined_results.csv", index=False)


if __name__ == "__main__":
    main()
