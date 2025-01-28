import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    datasets = [
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

    masking_values = [-1, 0, 1]

    all_results = []
    for mask in masking_values:
        print("Masking value of", mask)
        for dataset in datasets:
            print("Dataset", dataset["name"])
            currAdaptive = AdaptiveBaseline(
                model=LogisticRegression(solver='liblinear', random_state=1),
                bias_metric=example_bias_metric,
                threshold=0.1,
                sensitive_attribute=dataset["sensitive_attribute"],
                batching_function=batching_strats[-1],
                mask=mask
            )

            results = currAdaptive.main(
                filepath=dataset["filepath"],
                sensitive_attribute=dataset["sensitive_attribute"],
                target_column=dataset["target_column"],
            )

            results["Dataset"] = dataset["name"]
            results["Masking Value"] = mask
            all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Dataset", "Masking Value"] + [col for col in results_df.columns if col not in ["Dataset", "Masking Value"]]]
    results_df.sort_values(["Dataset", "Masking Value"], inplace=True)
    results_df.to_csv("masking_results.csv", index=False)

    simplified_results = results_df.drop(["Dataset"], axis=1)
    simplified_results = simplified_results.groupby(["Masking Value"]).mean().reset_index().round(4)
    simplified_results.to_csv("masking_simplified_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    main()
