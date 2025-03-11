import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats, batching_names


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
    all_results = []
    models = [
        ("LR", LogisticRegression(solver='liblinear', random_state=1)),
        ("RF", RandomForestClassifier(random_state=1))
    ]
    for model in models:
        print("Model: ", model[0])
        for batching_strategy, batching_name in zip(batching_strats, batching_names):
            print("Batching Strategy: ", batching_name)
            for dataset in datasets:
                print("Dataset", dataset["name"])
                currAdaptive = AdaptiveBaseline(
                    model=model[1],
                    bias_metric=example_bias_metric,
                    threshold=0.1,
                    sensitive_attribute=dataset["sensitive_attribute"],
                    batching_function=batching_strategy,
                )

                results = currAdaptive.main(
                    filepath=dataset["filepath"],
                    sensitive_attribute=dataset["sensitive_attribute"],
                    target_column=dataset["target_column"],
                )

                results["Dataset"] = dataset["name"]
                results["Batch Selection Strategy"] = batching_name
                results["Model"] = model[0]
                all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Dataset","Model", "Batch Selection Strategy"] + [col for col in results_df.columns if col not in ["Dataset","Model", "Batch Selection Strategy"]]]
    results_df.sort_values(["Dataset", "Model","Batch Selection Strategy"], inplace=True)
    results_df.to_csv("batching_results.csv", index=False)

    bias_results = results_df[[
        col for col in results_df.columns if col not in
                                             ["Accuracy","Bal. Acc.","Precision","Recall","F1 Score"]
        ]
    ]
    bias_results.to_csv("bias_only_results.csv", index=False)

    performance_results = results_df[[
        col for col in results_df.columns if col not in
                                             ["Disparate Impact","Statistical Parity Difference","Average Odds Difference","Equal Opportunity Difference"]
        ]
    ]
    performance_results.to_csv("performance_only_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    main()
