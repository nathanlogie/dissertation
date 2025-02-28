import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats
from masking_strategies_def import masking_strats


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income",
         "threshold": 0.15},
        {"name": "German Credit", "filepath": "../datasets/processed_datasets/german_credit.csv",
         "sensitive_attribute": "age",
         "target_column": "credit_risk",
         "threshold": 0.1},
        {"name": "Recidivism Compass", "filepath": "../datasets/processed_datasets/compass.csv",
         "sensitive_attribute": "race",
         "target_column": "two_year_recid",
         "threshold": 0.1}
    ]
    all_results = []
    models = [
        ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=1)),
        ("Random Forest", RandomForestClassifier(random_state=1))
    ]
    for model in models:
        print("Model: ", model[0])
        for masking_strategy in masking_strats:
            print("Masking Strategy: ", masking_strategy.__name__)
            for dataset in datasets:
                print("Dataset", dataset["name"])
                if masking_strategy == masking_strats[1]:
                    current_strategy = masking_strategy(dataset["threshold"])
                    print("Threshold: ", dataset["threshold"])
                else:
                    current_strategy = masking_strategy
                currAdaptive = AdaptiveBaseline(
                    model=model[1],
                    bias_metric=example_bias_metric,
                    threshold=0.1,
                    sensitive_attribute=dataset["sensitive_attribute"],
                    batching_function=batching_strats[-1],
                    masking_strategy=current_strategy
                )

                results = currAdaptive.main(
                    filepath=dataset["filepath"],
                    sensitive_attribute=dataset["sensitive_attribute"],
                    target_column=dataset["target_column"],
                )
                results["Model"] = model[0]
                results["Dataset"] = dataset["name"]
                results["Masking Strategy"] = masking_strategy.__name__
                all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Dataset", "Model", "Masking Strategy"] + [col for col in results_df.columns if col not in ["Dataset", "Model","Masking Strategy"]]]
    results_df.sort_values(["Dataset","Model", "Masking Strategy"], inplace=True)
    results_df.to_csv("masking_results.csv", index=False)

    bias_results = results_df[[
        col for col in results_df.columns if col not in
                                             ["Accuracy","Balanced Accuracy","Precision","Recall","F1 Score"]
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
