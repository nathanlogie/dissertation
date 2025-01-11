from sklearn.linear_model import LogisticRegression

from Adaptive_masking.adaptivebaseline import AdaptiveBaseline
from Adaptive_masking.bias_metrics import example_bias_metric
from batching_strats import batching_strats

def main():
    processed_datasets = [
        ("../datasets/processed_datasets/adult.csv", "sex", "income"),
        ("../datasets/processed_datasets/german_credit.csv", "age", "credit_risk"),
        ("../datasets/processed_datasets/compass.csv", "race", "two_year_recid")

    ]

    for dataset in processed_datasets:
        for batching_strategy in batching_strats:
            print(dataset[0], batching_strategy.__name__)
            currAdaptive = AdaptiveBaseline(
                model=LogisticRegression(),
                bias_metric=example_bias_metric,
                threshold=0.1,
                sensitive_attribute="sex",
                batching=batching_strategy
            )

            currAdaptive.main(
                filepath=dataset[0],
                sensitive_attribute=dataset[1],
                target_column=dataset[2]
            )

if __name__ == "__main__":
    main()