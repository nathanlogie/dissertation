import pandas as pd
from sklearn.linear_model import LogisticRegression

from Adaptive_masking.adaptivebaseline import AdaptiveBaseline
from Adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats
import warnings

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    adjusted_datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income"}
        ]

    all_results = []
    for batching_strategy in batching_strats:
        for i in range(5,105,5):
            print(i, batching_strategy.__name__)
            currAdaptive = AdaptiveBaseline(
                model=LogisticRegression(solver='liblinear', random_state=1),
                bias_metric=example_bias_metric,
                threshold=0.1,
                sensitive_attribute=adjusted_datasets[0]["sensitive_attribute"],
                batching=batching_strategy,
                batch_number=i
            )

            results = currAdaptive.main(
                filepath=adjusted_datasets[0]["filepath"],
                sensitive_attribute=adjusted_datasets[0]["sensitive_attribute"],
                target_column=adjusted_datasets[0]["target_column"],
            )

            results["Batching Strategy"] = batching_strategy.__name__
            results["Dataset"] = adjusted_datasets[0]["name"]
            results["Batches"] = i
            all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Batching Strategy", "Batches"] + [col for col in results_df.columns if col not in ["Batching Strategy", "Batches"]]]
    results_df.sort_values(["Batching Strategy", "Batches"], inplace=True)
    results_df.to_csv("batching_results.csv", index=False)

    return results_df

if __name__ == "__main__":
    main()