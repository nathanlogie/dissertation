import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm  # Added progress bar support

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    adjusted_datasets = [
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
    batch_sizes = [8, 16, 32, 64, 96, 128, 256]
    masking_values = [-1, 0, 1]
    models = [
        ("LR", LogisticRegression(solver='liblinear', random_state=1)),
        ("RF", RandomForestClassifier(random_state=1))
    ]

    total_iterations = len(models) * len(adjusted_datasets) * len(batch_sizes) * len(masking_values)
    with tqdm(total=total_iterations, desc="Total Progress", unit="iteration") as pbar:
        for model in models:
            for dataset in adjusted_datasets:
                for batch_size in batch_sizes:
                    for mask in masking_values:
                        pbar.set_description(
                            f"Model: {model[0]}, Dataset: {dataset['name']}, Batch Size: {batch_size}, Mask: {mask}")

                        currAdaptive = AdaptiveBaseline(
                            model=model[1],
                            bias_metric=example_bias_metric,
                            threshold=0.1,
                            sensitive_attribute=dataset["sensitive_attribute"],
                            batch_size=batch_size,
                            mask=mask
                        )

                        results = currAdaptive.main(
                            filepath=dataset["filepath"],
                            sensitive_attribute=dataset["sensitive_attribute"],
                            target_column=dataset["target_column"],
                        )

                        results["Dataset"] = dataset["name"]
                        results["Batch Size"] = batch_size
                        results["Masking Value"] = mask
                        results["Model"] = model[0]
                        all_results.append(results)

                        pbar.update(1)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Dataset", "Model", "Batch Size", "Masking Value"] + [col for col in results_df.columns if
                                                               col not in ["Dataset", "Model", "Batch Size",
                                                                           "Masking Value"]]]
    results_df.sort_values(by=["Dataset", "Model", "Batch Size", "Masking Value"], inplace=True)
    results_df.to_csv("batch_size_epoch_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    main()
