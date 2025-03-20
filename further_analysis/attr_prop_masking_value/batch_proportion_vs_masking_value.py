import warnings
import os
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Updated dataset configurations using new CSVs from the local 'datasets' folder
    datasets = [
        {
            "name": "Income Census",
            "sensitive_attribute": "sex",
            "target_column": "income",
            "versions": {
                "equal": os.path.join("datasets", "income_census_equal.csv"),
                "75_privileged": os.path.join("datasets", "income_census_75_privileged.csv"),
                "75_unprivileged": os.path.join("datasets", "income_census_75_unprivileged.csv")
            }
        },
        {
            "name": "German Credit",
            "sensitive_attribute": "age",
            "target_column": "credit_risk",
            "versions": {
                "equal": os.path.join("datasets", "german_credit_equal.csv"),
                "75_privileged": os.path.join("datasets", "german_credit_75_privileged.csv"),
                "75_unprivileged": os.path.join("datasets", "german_credit_75_unprivileged.csv")
            }
        },
        {
            "name": "Recidivism Compass",
            "sensitive_attribute": "race",
            "target_column": "two_year_recid",
            "versions": {
                "equal": os.path.join("datasets", "recidivism_compass_equal.csv"),
                "75_privileged": os.path.join("datasets", "recidivism_compass_75_privileged.csv"),
                "75_unprivileged": os.path.join("datasets", "recidivism_compass_75_unprivileged.csv")
            }
        }
    ]

    masking_values = [-1, 0, 1]

    all_results = []
    models = [
        ("LR", LogisticRegression(solver='liblinear', random_state=1)),
        ("RF", RandomForestClassifier(random_state=1, n_jobs=-1))
    ]

    total_iterations = len(models) * len(masking_values) * sum(len(ds["versions"]) for ds in datasets)

    with tqdm(total=total_iterations, desc="Processing runs", unit="run") as pbar:
        for model in models:
            for mask in masking_values:
                for dataset in datasets:
                    for version, filepath in dataset["versions"].items():
                        pbar.set_description(
                            f"Model: {model[0]}, Dataset: {dataset['name']}, Version: {version}, Mask: {mask}")
                        currAdaptive = AdaptiveBaseline(
                            model=model[1],
                            bias_metric=example_bias_metric,
                            threshold=0.1,
                            sensitive_attribute=dataset["sensitive_attribute"],
                            batching_function=batching_strats[-1],
                            mask=mask
                        )

                        results = currAdaptive.main(
                            filepath=filepath,
                            sensitive_attribute=dataset["sensitive_attribute"],
                            target_column=dataset["target_column"],
                        )

                        results["Dataset"] = dataset["name"]
                        results["Version"] = version
                        results["Masking Value"] = mask
                        results["Model"] = model[0]

                        all_results.append(results)
                        pbar.update(1)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Dataset", "Model", "Version", "Masking Value"] +
        [col for col in results_df.columns if col not in ["Dataset", "Model", "Version", "Masking Value"]]
        ]
    results_df.sort_values(["Dataset", "Model", "Version", "Masking Value"], inplace=True)
    results_df.to_csv("masking_results.csv", index=False)

    return results_df


if __name__ == "__main__":
    main()
