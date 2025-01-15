import pandas as pd

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats

def adaptive_baseline_main(datasets : list[dict], model) -> pd.DataFrame:
    all_results = []

    for dataset in datasets:
        for batching_strategy in batching_strats:
            currAdaptive = AdaptiveBaseline(
                model=model,
                bias_metric=example_bias_metric,
                threshold=0.1,
                sensitive_attribute=dataset["sensitive_attribute"],
                batching=batching_strategy,
                num_batches=15
            )

            results = currAdaptive.main(
                filepath=dataset["filepath"],
                sensitive_attribute=dataset["sensitive_attribute"],
                target_column=dataset["target_column"],
                show_plots=False
            )

            results["Batching Strategy"] = batching_strategy.__name__
            results["Dataset"] = dataset["name"]

            all_results.append(results)
    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "Adaptive Masking"
    results_df["Model"] = model.__class__.__name__
    results_df = results_df[
        ["Run Type", "Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]

    return results_df