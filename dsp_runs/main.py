import pandas as pd

from baseline_runs.baseline import run_baseline
from dsp_runs.dsp import run_dsp


def dsp_main(datasets : list[dict]) -> pd.DataFrame:

    all_results = []

    for dataset in datasets:
        result = run_dsp(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"]
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "DSP"
    results_df = results_df[["Run Type","Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    results_df.to_csv("individual_results/dsp_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    dsp_main()
