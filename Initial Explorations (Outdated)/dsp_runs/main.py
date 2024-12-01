from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dsp_runs.dsp import run_dsp


def dsp_main(datasets : list[dict],model:Union[LogisticRegression, RandomForestClassifier]) -> pd.DataFrame:

    all_results = []

    for dataset in datasets:
        result = run_dsp(
            filepath=dataset["filepath"],
            sensitive_attribute=dataset["sensitive_attribute"],
            target_column=dataset["target_column"],
            model=model
        )
        result["Dataset"] = dataset["name"]
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df["Run Type"] = "DSP"
    results_df = results_df[["Run Type","Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    model_name = model.__class__.__name__
    results_df.to_csv(f"individual_results/dsp_results_{model_name}.csv", index=False)
    return results_df

if __name__ == "__main__":
    dsp_main()
