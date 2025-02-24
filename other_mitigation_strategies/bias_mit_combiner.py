import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from other_mitigation_strategies.adverserial import adversarial_main
from other_mitigation_strategies.lfr import run_with_lfr
from other_mitigation_strategies.prejudice_remover import prejudice_main

@ignore_warnings(category=ConvergenceWarning)
def main():
    warnings.filterwarnings("ignore")

    datasets = [
        ("../datasets/processed_datasets/adult.csv", "sex", "income", "Income Census"),
        ("../datasets/processed_datasets/german_credit.csv", "age", "credit_risk", "German Credit"),
        ("../datasets/processed_datasets/compass.csv", "race", "two_year_recid", "Recidivism Compass")
    ]
    bias_mitigation_strategies = [
        ("Adversarial Debiasing",adversarial_main),
        ("Prejudice Remover",prejudice_main),
        ("LFR", run_with_lfr)
    ]

    total = []
    for i, bias_mit in enumerate(bias_mitigation_strategies):
        print("Running on bias mitigation strategy:", bias_mit[0])
        for dataset in datasets:
            print(f"Running on {dataset[3]}")
            if i == 2 and dataset[3] == "Recidivism Compass":
                continue
            curr = bias_mit[1](dataset[0], dataset[1], dataset[2])
            curr = pd.DataFrame([curr])
            curr["Dataset"] = dataset[3]
            curr["Run Type"] = bias_mit[0]
            total.append(curr)
    results_df = pd.concat(total, ignore_index=True)
    results_df = results_df[
        ["Run Type", "Dataset"] + [col for col in results_df.columns if col not in ["Run Type", "Dataset"]]]
    results_df.sort_values(["Dataset", "Run Type"], inplace=True)
    results_df.to_csv("combined_mitigation.csv", index=False)

if __name__ == "__main__":
    main()