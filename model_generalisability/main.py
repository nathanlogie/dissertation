import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from adaptive_masking.main import adaptive_baseline_main
from baseline.main import baseline_main
from sklearn.svm import LinearSVC

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

    models = [
        ("Linear SVC", LinearSVC(dual=False, random_state=1)),
        ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=1)),
        ("Random Forest", RandomForestClassifier(n_jobs=-1, random_state=1)),
        ("SVC", SVC(kernel="linear", random_state=1)),
        ("LGBM Classifier", LGBMClassifier(objective="binary", random_state=1)),
        ("XGBC Classifier",XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
        ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))
    ]

    main_runs = [
                ("Baseline", baseline_main),
                ("Adaptive Masking", adaptive_baseline_main)
                 ]
    all_combined_results = []
    errored_runs = []

    initial_time = time.time()

    for model_name, model in models:
        print(f"Running experiments with {model_name}")
        start_time = time.time()
        for run_name,run in main_runs:
            print(f"Running {run_name} on all datasets with {model_name}")
            try:
                result_df = run(datasets, model)
                result_df["Model"] = model_name
                all_combined_results.append(result_df)
            except Exception as e:
                print(e)
                print("errored on ", model_name, run_name)
                errored_runs.append((model_name, run_name))
                continue
        print(f"{model_name} took {time.time() - start_time:.2f} seconds")

    combined_results_df = pd.concat(all_combined_results, ignore_index=True)
    combined_results_df.sort_values(["Dataset", "Model", "Run Type"], inplace=True)
    combined_results_df = combined_results_df[
        ["Run Type", "Dataset", "Model"] +
        [col for col in combined_results_df.columns if col not in ["Run Type", "Dataset", "Model"]]
    ]

    print(combined_results_df)
    combined_results_df.to_csv("model_generalisability.csv", index=False)

    bias_results = combined_results_df[[
        col for col in combined_results_df.columns if col not in
                                             ["Accuracy", "Bal. Acc.", "Precision", "Recall", "F1 Score"]
    ]
    ]
    bias_results.to_csv("bias_only_model_generalisability_results.csv", index=False)

    performance_results = combined_results_df[[
        col for col in combined_results_df.columns if col not in
                                             ["Disparate Impact", "Statistical Parity Difference",
                                              "Average Odds Difference", "Equal Opportunity Difference"]
    ]
    ]
    performance_results.to_csv("performance_only_model_generalisability_combined_results.csv", index=False)
    print("COMPLETE")
    print(errored_runs)
    print("Total time taken: ", time.time() - initial_time)
    return combined_results_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
