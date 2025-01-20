import warnings

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from adaptive_masking.adaptivebaseline import AdaptiveBaseline
from adaptive_masking.bias_metrics import example_bias_metric
from batching_strategies.batching_strats import batching_strats


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    adjusted_datasets = [
        {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv",
         "sensitive_attribute": "sex",
         "target_column": "income"}
    ]

    all_results = []
    batch_sizes = [8,16,32,64,96,128,256]
    for i in batch_sizes:
        print(i)
        currAdaptive = AdaptiveBaseline(
            model=LogisticRegression(solver='liblinear', random_state=1),
            bias_metric=example_bias_metric,
            threshold=0.1,
            sensitive_attribute=adjusted_datasets[0]["sensitive_attribute"],
            num_batches=i
        )

        results = currAdaptive.main(
            filepath=adjusted_datasets[0]["filepath"],
            sensitive_attribute=adjusted_datasets[0]["sensitive_attribute"],
            target_column=adjusted_datasets[0]["target_column"],
        )

        results["Dataset"] = adjusted_datasets[0]["name"]
        results["Batch Size"] = i
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["Batch Size"] + [col for col in results_df.columns if
                                            col not in ["Batch Size"]]]
    results_df.to_csv("batching_results.csv", index=False)
    plot(results_df)
    return results_df


def plot(results_df):
    batching_strategies = results_df

    plt.figure(figsize=(10, 6))

    accuracy_data = batching_strategies[["Batch Size", 'Accuracy']]
    accuracy_data = accuracy_data.groupby("Batch Size").mean()
    plt.plot(accuracy_data.index, accuracy_data['Accuracy'], label='Accuracy', color='blue', marker='o')

    FPR_data = batching_strategies[["Batch Size", 'FPR Parity']]
    FPR_data = FPR_data.groupby("Batch Size").mean()
    plt.plot(FPR_data.index, FPR_data['FPR Parity'], label='FPR Parity',
             color='red', marker='x')

    PPV_data = batching_strategies[["Batch Size", 'PPV Parity']]
    PPV_data = PPV_data.groupby("Batch Size").mean()
    plt.plot(PPV_data.index, PPV_data['PPV Parity'], label='PPV Parity',
             color='red', marker='x')

    plt.title(f'Accuracy and Disparate Impact vs "Batch Size"')
    plt.xlabel("Batch Size")
    plt.ylabel('Value')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
