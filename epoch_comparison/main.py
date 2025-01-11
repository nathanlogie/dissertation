import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sympy.plotting import plot3d

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
    plot(results_df)
    return results_df

def plot(results_df):
    batching_strategies = results_df['Batching Strategy'].unique()

    for batching_strategy in batching_strategies:
        plt.figure(figsize=(10, 6))

        strategy_data = results_df[results_df['Batching Strategy'] == batching_strategy]

        accuracy_data = strategy_data[['Batches', 'Accuracy']]
        accuracy_data = accuracy_data.groupby('Batches').mean()
        plt.plot(accuracy_data.index, accuracy_data['Accuracy'], label='Accuracy', color='blue', marker='o')

        FPR_data = strategy_data[['Batches', 'FPR Parity']]
        FPR_data = FPR_data.groupby('Batches').mean()
        plt.plot(FPR_data.index, FPR_data['FPR Parity'], label='FPR Parity',
                 color='red', marker='x')

        PPV_data = strategy_data[['Batches', 'PPV Parity']]
        PPV_data = PPV_data.groupby('Batches').mean()
        plt.plot(PPV_data.index, PPV_data['PPV Parity'], label='PPV Parity',
                 color='red', marker='x')

        plt.title(f'Accuracy and Disparate Impact vs Number of Batches ({batching_strategy})')
        plt.xlabel('Number of Batches')
        plt.ylabel('Value')
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()