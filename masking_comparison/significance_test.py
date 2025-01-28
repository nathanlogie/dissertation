import pandas as pd
from pandas import read_csv
from scipy.stats import f_oneway

data = read_csv("masking_results.csv")
df = pd.DataFrame(data)

metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]


counter = {metric : 0 for metric in metrics}

for dataset in df["Dataset"].unique():
    curr = df[df["Dataset"] == dataset]
    results = {}
    print(dataset)
    for metric in metrics:
        groups = [df[df["Masking Value"] == val][metric] for val in df["Masking Value"].unique()]
        f_stat, p_value = f_oneway(*groups)
        significance = "Significant" if p_value < 0.10 else "Insignificant"
        results[metric] = significance
        if significance == "Significant":
            counter[metric] += 1

for metric, count in counter.items():
    print(f"{metric}: {count}")
