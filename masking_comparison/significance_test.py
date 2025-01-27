import pandas as pd
from pandas import read_csv
from scipy.stats import f_oneway

data = read_csv("masking_results.csv")
df = pd.DataFrame(data)

metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]


results = {}

for metric in metrics:
    groups = [df[df["Masking Value"] == val][metric] for val in df["Masking Value"].unique()]
    f_stat, p_value = f_oneway(*groups)
    significance = "Significant" if p_value < 0.05 else "Insignificant"
    results[metric] = significance

for metric, significance in results.items():
    print(f"{metric}: {significance}")
