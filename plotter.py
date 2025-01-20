import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("combined_results.csv")

df = pd.DataFrame(data)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Disparate Impact", "Statistical Parity Difference"]
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(data=df, x="Dataset", y=metric, hue="Run Type")
    plt.title(metric)
    plt.xlabel("")
    plt.ylabel(metric)

plt.tight_layout()
plt.show()

summary_table = df.pivot(index=["Dataset"], columns="Run Type", values=["Accuracy", "Precision", "Recall", "F1 Score", "Disparate Impact", "Statistical Parity Difference"])
