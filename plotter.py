import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("combined_results.csv")

df = pd.DataFrame(data)
df['Dataset'] = pd.Categorical(df['Dataset'])

metrics = ["Accuracy", "Prec.", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]
plt.figure(figsize=(16, 10))
sns.set_theme(style="ticks")
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    sns.barplot(data=df, x="Dataset", y=metric, hue="Run Type")

    plt.title(f'{metric}', fontsize=14, fontweight='bold')
    plt.xlabel("Datasets", fontsize=12)
    plt.ylabel(metric, fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)

plt.tight_layout()

plt.show()