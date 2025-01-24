import pandas as pd
from scipy.stats import f_oneway

# Data input
data = {
    "Dataset": ["German Credit", "German Credit", "German Credit",
                "Income Census", "Income Census", "Income Census",
                "Recidivism Compass", "Recidivism Compass", "Recidivism Compass"],
    "Masking Value": [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    "Accuracy": [0.777, 0.78, 0.787, 0.849, 0.849, 0.846, 0.674, 0.683, 0.678],
    "Precision": [0.809, 0.807, 0.817, 0.762, 0.741, 0.761, 0.663, 0.682, 0.681],
    "Recall": [0.89, 0.9, 0.895, 0.585, 0.618, 0.568, 0.845, 0.808, 0.795],
    "F1 Score": [0.847, 0.851, 0.854, 0.662, 0.674, 0.65, 0.743, 0.739, 0.733],
    "Disparate Impact": [0.881, 0.881, 0.881, 0.359, 0.359, 0.359, 0.832, 0.832, 0.832],
    "Statistical Parity Difference": [-0.085, -0.085, -0.085, -0.204, -0.204, -0.204, -0.106, -0.106, -0.106],
    "PPV Parity": [0.028, 0.026, 0.025, 0.002, 0.019, 0.128, 0.043, 0.02, 0.065],
    "FPR Parity": [0.123, 0.138, 0.269, 0.064, 0.08, 0.088, 0.206, 0.254, 0.073],
}

# Create DataFrame
df = pd.DataFrame(data)

# Metrics to evaluate
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Disparate Impact",
           "Statistical Parity Difference", "PPV Parity", "FPR Parity"]

# Initialize results dictionary
results = {}

# Perform one-way ANOVA for each metric
for metric in metrics:
    groups = [df[df["Masking Value"] == val][metric] for val in df["Masking Value"].unique()]
    f_stat, p_value = f_oneway(*groups)
    significance = "Significant" if p_value < 0.05 else "Insignificant"
    results[metric] = significance

# Print results
for metric, significance in results.items():
    print(f"{metric}: {significance}")
