import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Census Income dataset
print("----Census Income----")
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
census_data = pd.read_csv("../raw_datasets/Census_Income/adult.data", header=None, names=column_names, na_values='?',
                          skipinitialspace=True)
cors = census_data.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
census_cors = cors["sex"].sort_values(ascending=False).round(3)  # Round to 3 decimal places

# Load Recidivism dataset
print("----Recidivism----")
recidivsm = pd.read_csv("../raw_datasets/Recidivism/compas-scores-two-years.csv")
cors = recidivsm.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
recid_cors = cors["race"].sort_values(ascending=False).round(3)  # Round to 3 decimal places
max_attr_count = len(census_cors)# Ensure alignment with German dataset
recid_cors = recid_cors.iloc[:max_attr_count]

# Load German Credit dataset
print("----German Credit----")
column_names = [
    "status", "duration", "credit_history", "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "credit_risk"
]
german_credit = pd.read_csv("../raw_datasets/German_Credit/german.data", header=None, names=column_names, sep=" ")
cors = german_credit.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
credit_cors = cors["age"].sort_values(ascending=False).round(3)
credit_cors = credit_cors.iloc[:max_attr_count]  # Round to 3 decimal places

# Prepare for visualization
datasets = [
    ("Census Income", "sex", census_cors),
    ("Recidivism", "race", recid_cors),
    ("German Credit", "age", credit_cors)
]

# Increase figure size for slightly larger heatmaps
fig, axes = plt.subplots(ncols=3, figsize=(8, 6))  # Adjust figsize for larger heatmaps

for i, (ax, (dataset_name, sensitive_attr, correlations)) in enumerate(zip(axes, datasets)):
    correlation_matrix = np.array(correlations).reshape(-1, 1)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',  # Format numbers to 3 decimal places
        cmap="coolwarm",
        vmin=0, vmax=max(correlations[:-2]),  # Pearson correlation absolute values
        xticklabels=[sensitive_attr],
        yticklabels=[f"{name}" for name in correlations.index],  # Add the attribute name on the left
        cbar=False,
        linewidths=1.5,
        linecolor='black',
        ax=ax,
        square=True  # Ensure the heatmap cells are square
    )
    ax.set_title(f"{dataset_name}", fontsize=12, fontweight='bold', color='black')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight='bold', color='black')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, fontweight='bold', rotation=0, color='black')

plt.tight_layout()  # Adjust layout before saving
plt.gcf().tight_layout()  # Recalculate tight layout before saving
plt.savefig('heatmaps.png', dpi=300, bbox_inches='tight')  # Save with tight bounding box

plt.show()
