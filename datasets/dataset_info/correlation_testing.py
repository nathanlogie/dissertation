import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

census_columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
census_data = pd.read_csv(
    "../raw_datasets/Census_Income/adult.data",
    header=None,
    names=census_columns,
    na_values='?',
    skipinitialspace=True
)
census_corr = census_data.apply(lambda x: pd.factorize(x)[0]).corr().abs()


recidivism = pd.read_csv("../raw_datasets/Recidivism/compas-scores-two-years.csv")
recidivism_corr = recidivism.apply(lambda x: pd.factorize(x)[0]).corr().abs()

german_columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "credit_risk"
]
german_credit = pd.read_csv(
    "../raw_datasets/German_Credit/german.data",
    header=None,
    names=german_columns,
    sep=" "
)
german_corr = german_credit.apply(lambda x: pd.factorize(x)[0]).corr().abs()
common_attrs = min(census_corr.shape[0], recidivism_corr.shape[0], german_corr.shape[0])

datasets = [
    ("Census Income", "sex", census_corr),
    ("COMPAS Recidivism", "race", recidivism_corr),
    ("German Credit", "age", german_corr)
]
trimmed_matrices = {}
for dataset_name, sensitive_attr, corr_matrix in datasets:
    if corr_matrix.shape[0] > common_attrs:
        # Sort features descending by absolute correlation with the sensitive attribute
        sorted_features = corr_matrix[sensitive_attr].abs().sort_values(ascending=False).index.tolist()
        if sensitive_attr not in sorted_features[:common_attrs]:
            sorted_features = sorted_features[:common_attrs - 1] + [sensitive_attr]
            sorted_features = sorted(sorted_features, key=lambda x: corr_matrix[sensitive_attr].abs()[x], reverse=True)
        else:
            sorted_features = sorted_features[:common_attrs]
        corr_matrix = corr_matrix.loc[sorted_features, sorted_features]
    trimmed_matrices[dataset_name] = (sensitive_attr, corr_matrix)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (dataset_name, (sensitive_attr, corr_matrix)) in zip(axes, trimmed_matrices.items()):
    num_features = len(corr_matrix.columns)
    hm = sns.heatmap(
        corr_matrix,
        ax=ax,
        annot=True,
        fmt='.1f',
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.5,
        linecolor='black',
        square=True
    )

    custom_xticklabels = [col if col == sensitive_attr else "" for col in corr_matrix.columns]
    custom_yticklabels = [row if row == sensitive_attr else "" for row in corr_matrix.index]
    ax.set_xticks(np.arange(num_features) + 0.5)
    ax.set_xticklabels(custom_xticklabels, rotation=0, fontsize=10)
    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(custom_yticklabels, rotation=0, fontsize=10)

    # Outline the column corresponding to the sensitive attribute in red
    if sensitive_attr in corr_matrix.columns:
        sensitive_index = list(corr_matrix.columns).index(sensitive_attr)
        ax.add_patch(plt.Rectangle((sensitive_index, 0), 1, num_features,
                                   fill=False, edgecolor='red', lw=3))
    # Outline the row corresponding to the sensitive attribute in red
    if sensitive_attr in corr_matrix.index:
        sensitive_index_row = list(corr_matrix.index).index(sensitive_attr)
        ax.add_patch(plt.Rectangle((0, sensitive_index_row), num_features, 1,
                                   fill=False, edgecolor='red', lw=3))

    ax.set_title(dataset_name, fontsize=12, fontweight='bold', color='black')
    ax.set_xlabel("")
    ax.set_ylabel("")

norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, orientation='vertical').ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig('combined_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()
