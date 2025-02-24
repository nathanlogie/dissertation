import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("----Census Income----")
column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
census_data = pd.read_csv("../raw_datasets/Census_Income/adult.data", header=None, names=column_names, na_values='?'
                          , skipinitialspace=True)
cors = census_data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
# print(cors[cors["sex"]>0.15]["sex"].sort_values(ascending=False))
print(cors["sex"].sort_values(ascending=False))
census_cors = cors["sex"].sort_values(ascending=False).to_frame().T

print("----Recidivism----")
recidivsm = pd.read_csv("../raw_datasets/Recidivism/compas-scores-two-years.csv")
cors = recidivsm.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
# print(cors[cors["race"]>0.10]["race"].sort_values(ascending=False))
print(cors["race"].sort_values(ascending=False))
recid_cors = cors["race"].sort_values(ascending=False).to_frame().T
max_attr_count = max(len(census_cors), 20)  # 20 is the German dataset attribute count
recid_cors = recid_cors.iloc[:max_attr_count]
print("----German Credit----")
column_names = [
    "status", "duration", "credit_history", "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

german_credit = pd.read_csv("../raw_datasets/German_Credit/german.data", header=None, names=column_names, sep=" ")
cors = german_credit.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()
# print(cors[cors["age"]>0.1]["age"].sort_values(ascending=False))
print(cors["age"].sort_values(ascending=False))
credit_cors = cors["age"].sort_values(ascending=False)
credit_cors.to_csv("whatever.csv")

# fig, axes = plt.subplots(3, 1, figsize=(5, 18))  # Tall, narrow layout
#
# # Census Income Heatmap
# sns.heatmap(census_cors, cmap="viridis", annot=True, fmt=".3f", linewidths=0.5, cbar=True, ax=axes[0])
# axes[0].set_title("Census Income")
# axes[0].set_xlabel("Sex")  # Sensitive attribute at the top
# axes[0].set_ylabel("Attributes")  # Other attributes listed
#
# # Recidivism Heatmap
# sns.heatmap(recid_cors, cmap="viridis", annot=True, fmt=".3f", linewidths=0.5, cbar=True, ax=axes[1])
# axes[1].set_title("Recidivism")
# axes[1].set_xlabel("Race")
# axes[1].set_ylabel("Attributes")
#
# # German Credit Heatmap
# sns.heatmap(credit_cors, cmap="viridis", annot=True, fmt=".3f", linewidths=0.5, cbar=True, ax=axes[2])
# axes[2].set_title("German Credit")
# axes[2].set_xlabel("Age")
# axes[2].set_ylabel("Attributes")
#
# plt.tight_layout()
# plt.show()

ax = sns.heatmap(credit_cors, linewidth=0.5)
plt.show()
