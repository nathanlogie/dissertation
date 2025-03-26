import os

import pandas as pd

datasets = [
    {
        "name": "Income Census",
        "sensitive_attribute": "sex",
        "target_column": "income",
        "versions": {
            "equal": os.path.join("datasets", "income_census_equal.csv"),
            "75_privileged": os.path.join("datasets", "income_census_75_privileged.csv"),
            "75_unprivileged": os.path.join("datasets", "income_census_75_unprivileged.csv")
        }
    },
    {
        "name": "German Credit",
        "sensitive_attribute": "age",
        "target_column": "credit_risk",
        "versions": {
            "equal": os.path.join("datasets", "german_credit_equal.csv"),
            "75_privileged": os.path.join("datasets", "german_credit_75_privileged.csv"),
            "75_unprivileged": os.path.join("datasets", "german_credit_75_unprivileged.csv")
        }
    },
    {
        "name": "Recidivism Compass",
        "sensitive_attribute": "race",
        "target_column": "two_year_recid",
        "versions": {
            "equal": os.path.join("datasets", "recidivism_compass_equal.csv"),
            "75_privileged": os.path.join("datasets", "recidivism_compass_75_privileged.csv"),
            "75_unprivileged": os.path.join("datasets", "recidivism_compass_75_unprivileged.csv")
        }
    }
]

all_summ = []
for dataset in datasets:
    for version in dataset["versions"]:
        df = pd.read_csv(dataset["versions"][version])
        print(f"Dataset: {dataset['name']}, Version: {version}, Rows: {len(df)}")
        print(f"Number of privileged: {len(df[df[dataset['sensitive_attribute']] == 1])}")
        print(f"Number of unprivileged: {len(df[df[dataset['sensitive_attribute']] == 0])}")

        all_summ.append({
            "Dataset": dataset["name"],
            "Version": version,
            "n": len(df),
            "Privileged Count": len(df[df[dataset['sensitive_attribute']] == 1]),
            "Unprivileged Count": len(df[df[dataset['sensitive_attribute']] == 0])
        })

df = pd.DataFrame(all_summ)
df.to_csv("dataset_info.csv", index=False)
