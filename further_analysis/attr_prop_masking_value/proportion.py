import pandas as pd

datasets = [
    {"name": "Income Census", "filepath": "../datasets/processed_datasets/adult.csv", "sensitive_attribute": "sex",
     "target_column": "income"},
    {"name": "German Credit", "filepath": "../datasets/processed_datasets/german_credit.csv",
     "sensitive_attribute": "age", "target_column": "credit_risk"},
    {"name": "Recidivism Compass", "filepath": "../datasets/processed_datasets/compass.csv",
     "sensitive_attribute": "race", "target_column": "two_year_recid"}]

for dataset in datasets:
    print(f"Processing {dataset['name']}...")

    df = pd.read_csv(dataset["filepath"])
    sensitive_attr = dataset["sensitive_attribute"]

    privileged_df = df[df[sensitive_attr] == 1].copy()
    unprivileged_df = df[df[sensitive_attr] == 0].copy()

    equal_n = min(len(privileged_df), len(unprivileged_df))
    equal_df = pd.concat(
        [privileged_df.sample(equal_n, random_state=42), unprivileged_df.sample(equal_n, random_state=42)]).sample(
        frac=1, random_state=42)  # Shuffle the combined DataFrame

    y = min(len(unprivileged_df), len(privileged_df) // 3)
    x = 3 * y
    majority_priv_df = pd.concat(
        [privileged_df.sample(x, random_state=42), unprivileged_df.sample(y, random_state=42)]).sample(frac=1,
                                                                                                       random_state=42)

    x2 = min(len(privileged_df), len(unprivileged_df) // 3)
    y2 = 3 * x2
    majority_unpriv_df = pd.concat(
        [privileged_df.sample(x2, random_state=42), unprivileged_df.sample(y2, random_state=42)]).sample(frac=1,
                                                                                                         random_state=42)

    base_name = dataset["name"].replace(" ", "_").lower()

    equal_filename = f"datasets/{base_name}_equal.csv"
    majority_priv_filename = f"datasets/{base_name}_75_privileged.csv"
    majority_unpriv_filename = f"datasets/{base_name}_75_unprivileged.csv"

    equal_df.to_csv(equal_filename, index=False)
    majority_priv_df.to_csv(majority_priv_filename, index=False)
    majority_unpriv_df.to_csv(majority_unpriv_filename, index=False)

    print(f"  -> Saved equal version to: {equal_filename}")
    print(f"  -> Saved 75% privileged version to: {majority_priv_filename}")
    print(f"  -> Saved 75% unprivileged version to: {majority_unpriv_filename}\n")
