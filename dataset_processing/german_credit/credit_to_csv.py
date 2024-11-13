import pandas as pd
from dataset_processing.helper_functions import one_hot_encode_column

column_names = [
    "status", "duration", "credit_history", "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

data = pd.read_csv("../../raw_datasets/German_Credit/german.data", header=None, names=column_names, sep=" ")

data.dropna(inplace=True)

# credit_risk: 1 for Good Credit Risk, 0 for Bad Credit Risk
data['credit_risk'] = data['credit_risk'].apply(lambda x: 1 if x == 1 else 0)  # Target value

# 2 protected attributes sex and age
# 1 for privileged group and 0 for unprivileged group
data['sex'] = data['personal_status_sex'].apply(lambda x: 1 if x.strip() in ['A91', 'A93', 'A94'] else 0)
data.drop("personal_status_sex", axis=1, inplace=True)
data['age'] = data['age'].apply(lambda x: 1 if x>25 else 0)

categorical_columns = [
    'status', 'credit_history', 'purpose', 'savings', 'employment',
    'other_debtors', 'property', 'other_installment_plans', 'housing', 'job',
    'telephone', 'foreign_worker'
]

for column in categorical_columns:
    data = one_hot_encode_column(data, column)

clean_csv_filename = "../../processed_datasets/german_credit.csv"
data.to_csv(clean_csv_filename, index=False)
