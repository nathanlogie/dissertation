import pandas as pd
from dataset_processing.helper_functions import one_hot_encode_column

# Reading the dataset
data = pd.read_csv("../../raw_datasets/Recidivism/compas-scores-two-years.csv",
                   header=0, skipinitialspace=True)


# Dropping irrelevant columns
data.drop(["id", "name", "first", "last", "compas_screening_date", "dob", "c_case_number",
           "c_offense_date", "c_arrest_date", "c_jail_in", "c_jail_out"], axis=1, inplace=True)

# Target value is already in binary
data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Female' else 0)
data['race'] = data['race'].apply(lambda x: 1 if x.strip() == "Caucasian" else 0)

categorical_columns = ['age_cat', 'c_charge_degree', 'c_charge_desc',
       'r_case_number', 'r_charge_degree', 'r_offense_date', 'r_charge_desc',
       'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree',
       'vr_offense_date', 'vr_charge_desc', 'type_of_assessment', 'score_text',
       'screening_date', 'v_type_of_assessment', 'v_score_text',
       'v_screening_date', 'in_custody', 'out_custody']

for column in categorical_columns:
    data = one_hot_encode_column(data, column)

clean_csv_filename = "../../processed_datasets/compas.csv"
data.to_csv(clean_csv_filename, index=False)
print("Cleaned CSV created")
