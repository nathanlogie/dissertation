import pandas as pd
from dataset_processing.helper_functions import one_hot_encode_column

# Reading the dataset
data = pd.read_csv("../../raw_datasets/Recidivism/compas-scores-two-years.csv",
                   header=0, skipinitialspace=True)

data = data[['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid']]

# Target value is already in binary
data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Female' else 0)
data['race'] = data['race'].apply(lambda x: 1 if x.strip() == "Caucasian" else 0)

categorical_columns = ['age_cat']

for column in categorical_columns:
    data = one_hot_encode_column(data, column)

clean_csv_filename = "../../processed_datasets/compass.csv"
data.to_csv(clean_csv_filename, index=False)
print("Cleaned CSV created")
