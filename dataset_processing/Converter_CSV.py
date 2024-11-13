import pandas as pd
from dataset_processing.helper_functions import one_hot_encode_column

# Headers as of the adult.names file
# age,workclass, fnlwgt, education, education-num, marital-status:, occupation, relationship, race, sex, capital-gain,
# capital-loss,hours-per-week, native-country
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

data = pd.read_csv("../adult/adult.data", header=None, names=column_names, na_values='?', skipinitialspace=True)

data.dropna(inplace=True)

# This column will be encoded as part of processing so is unnecessary here
data.drop("education-num", axis=1)

# Making the binary data numerical
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0) # Target Value
data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Male' else 0)

categorical_columns = ['race','workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']

for column in categorical_columns:
    data = one_hot_encode_column(data, column)

# Producing the final CSV
clean_csv_filename = "adult.csv"
data.to_csv(clean_csv_filename, index=False)
