import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

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

# Making the categorical data numerical
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0) # Target Value
data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Male' else 0)

ord_encoder = OrdinalEncoder()

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_workclass_df = pd.DataFrame(
    onehot_encoder.fit_transform(data[['workclass']]),
    columns=onehot_encoder.get_feature_names_out(['workclass'])
)
data = data.drop(columns=['workclass']).reset_index(drop=True)
data = pd.concat([data, encoded_workclass_df], axis=1)

data['race'] = ord_encoder.fit_transform(data[['race']])
data['workclass'] = ord_encoder.fit_transform(data[['workclass']])

# Producing the final CSV
clean_csv_filename = "adult.csv"
data.to_csv(clean_csv_filename, index=False)
