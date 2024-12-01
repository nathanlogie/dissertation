import pandas as pd
from datasets.dataset_processing.helper_functions import one_hot_encode_column


def census_to_csv(create_csv: bool = True) -> pd.DataFrame:
    # Headers as of the raw_datasets.names file
    # age,workclass, fnlwgt, education, education-num, marital-status:, occupation, relationship, race, sex, capital-gain,
    # capital-loss,hours-per-week, native-country
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    data = pd.read_csv("../../raw_datasets/Census_Income/adult.data", header=None, names=column_names, na_values='?',
                       skipinitialspace=True)

    data.dropna(inplace=True)
    # This column will be encoded as part of processing so is unnecessary here
    data.drop(["education-num", 'fnlwgt', 'race'], axis=1, inplace=True)

    # Making the binary data numerical
    data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # Target Value
    data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Male' else 0)

    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']

    for column in categorical_columns:
        data = one_hot_encode_column(data, column)

    # Producing the final CSV
    if create_csv:
        clean_csv_filename = "../../processed_datasets/adult.csv"
        data.to_csv(clean_csv_filename, index=False)
    return data


if __name__ == "__main__":
    census_to_csv()
