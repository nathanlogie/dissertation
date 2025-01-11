import pandas as pd
from datasets.dataset_processing.helper_functions import one_hot_encode_column


def recidivism_to_csv(create_csv: bool = True) -> pd.DataFrame:
    # Reading the dataset
    data = pd.read_csv("../../raw_datasets/Recidivism/compas-scores-two-years.csv",
                       header=0, skipinitialspace=True)

    data = data[['sex', 'age', 'age_cat', 'race',
                 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                 'priors_count', 'c_charge_degree', 'c_charge_desc',
                 'two_year_recid']]

    # Target value is already in binary but has to be flipped as a positive prediction is a negative outcome
    data['two_year_recid'] = data['two_year_recid'].apply(lambda x: 1 if x == 0 else 0)
    data['sex'] = data['sex'].apply(lambda x: 1 if x.strip() == 'Female' else 0)
    data['race'] = data['race'].apply(lambda x: 1 if x.strip() == "Caucasian" else 0)
    data['c_charge_degree'] = data['c_charge_degree'].apply(lambda x: 1 if x.strip() == "M" else 0)

    categorical_columns = ['age_cat', 'c_charge_desc']

    for column in categorical_columns:
        data = one_hot_encode_column(data, column)

    if create_csv:
        clean_csv_filename = "../../processed_datasets/compass.csv"
        data.to_csv(clean_csv_filename, index=False)
        print("Cleaned CSV created")

    return data


if __name__ == "__main__":
    recidivism_to_csv()
