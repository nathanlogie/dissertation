from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def one_hot_encode_column(data, column_name):
    """
    One-hot encode a specified column and merge it into the original DataFrame.

    Parameters:
    - data (pd.DataFrame): The Current DataFrame.
    - column_name (str): The column name to one-hot encode.

    Returns:
    - pd.DataFrame: The updated DataFrame with one-hot encoded columns.
    """
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    encoded_df = pd.DataFrame(
        onehot_encoder.fit_transform(data[[column_name]]),
        columns=onehot_encoder.get_feature_names_out([column_name])
    )

    encoded_df.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data.drop(columns=[column_name]), encoded_df], axis=1)

    return data
