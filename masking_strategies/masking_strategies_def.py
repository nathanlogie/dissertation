from typing import Union

import numpy as np
import pandas as pd


def baseline_masking(x_data: Union[np.ndarray, pd.DataFrame], sensitive_attribute: str, mask: int) -> Union[np.ndarray, pd.DataFrame]:
    masked_data = x_data.copy()
    if sensitive_attribute in masked_data.columns:
        masked_data[sensitive_attribute] = mask
    return masked_data


def expanded_masking(x_data: Union[np.ndarray, pd.DataFrame], sensitive_attribute: str, mask: int) -> Union[np.ndarray, pd.DataFrame]:
    masked_data = x_data.copy()
    threshold = 0.8

    if sensitive_attribute in masked_data.columns:
        corr_matrix = masked_data.corr().abs()

        correlated_features = corr_matrix[sensitive_attribute][corr_matrix[sensitive_attribute] > threshold].index.tolist()

        for feature in correlated_features:
            masked_data[feature] = mask

        masked_data[sensitive_attribute] = mask

    return masked_data

masking_strats = [baseline_masking, expanded_masking]


