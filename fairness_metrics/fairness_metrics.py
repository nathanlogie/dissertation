import numpy as np
from aif360.sklearn.metrics import disparate_impact_ratio

def demographic_parity(y_pred, sensitive_attr):
    # Check for potentaal issues with the sensitive attribute
    group1 = y_pred[sensitive_attr == 1]
    group0 = y_pred[sensitive_attr == 0]

    # Avoid division by zero or empty slices
    if group1.size == 0 or group0.size == 0:
        return np.nan  # Return NaN if any group has no instances

    dp_group1 = group1.mean()
    dp_group0 = group0.mean()

    return dp_group1 - dp_group0

def equal_opportunity(data, protected_attribute, predictions, true_labels, positive_class):
    tpr = data.groupby(protected_attribute).apply(
        lambda x: ((x[true_labels] == positive_class) & (x[predictions] == positive_class)).sum() /
                  (x[true_labels] == positive_class).sum())
    return tpr