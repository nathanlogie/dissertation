from aif360.sklearn.metrics import disparate_impact_ratio

def demographic_parity(y_true, y_pred, sensitive_attr):
    dp_group1 = y_pred[sensitive_attr == 1].mean()
    dp_group0 = y_pred[sensitive_attr == 0].mean()
    return abs(dp_group1 - dp_group0)

def equal_opportunity(data, protected_attribute, predictions, true_labels, positive_class):
    tpr = data.groupby(protected_attribute).apply(
        lambda x: ((x[true_labels] == positive_class) & (x[predictions] == positive_class)).sum() /
                  (x[true_labels] == positive_class).sum())
    return tpr