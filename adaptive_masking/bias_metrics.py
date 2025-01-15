def example_bias_metric(y_true, y_pred, sensitive_attr):
    group_0_mask = (sensitive_attr == 0)
    group_1_mask = (sensitive_attr == 1)

    group_0_mean = y_pred[group_0_mask].mean() if group_0_mask.any() else 0
    group_1_mean = y_pred[group_1_mask].mean() if group_1_mask.any() else 0

    return abs(group_1_mean - group_0_mean)