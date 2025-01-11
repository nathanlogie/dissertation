import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances


def batch_equal_sensitive(training_data, target, sensitive_attribute, epochs) -> list:
    """
        Batches the training data into batches with an equal proportion of sensitive and non-sensitive examples.

        Parameters:
        - training_data (pandas.DataFrame): The training data to be batched.
        - target (str): The column name of the target variable in the dataset. (unused in this function)
        - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
        - epochs (int): The number of batches to create.

        Returns:
        - list: A list of batches, each containing an equal proportion of sensitive and non-sensitive examples.

        """

    batches = []
    for value in training_data[sensitive_attribute].unique():
        subset = training_data[training_data[sensitive_attribute] == value]
        split_batches = np.array_split(subset, epochs)
        for i in range(epochs):
            if len(batches) < epochs:
                batches.append(split_batches[i])
            else:
                batches[i] = pd.concat([batches[i], split_batches[i]])
    return batches


def batch_demographic_parity(training_data, target, sensitive_attribute, epochs) -> list:
    """
          Batches the training data into batches that all meet the demographic parity 4/5 ratio criterion.

          Parameters:
          - training_data (pandas.DataFrame): The training data to be batched.
          - target (str): The column name of the target variable in the dataset.
          - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
          - epochs (int): The number of batches to create.

          Returns:
          - list: A list of batches, each meeting the demographic parity 4/5 ratio criterion.

      """

    def meets_demographic_parity(batch):
        priv_outcomes = batch[batch[sensitive_attribute] == 1][target]
        unpriv_outcomes = batch[batch[sensitive_attribute] == 0][target]
        if priv_outcomes.mean() > 0:
            return (unpriv_outcomes.mean() / priv_outcomes.mean()) >= 0.8
        return True

    batches = []
    subsets = np.array_split(training_data, epochs)
    for subset in subsets:
        if meets_demographic_parity(subset):
            batches.append(subset)
    return batches


def batch_by_correlation(training_data, target, sensitive_attribute, epochs) -> list:
    """
     Batches the training data into batches ordered by how closely correlated the sensitive attribute is to the target variable.

     Parameters:
     - training_data (pandas.DataFrame): The training data to be batched.
     - target (str): The column name of the target variable in the dataset.
     - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
     - epochs (int): The number of batches to create.

     Returns:
     - list: A list of batches, ordered by correlation between the sensitive attribute and the target variable.

     """

    batches = np.array_split(training_data, epochs)
    correlations = [
        (batch, abs(pearsonr(batch[sensitive_attribute], batch[target])[0]))
        for batch in batches
    ]
    sorted_batches = [b[0] for b in sorted(correlations, key=lambda x: x[1])]
    return sorted_batches


def batch_by_similarity(data, target, sensitive_attribute, epochs) -> list:
    """
       Batches the training data into batches ordered by how similar the sensitive attribute distributions are

       Parameters:
       - training_data (pandas.DataFrame): The training data to be batched.
       - target (str): The column name of the target variable in the dataset. (unused in this function)
       - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
       - epochs (int): The number of batches to create.

       Returns:
       - list: A list of batches, ordered by correlation between the sensitive attribute and the target variable.

   """

    def compute_similarity(batch):
        features = [col for col in data.columns if col not in {target, sensitive_attribute}]

        priv_group = batch[batch[sensitive_attribute] == 1][features]
        unpriv_group = batch[batch[sensitive_attribute] == 0][features]

        if priv_group.empty or unpriv_group.empty:
            return float('-inf')

        return -pairwise_distances(priv_group.mean().values.reshape(1, -1),
                                   unpriv_group.mean().values.reshape(1, -1),
                                   metric='euclidean')[0, 0]


    batches = np.array_split(data, epochs)

    similarities = [(batch, compute_similarity(batch)) for batch in batches]
    sorted_batches = [b[0] for b in sorted(similarities, key=lambda x: x[1], reverse=True)]

    return sorted_batches

def no_batching(training_data, target, sensitive_attribute, epochs):
    return np.array_split(training_data, epochs)

batching_strats = [batch_equal_sensitive, batch_demographic_parity, batch_by_correlation, batch_by_similarity, no_batching]