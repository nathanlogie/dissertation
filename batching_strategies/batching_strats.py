import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances


def batch_by_correlation(training_data, target, sensitive_attribute, batch_size) -> list:
    """
     Batches the training data into batches ordered by how closely correlated the sensitive attribute is to the target variable.

     Parameters:
     - training_data (pandas.DataFrame): The training data to be batched.
     - target (str): The column name of the target variable in the dataset.
     - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
     - batch_size (int): The size of batch to use.

     Returns:
     - list: A list of batches, ordered by correlation between the sensitive attribute and the target variable.

     """
    epochs = int(len(training_data) / batch_size)
    batches = np.array_split(training_data, epochs)
    correlations = [
        (batch, abs(pearsonr(batch[sensitive_attribute], batch[target])[0]))
        for batch in batches
    ]
    sorted_batches = [b[0] for b in sorted(correlations, key=lambda x: x[1])]
    return sorted_batches


def batch_by_similarity(training_data, target, sensitive_attribute, batch_size) -> list:
    """
       Batches the training data into batches ordered by how similar the sensitive attribute distributions are

       Parameters:
       - training_data (pandas.DataFrame): The training data to be batched.
       - target (str): The column name of the target variable in the dataset. (unused in this function)
       - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
       - batch_size (int): The size of batch to use.

       Returns:
       - list: A list of batches, ordered by correlation between the sensitive attribute and the target variable.

   """

    def compute_similarity(batch):
        features = [col for col in training_data.columns if col not in {target, sensitive_attribute}]

        priv_group = batch[batch[sensitive_attribute] == 1][features]
        unpriv_group = batch[batch[sensitive_attribute] == 0][features]

        if priv_group.empty or unpriv_group.empty:
            return float('-inf')

        return -pairwise_distances(priv_group.mean().values.reshape(1, -1),
                                   unpriv_group.mean().values.reshape(1, -1),
                                   metric='euclidean')[0, 0]

    epochs = int(len(training_data) / batch_size)
    batches = np.array_split(training_data, epochs)

    similarities = [(batch, compute_similarity(batch)) for batch in batches]
    sorted_batches = [b[0] for b in sorted(similarities, key=lambda x: x[1], reverse=True)]

    return sorted_batches


def random_batching(training_data, target, sensitive_attribute, batch_size):
    """
       Batches the training data into batches ordered randomly

       Parameters:
       - training_data (pandas.DataFrame): The training data to be batched.
       - target (str): The column name of the target variable in the dataset. (unused in this function)
       - sensitive_attribute (str): The column name of the sensitive attribute in the dataset.
       - batch_size (int): The size of batch to use.

       Returns:
       - list: A list of batches, ordered randomly

   """

    epochs = int(len(training_data) / batch_size)
    return np.array_split(training_data, epochs)


batching_strats = [batch_by_correlation, batch_by_similarity,
                   random_batching]

batching_names = ["Correlation-Based", "Distribution-Based", "Random"]