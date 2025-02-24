import pandas as pd
from aif360.algorithms.inprocessing import PrejudiceRemover
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset


def main(filepath, sensitive_attribute, target_column):
    # Load dataset
    data = pd.read_csv(filepath)

    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Train-test split the original dataset before converting to StandardDataset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

    # Convert to StandardDataset format after splitting
    favorable_class = [1]  # Assuming favorable class is 1 (e.g., income > 50K)
    train_dataset = StandardDataset(train_data, label_name=target_column,
                                    protected_attribute_names=[sensitive_attribute],
                                    privileged_classes=[[1]], favorable_classes=favorable_class)
    test_dataset = StandardDataset(test_data, label_name=target_column,
                                   protected_attribute_names=[sensitive_attribute],
                                   privileged_classes=[[1]], favorable_classes=favorable_class)

    # Initialize the PrejudiceRemover
    model = LogisticRegression(solver='liblinear', random_state=1)
    technique = PrejudiceRemover(sensitive_attr=sensitive_attribute, class_attr=target_column)

    # Train the model with the prejudice remover
    technique.fit(train_dataset)

    # Predict on the test set
    y_pred = technique.predict(test_dataset)

    print(y_pred)

if __name__ == "__main__":
    main("../datasets/processed_datasets/adult.csv", sensitive_attribute="sex", target_column="income")
