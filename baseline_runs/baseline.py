import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

df = pd.read_csv("../processed_datasets/adult.csv", header=0, skipinitialspace=True)
sensitive_attribute = 'sex'
target_column = 'income'

df[sensitive_attribute] = LabelEncoder().fit_transform(df[sensitive_attribute])

y = df[target_column]
X = df.drop(columns=[target_column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

X_test['income'] = y_pred

X_test[sensitive_attribute] = LabelEncoder().fit_transform(X_test[sensitive_attribute])

dataset = BinaryLabelDataset(df=X_test, label_names=[target_column], protected_attribute_names=[sensitive_attribute])
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{sensitive_attribute: 1}], unprivileged_groups=[{sensitive_attribute: 0}])

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')