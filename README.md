## Overview

This GitHub repo serves as the primary location of all code related to my dissertation work, which currently aims to propose a new bias mitigation strategy in the form of adaptive masking.

## Personnel

Supervisee - Nathan George Logie: Student at UofG

Supervisor - Dr Graham Mcdonald: Senior Lecturer in Information Retrieval

## Datasets Used

The following datasets were selected based on availability and to allow accurate comparison to other methods commonly tested on them.

- **Census Income Dataset**
    - Sourced from: https://archive.ics.uci.edu/dataset/2/adult
    - Official description: “Predict whether the annual income of an individual exceeds $50K/yr based on census data”
- **German Credit Dataset**
    - Sourced from https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    - Official description: “This dataset classifies people described by a set of attributes as good or bad credit risks”
- **ProPublica Recidivism/COMPAS Dataset**
    - Sourced from https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
    - Official description: “This dataset is used to assess the likelihood that a criminal defendant will re-offend”
    - Relevant paper: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

## Models  Used

Both models were selected to allow for testing with both classifier and regression models and based on their use in AIF360

- Random Forest Classifier
    - Sourced from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Linear Regression
    - Sourced from https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LinearRegression.html
