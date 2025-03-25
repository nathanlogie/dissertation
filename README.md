## Overview

This GitHub repo serves as the primary location of all code related to my dissertation work, which currently aims to
propose a new bias mitigation strategy in the form of adaptive masking.

## Personnel

Supervisee - Nathan George Logie: Student at UofG

Supervisor - Dr Graham Mcdonald: Senior Lecturer in Information Retrieval

## Datasets Used

The following datasets were selected based on availability and to allow accurate comparison to other methods commonly
tested on them.

- **Census Income Dataset**
    - Sourced from: https://archive.ics.uci.edu/dataset/2/adult
    - Official description: “Predict whether the annual income of an individual exceeds $50K/yr based on census data”
- **German Credit Dataset**
    - Sourced from https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    - Official description: “This dataset classifies people described by a set of attributes as good or bad credit
      risks”
- **ProPublica Recidivism/COMPAS Dataset**
    - Sourced from https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
    - Official description: “This dataset is used to assess the likelihood that a criminal defendant will re-offend”
    - Relevant paper: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

## Models  Used

Both models were selected to allow for testing with two different models and based on their use in AIF360

- Random Forest Classifier
    - Sourced from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Linear Regression
    - Sourced from https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LinearRegression.html

## Adaptive Masking and Baseline

The adaptive masking class is defined in the `adaptive_masking` folder with the `adaptivebaseline.py` file defining the class itself, `bias_metrics.py` defining the metric used in the evaluation, and `main.py` providing a function to run and evaluate adaptive masking on a given dataset/model

The baseline is defined in the `baseline` folder with the `baseline.py` file defining the run itself and `main.py` providing a standard run which gives baseline results for the given dataset(s)/model

## Running Experiments

Each experiment has a `main.py` file, which runs all the associated files with that experiment and outputs the results as a `.csv` file in the same directory. All results are already in each directory but are replaced when running the experiment anew

- Comparison of Batch Selection Strategies
    - Found in the `batching_strategies` folder, the batching strategies are defined in `batching_strats.py`
- Comparison of Batch Sizes
    - Found in the `epoch_comparison` folder
- Comparison of Attribute Masking Strategies
    - Found in the `masking_strategies` folder, masking strategies are defined in `masking_strategies_def.py`
- Comparison of Masking Values
- Found in the `masking_comparison` folder
- Comparison of Adaptive Masking Across Models
    - Found in the `model_generalisability` folder
    - Additional Models used are
        - SVC (Linear Kernel) - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        - XGBC Classifier - https://xgboost.readthedocs.io/en/stable/
        - MLP Classifier - [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- Comparison of Adaptive Masking against Other Mitigation Strategies
    - Found in the other_mitigation strategies folder
    - Other mitigation strategies used include
        - LFR
            - https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.preprocessing.LFR.html#aif360.algorithms.preprocessing.LFR
            - https://proceedings.mlr.press/v28/zemel13.html
        - Prejudice Remover
            - https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html#aif360.algorithms.inprocessing.PrejudiceRemover
            - https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3
        - Adversarial Debiasing
            - https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html#aif360.algorithms.inprocessing.AdversarialDebiasing
            - https://dl.acm.org/doi/10.1145/3278721.3278779
