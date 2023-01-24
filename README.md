# OTTO-kaggle-tetsuro731

Codes and data used for Kaggle OTTO competition:
https://www.kaggle.com/competitions/otto-recommender-system


# Strategy
- I introduce 2 phase model
  - Candidate phase
  - Ranking phase

# 0. Prepare matrix
- Generate Co-Visitatio Matrix which is based on Kaggle Code
  - `./kaggle_notebook/generate_matrix.ipynb`
  - This code is copied from Kaggle notebook and it can be run on the Kaggle environment.

# 1. Candidate phase

- Generate candidates for each session by user history and Co-Visitation Matrix
Run `otto_feature_generation.ipynb`

# 2. Ranking phase

## Feature generation
- generate session (user) and aid (item) features which are used for ranking model generation and predictions.

Run `generate_session_aid_features.ipynb`

## 2-1. Training
- LightGBM is used 
  - nDCG@20 and recall is applied as offline metrics
 
Run `otto_lgb_train.ipynb`

## 2-2. Prediction
 
Run `otto_lgb_test.ipynb`


# Others

`./src`
- functions used for train/prediction.
