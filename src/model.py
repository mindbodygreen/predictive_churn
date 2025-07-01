# model.py

import xgboost as xgb
import pandas as pd


def train_churn_model(dtrain, dval=None):
    """
    Train an XGBoost model using DMatrix inputs.
    
    Parameters:
    - dtrain: xgb.DMatrix for training data (with label)
    - dval:   xgb.DMatrix for validation data (with label), optional
    
    Returns:
    - booster: trained xgboost Booster model
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "scale_pos_weight": 42,
        "verbosity": 0,
    }
    num_boost_round = 1000
    early_stopping_rounds = 30

    evals = [(dtrain, "train")]
    if dval is not None:
        evals.append((dval, "validation"))

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dval else None,
        verbose_eval=False,
    )

    return booster


def predict_churn(booster, dtest, threshold=0.7):
    """
    Predict churn probabilities and classes using the trained Booster.
    
    Parameters:
    - booster: trained xgboost Booster model
    - dtest: xgb.DMatrix for test data
    - threshold: probability cutoff to assign positive class (default 0.7)
    
    Returns:
    - pandas DataFrame with PREDICTED_PROBABILITY and PREDICTED_CLASS columns
    """
    y_prob = booster.predict(dtest)  
    y_pred = (y_prob > threshold).astype(int)

    

    # dtest doesn't contain feature names or index; you might want to add them externally
    results_df = pd.DataFrame({
        "PREDICTED_PROBABILITY": y_prob,
        "PREDICTED_CLASS": y_pred
    })

    return results_df

