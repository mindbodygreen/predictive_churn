import pandas as pd
import xgboost as xgb
import yaml
import os
from datetime import datetime as dt


def split_by_snapshot_dmatrix(df, train_end, val_end, config_file="bimonthly.yaml", 
                             DATE_COLUMN="SNAPSHOT_WEEK", train=True):
    """
    Split dataframe by snapshot dates and create XGBoost DMatrix objects.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features and target
    train_end : datetime.date
        End date for training data (inclusive)
    val_end : datetime.date
        End date for validation data (inclusive)
    config_file : str, default="bimonthly.yaml"
        Name of the YAML config file containing features and label definitions
    DATE_COLUMN : str, default="SNAPSHOT_WEEK"
        Name of the date column used for splitting
    train : bool, default=True
        Whether to return training data (currently unused but kept for compatibility)
    
    Returns:
    --------
    tuple
        (dtrain, dval, dinfer, df_infer) - XGBoost DMatrix objects and inference dataframe
    """
    # Get the directory of the current file and construct the path to configs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(os.path.dirname(current_dir), "configs")
    yaml_file_path = os.path.join(configs_dir, config_file)
    
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)

    features = config["features"]
    label = config["label"]

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    df_train = df[df[DATE_COLUMN].dt.date <= train_end]
    df_val   = df[(df[DATE_COLUMN].dt.date > train_end) & 
                  (df[DATE_COLUMN].dt.date <= val_end)]
    df_infer = df[df[DATE_COLUMN].dt.date == val_end]

    X_train = df_train[features]
    y_train = df_train[label]

    X_val = df_val[features]
    y_val = df_val[label]

    X_infer = df_infer[features]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dinfer = xgb.DMatrix(X_infer)

    return dtrain, dval, dinfer, df_infer
