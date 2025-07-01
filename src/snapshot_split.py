import pandas as pd
import xgboost as xgb
import yaml
from datetime import datetime as dt




def split_by_snapshot_dmatrix(df, train_end, val_end, train=True, yaml_path="bimonthly.yaml"):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    features = config["features"]
    label = config["label"]

    df = df.copy()
    df['SNAPSHOT_WEEK'] = pd.to_datetime(df['SNAPSHOT_WEEK'])

    df_train = df[df['SNAPSHOT_WEEK'].dt.date <= train_end]
    df_val   = df[(df['SNAPSHOT_WEEK'].dt.date > train_end) & 
                  (df['SNAPSHOT_WEEK'].dt.date <= val_end)]
    df_infer = df[df['SNAPSHOT_WEEK'].dt.date == val_end]

    X_train = df_train[features]
    y_train = df_train[label]

    X_val = df_val[features]
    y_val = df_val[label]

    X_infer = df_infer[features]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dinfer = xgb.DMatrix(X_infer)

    return dtrain, dval, dinfer, df_infer
