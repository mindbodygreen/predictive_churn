import xgboost as xgb
from sklearn.model_selection import train_test_split





model = xgb.XGBClassifier(use_label_encoder=False, 
                          eval_metric='logloss',
                          n_estimators=500,
                          eta=0.05,
                          scale_pos_weight=42,
                          enable_categorical=True
                         )

model.fit(X_train, y_train)