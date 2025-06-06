from sagemaker.feature_store.feature_processor import (
    FeatureProcessor,
    FeatureProcessorConfig,
    FeatureGroupQuery,
)

# Define the query (we assume event_time is available)
query = FeatureGroupQuery(
    feature_group_name=feature_group_name,
    record_identifier_name="subscription_id",
    event_time_feature_name="event_time"
)

# Point-in-time lookup config
config = FeatureProcessorConfig(
    query=query,
    target_stores=["OfflineStore"],
    start_time="2025-05-01T00:00:00Z",
    end_time="2025-06-01T00:00:00Z"
)

# Run the feature processor
processor = FeatureProcessor(config=config, sagemaker_session=sagemaker_session)
feature_df = processor.run()

# Now you have point-in-time correct features
X = feature_df.drop(columns=["churn_label_14d", "subscription_id", "event_time"])
y = feature_df["churn_label_14d"]

# Train an XGBoost model
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X, y)
