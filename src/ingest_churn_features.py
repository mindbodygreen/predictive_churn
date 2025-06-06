import boto3
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
import time

sagemaker_session = Session()
region = sagemaker_session.boto_region_name
s3_bucket = sagemaker_session.default_bucket()

# Create your DataFrame (example)
df = pd.DataFrame({
    'subscription_id': ['sub_001', 'sub_002'],
    'customer_id': ['cust_001', 'cust_002'],
    'email': ['a@example.com', 'b@example.com'],
    'cycles_completed': [3, 5],
    'is_active': [1, 0],
    'next_order_date': ['2025-06-15', '2025-06-18'],
    'churn_label_14d': [0, 1],
    'event_time': pd.to_datetime(['2025-06-01', '2025-06-01'])
})

# Define feature group name
feature_group_name = "subscription_churn_features"

# Define schema
feature_definitions = [
    FeatureDefinition("subscription_id", FeatureTypeEnum.STRING),
    FeatureDefinition("customer_id", FeatureTypeEnum.STRING),
    FeatureDefinition("email", FeatureTypeEnum.STRING),
    FeatureDefinition("cycles_completed", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("is_active", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("next_order_date", FeatureTypeEnum.STRING),  # or TIMESTAMP
    FeatureDefinition("churn_label_14d", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("event_time", FeatureTypeEnum.STRING),  # Must be ISO format
]

# Create the feature group
feature_store_client = boto3.client('sagemaker-featurestore-runtime', region_name=region)
feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

# Register the feature group (only once)
feature_group.create(
    feature_definitions=feature_definitions,
    record_identifier_name="subscription_id",
    event_time_feature_name="event_time",
    role_arn="arn:aws:iam::<your-account-id>:role/<your-sagemaker-execution-role>",
    enable_online_store=False,
    offline_store_config={
        "S3StorageConfig": {"S3Uri": f"s3://{s3_bucket}/feature-store/"},
        "DisableGlueTableCreation": False,
        "DataCatalogConfig": {
            "TableName": feature_group_name,
            "Database": "sagemaker_featurestore"
        },
    }
)

# Wait until feature group is ready
feature_group.wait_for_create()

# Ingest data
feature_group.ingest(data_frame=df, max_workers=3, wait=True)
