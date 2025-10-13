from sklearn.pipeline import Pipeline
from src.pipelines.load_merge import LoadMerge
from src.pipelines.impute_missing import ImputeMissing
from src.pipelines.feature_engineering import FeatureEngineering

sources = {
    "transaction": "data/raw/train_transaction.parquet",
    "identity": "data/raw/train_identity.parquet"
}

main_pipeline = Pipeline([
    ("load_merge", LoadMerge(sources=sources, join_key="TransactionID")),
    ("imputer", ImputeMissing(missing_threshold=0.8)),
    ("features", FeatureEngineering(model_type="tree")),
])