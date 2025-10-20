from sklearn.pipeline import Pipeline
from src.pipelines.impute_missing import ImputeMissing
from src.pipelines.feature_engineering import FeatureEngineering




tree_pipeline = Pipeline([
    ("imputer", ImputeMissing(missing_threshold=0.8)),
    ("features", FeatureEngineering(model_type="tree")),
])

normal_pipeline = Pipeline([
    ("imputer", ImputeMissing(missing_threshold=0.8)),
    ("features", FeatureEngineering()),
])

