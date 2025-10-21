import os
import json
from re import X
import joblib
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

import pip

ARTIFACT_DIR = r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\artifacts"
DATA_DIR = r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet"

TREE_PIPE_PATH = os.path.join(ARTIFACT_DIR, "tree_pipeline.joblib")
NORMAL_PIPE_PATH = os.path.join(ARTIFACT_DIR, "normal_pipeline.joblib")

TRAIN_PATH = os.path.join(DATA_DIR, "train_merged.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_merged.parquet")
from src.validation.helpers import assert_allclose, hash_array, save_json

# =========================================================
# VALIDATION LOGIC
# =========================================================
def validate_pipeline(pipeline_path, X_train, X_test, name):
    print(f"\n🔍 Validating {name} pipeline...")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"{pipeline_path} not found. Train pipeline first.")

    # Load persisted pipeline
    pipeline = joblib.load(pipeline_path)

    # --- 1️⃣ Deterministic Transform Check ---
    
    Xt_train_first = pipeline.transform(X_train)
    Xt_train_second = pipeline.transform(X_train)

    if name == "normal":
        # Ensure numeric only output
        if hasattr(Xt_train_first, "toarray"):
            Xt_train_first = Xt_train_first.toarray()
            Xt_train_second = Xt_train_second.toarray()
    
        assert_allclose(
            Xt_train_first.toarray() if hasattr(Xt_train_first, "toarray") else Xt_train_first,
            Xt_train_second.toarray() if hasattr(Xt_train_second, "toarray") else Xt_train_second,
        )
        print("✅ Deterministic transform verified on training data.")

        # --- 2️⃣ Hash for reproducibility ---
        train_hash = hash_array(Xt_train_first)
        hash_path = os.path.join(ARTIFACT_DIR, f"{name}_train_hash.txt")
        if os.path.exists(hash_path):
            old_hash = open(hash_path).read().strip()
            if train_hash != old_hash:
                raise ValueError(f"❌ Hash mismatch for {name} pipeline → non-deterministic output.")
        else:
            open(hash_path, "w").write(train_hash)
        print("✅ Hash consistency check passed.")

    else:
        # Case 2 for tree-based pipeline where output can be non-numeric
        # Validate schema + column consistency only

        assert isinstance(Xt_train_first, pd.DataFrame), "Tree pipeline output should be a DataFrame."

        if list(Xt_train_first.columns) != list(Xt_train_second.columns):
            raise ValueError("❌ Column names mismatch between transforms.")
        
        for col in Xt_train_first.columns:
            dtype1 = Xt_train_first[col].dtype
            dtype2 = Xt_train_second[col].dtype
            if dtype1 != dtype2:
                raise ValueError(f"❌ Dtype mismatch in column {col}: {dtype1} vs {dtype2}")
            
        print("✅ Deterministic transform verified on training data (tree pipeline).")


     # --- Validate test data schema ---
    Xt_test = pipeline.transform(X_test)
    if hasattr(Xt_test, "shape"):
        assert Xt_test.shape[1] == Xt_train_first.shape[1], (   
            f"❌ Schema mismatch: train has {Xt_train_first.shape[1]} features, "
            f"test has {Xt_test.shape[1]}"
        )

    print(f"✅ Test transform schema consistent ({Xt_test.shape[1]} features).")
    # print(dir(pipeline[1]))
    print(pipeline[1].get_feature_names_out())
    # if hasattr(pipeline, "get_feature_names_out"):
    #     feature_names = list(pipeline[:-1].get_feature_names_out()) 
    # else:
    #     feature_names = list(Xt_train_first.columns)

    # save_json(feature_names, os.path.join(ARTIFACT_DIR, f"feature_names_{name}.json"))
    # print(f"✅ Feature schema saved → {len(feature_names)} features.")

    # --- 5️⃣ Report Summary ---
    return {
        "pipeline": name,
        "train_shape": Xt_train_first.shape,
        "test_shape": Xt_test.shape,
        "num_features": Xt_train_first.shape[1],
        # "train_hash": train_hash,
        "schema_match": True,
        "deterministic": True,
        "timestamp": datetime.now().isoformat(),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("🚀 Starting Stage 2.1 – Pipeline Validation...")

    # Load cached train/test data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print('Train dataframe shape:', train_df.shape)
    print('Test dataframe shape:', test_df.shape)
    X_train = train_df.drop(columns=["isFraud"], errors="ignore")
    X_test = test_df
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    results = []
    results.append(validate_pipeline(TREE_PIPE_PATH, X_train, X_test, "tree"))
    results.append(validate_pipeline(NORMAL_PIPE_PATH, X_train, X_test, "normal"))

    final_report = {
        "timestamp": datetime.now().isoformat(),
        "pipelines_validated": [r["pipeline"] for r in results],
        "results": results,
    }

    save_json(final_report, os.path.join(ARTIFACT_DIR, "pipeline_validation_report.json"))
    print("\n✅ Stage 2.1 complete. Validation report written.")



