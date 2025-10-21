# src/pipelines/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from src.utils.logger import get_logger

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Full-feature engineering transformer for fraud detection.

    Features included:
      - Cardinality-based encoding (OHE / frequency)
      - Log + robust scaling
      - Temporal, aggregation, ratio, interaction features
      - Optional PCA on V-features
      - Dual-mode: numeric / tree

    Parameters
    ----------
    model_type : {"numeric", "tree"}
        "numeric" → encode, scale, PCA
        "tree"    → keep raw categoricals
    pca_components : int or None
        Number of PCA components for V-features.
    feature_prefix : str
        Prefix for engineered feature names.
    verbose : bool
        Whether to self.logger.info progress.
    """

    def __init__(self, model_type="numeric", pca_components=30,
                 feature_prefix="feat_", verbose=True):
        self.model_type = model_type
        self.pca_components = pca_components
        self.feature_prefix = feature_prefix
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)

        # learned state
        self.ohe_ = None
        self.scaler_ = None
        self.robust_scaler_ = None
        self.pca_ = None
        self.low_card_cols_ = []
        self.med_card_cols_ = []
        self.high_card_cols_ = []
        self.freq_maps_ = {}          # frequency maps for freq-encoded columns
        self.feature_names_ = None

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        self.logger.info(f"[FeatureEngineering] Starting fit. Initial shape: {X.shape}")
        # --- Detect categorical columns and split by cardinality ---
        object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in object_cols:
            uniq = X[col].nunique(dropna=False)
            if uniq < 10:
                self.low_card_cols_.append(col)
            elif 10 <= uniq <= 100:
                self.med_card_cols_.append(col)
            else:
                self.high_card_cols_.append(col)

        self.logger.info(f"[FeatureEngineering] Detected {len(self.low_card_cols_)} low-card, {len(self.med_card_cols_)} med-card, {len(self.high_card_cols_)} high-card categorical columns.")
        # --- Fit OneHotEncoder on low-card cols ---
        if self.model_type == "numeric" and self.low_card_cols_:
            self.logger.info(f"[FeatureEngineering] Fitting OneHotEncoder on {len(self.low_card_cols_)} low-cardinality columns.")
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe_.fit(X[self.low_card_cols_])
            self.ohe_features_  = self.ohe_.get_feature_names_out(self.low_card_cols_).tolist()

        # --- Learn frequency maps for med/high cardinality ---
        self.logger.info(f"[FeatureEngineering] Learning frequency encoding for {len(self.med_card_cols_) + len(self.high_card_cols_)} medium/high-cardinality columns.")
        for col in self.med_card_cols_ + self.high_card_cols_:

            freqs = X[col].value_counts(dropna=False)
            self.freq_maps_[col] = freqs / freqs.sum()


        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()


        
        # Fit standard scaler on numeric (for C*, D* features)
        self.std_cols = [c for c in numeric_cols if c.startswith(("C", "D"))]

        if self.model_type == "numeric" and self.std_cols:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.std_cols])

        # Fit robust scaler on heavy-tailed vars
        if self.model_type == "numeric" and "TransactionAmt" in X.columns:
            amt_log = np.log1p(X["TransactionAmt"])
            self.robust_scaler_ = RobustScaler()
            
            self.robust_scaler_.fit(amt_log.values.reshape(-1, 1)) # type: ignore

        self.logger.info(f"[FeatureEngineering] Fitted scalers on {len(self.std_cols)} C*/D* features and 'TransactionAmt_log'.")
        self.logger.info(f"[FeatureEngineering] Data shape before PCA: {X.shape}")

        agg_configs = {
            "card1": ["mean", "std", "count"],
            "addr1": ["mean", "std"],
            "P_emaildomain": ["mean"]
        }

        self.agg_maps = {}

        for key, funcs in agg_configs.items():
            if key in X.columns and "TransactionAmt" in X.columns:
                agg = X.groupby(key)["TransactionAmt"].agg(funcs) # type: ignore
                self.agg_maps[key] = agg
                

        # --- PCA on V-features ---
        self.v_cols = [c for c in X.columns if c.startswith("V")]
        self.logger.info(f"[FeatureEngineering] Fitting PCA on {len(self.v_cols)} V-features.")
        if self.model_type == "numeric" and self.v_cols and self.pca_components:
            self.pca_ = PCA(n_components=self.pca_components, random_state=42)
            self.pca_.fit(X[self.v_cols].fillna(0))
            self.logger.info(f"[FeatureEngineering] Fitted PCA on {len(self.v_cols)} V-features, retaining {self.pca_components} components.")
            self.logger.info(f"[FeatureEngineering] Data shape after PCA will be: {X.shape[0]} samples, No of columns is {X.shape[1]}")
        return self

    # =========================================================
    # TRANSFORM
    # =========================================================
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        self.logger.info(f"[FeatureEngineering] Starting transform. Initial shape: {X.shape}")
        # --- Recreate TransactionAmt_log ---
        if "TransactionAmt" in X.columns:
            X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])

        self.logger.info(f"[FeatureEngineering] log-transforming 'TransactionAmt'.")
        self.logger.info(f"[FeatureEngineering] Shape after log-transform: {X.shape}")
        # --- Apply temporal features ---
        if "TransactionDT" in X.columns:
            X["hour"] = (X["TransactionDT"] / 3600) % 24
            X["day"] = (X["TransactionDT"] / (3600 * 24)) % 7
            X["week"] = (X["TransactionDT"] / (3600 * 24 * 7))
            X["is_weekend"] = X["day"].isin([5, 6]).astype(int)

            self.logger.info(f"[FeatureEngineering] Created temporal features: hour, day, week, is_weekend.")
            self.logger.info(f"[FeatureEngineering] Shape after temporal features: {X.shape}")

        # --- Aggregation features ---
        
        for key,agg in self.agg_maps.items():
            unseen_ratio = (~X[key].isin(self.agg_maps[key].index)).mean()
            if unseen_ratio > 0:
                self.logger.warning(f"[FeatureEngineering] {unseen_ratio:.2%} unseen '{key}' values during transform.")
            agg_df = agg.reset_index()
            agg_df.columns = [key] + [f"{key}_TransactionAmt_{stat}" for stat in agg.columns]
            X = X.merge(agg_df, on=key, how="left")

        # --- Ratio & interaction features ---
        if "TransactionAmt" in X.columns and "card1_TransactionAmt_mean" in X.columns:
            X["amt_to_mean_card1"] = X["TransactionAmt"] / (X["card1_TransactionAmt_mean"] + 1e-5)

        if {"card4", "ProductCD"}.issubset(X.columns):
            X["card4_ProductCD"] = X["card4"].astype(str) + "_" + X["ProductCD"].astype(str)

        self.logger.info(f"[FeatureEngineering] Created ratio and interaction features.")
        self.logger.info(f"[FeatureEngineering] Shape after ratio and interaction features: {X.shape}")
        # ======================================================
        # Model-specific transformations
        # ======================================================
        if self.model_type == "numeric":
            self.logger.info(f"[FeatureEngineering] Applying numeric transformations.")
            X_final = self._transform_numeric(X)
        else:
            self.logger.info(f"[FeatureEngineering] Applying tree-based transformations.")
            X_final = self._transform_tree(X)

        self.logger.info(f"[FeatureEngineering] Completed transform. Final shape: {X_final.shape}")
        if self.verbose:
            self.logger.info(f"[FeatureEngineering] Final shape: {X_final.shape}")


        self.feature_names_ = X_final.columns.tolist()
        return X_final

    # =========================================================
    # Internal numeric transformations
    # =========================================================
    def _transform_numeric(self, X: pd.DataFrame):
        # --- One-hot encoding for low-card cols ---
        ohe_df = pd.DataFrame(index=X.index)
        if self.ohe_ and self.low_card_cols_:
            ohe_df = pd.DataFrame(
                self.ohe_.transform(X[self.low_card_cols_]), # type: ignore
                columns=[self.feature_prefix + f for f in self.ohe_.get_feature_names_out(self.low_card_cols_)],
                index=X.index
            ) # type: ignore

        # --- Frequency encoding for med/high-card cols ---
        freq_df = pd.DataFrame(index=X.index)
        for col, mapping in self.freq_maps_.items():
            freq_df[self.feature_prefix + f"{col}_freq"] = X[col].map(mapping).fillna(0)

        # --- Numeric scaling ---
        scaled_df = pd.DataFrame(index=X.index)
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

        # Standard scaling on C*, D* features
        self.std_cols = [c for c in num_cols if c.startswith(("C", "D"))]
        missing_std_cols = [c for c in self.std_cols if c not in X.columns]
        if missing_std_cols:
            self.logger.error(f"Missing standard scaling columns in transform: {missing_std_cols}")
            raise ValueError(f"Missing standard scaling input columns: {missing_std_cols}")
        if self.scaler_ and self.std_cols:
            scaled_df[self.std_cols] = self.scaler_.transform(X[self.std_cols])

        # Robust scaling on TransactionAmt_log
        if self.robust_scaler_ and "TransactionAmt_log" in X.columns:
            scaled_df["TransactionAmt_log"] = self.robust_scaler_.transform(X[["TransactionAmt_log"]])

        # --- PCA on V-features ---
        missing_v = [c for c in self.v_cols if c not in X.columns]
        if missing_v:
            self.logger.error(f"Missing V columns in transform: {missing_v}")
            raise ValueError(f"Missing PCA input columns: {missing_v}")
        
        if self.pca_ and self.v_cols:
            v_pca = self.pca_.transform(X[self.v_cols].fillna(0))
            v_pca_df = pd.DataFrame(
                v_pca,
                columns=[self.feature_prefix + f"V_pca_{i+1}" for i in range(self.pca_components)],
                index=X.index
            )
            scaled_df = pd.concat([scaled_df.drop(columns=self.v_cols, errors="ignore"), v_pca_df], axis=1)

        # --- Combine all feature sets ---
        X_final = pd.concat([scaled_df, ohe_df, freq_df], axis=1)
        return X_final

    # =========================================================
    # Internal tree transformations
    # =========================================================
    def _transform_tree(self, X: pd.DataFrame):
        """Light transformations for tree-based models."""
        cat_cols = X.select_dtypes(exclude=["number"]).columns
        for c in cat_cols:
            X[c] = X[c].astype("category")
        return X

    # =========================================================
    # Helpers
    # =========================================================

    
    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_", [])

