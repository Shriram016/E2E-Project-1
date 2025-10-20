# src/pipelines/impute_missing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import json, os
from src.utils.logger import get_logger
class ImputeMissing(BaseEstimator, TransformerMixin):
    """
    Handles missing-value treatment in a consistent, sklearn-compatible way.

    Steps:
      1️⃣ Drops columns with missing ratio > threshold.
      2️⃣ Imputes numeric columns with median.
      3️⃣ Imputes categorical columns with 'Missing' (constant).
      4️⃣ Logs imputation metadata for reproducibility.

    Parameters
    ----------
    missing_threshold : float
        Columns with missing ratio above this threshold are dropped. (Default=0.8)
    log_path : str or None
        Optional path to save imputation summary JSON.
    verbose : bool
        Whether to self.logger.info progress information.
    """

    def __init__(self, missing_threshold: float = 0.8, log_path: str|None = None, verbose: bool = True,strict_schema:bool=False):
        self.missing_threshold = missing_threshold
        self.log_path = log_path if log_path else ""
        self.verbose = verbose
        self.strict_schema = strict_schema

        # internal state after fit
        self.drop_cols_ = []
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.fill_values_ = {}  # only for loggingssss
        self.logger = get_logger(self.__class__.__name__)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers on training data."""
        X = X.copy()

        # 1️⃣ Drop columns above missing threshold
        missing_ratio = X.isna().mean()
        self.drop_cols_ = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        self.logger.info(f"Shape of X before dropping columns: {X.shape}")
        X = X.drop(columns=self.drop_cols_, errors="ignore")
        self.logger.info(f"Shape of X after dropping columns: {X.shape}")
        self.logger.info(f"[ImputeMissing] Identified {len(self.drop_cols_)} columns to drop due to >{self.missing_threshold:.0%} missing values.")
         
        if self.verbose and self.drop_cols_:
            self.logger.info(f"[ImputeMissing] Dropping {len(self.drop_cols_)} columns above {self.missing_threshold:.0%} missing threshold.")

        # 2️⃣ Separate numeric and categorical columns
        self.num_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=["number"]).columns.tolist()

        # 3️⃣ Fit imputers
        if self.num_cols_:
            self.num_imputer_ = SimpleImputer(strategy="median")
            self.num_imputer_.fit(X[self.num_cols_])
            for c, val in zip(self.num_cols_, self.num_imputer_.statistics_):
                if pd.isna(val):
                    self.logger.warning(f"[ImputeMissing] Column {c} median is NaN. This may indicate all values are missing.")
                self.fill_values_[c] = val

        if self.cat_cols_ :
            self.cat_imputer_ = SimpleImputer(strategy="constant", fill_value="Missing")
            self.cat_imputer_.fit(X[self.cat_cols_ ])
            for c in self.cat_cols_ :
                self.fill_values_[c] = "Missing"

        # 4️⃣ Log metadata
        if self.log_path:
            self._save_log()

        return self

    def transform(self, X: pd.DataFrame):
        """Apply column dropping and imputation."""
        expected_cols = set(self.num_cols_ + self.cat_cols_ )
        missing_cols = expected_cols - set(X.columns)
        if missing_cols:
            self.logger.warning(f"[ImputeMissing] Input data is missing expected columns: {missing_cols}")
            raise ValueError(f"Input data is missing expected columns: {missing_cols}")
        
        
        X = X.copy()

        self.logger.info(f"[ImputeMissing] Starting transform. Initial shape: {X.shape}")
        # Drop columns
        X = X.drop(columns=self.drop_cols_, errors="ignore")
        self.logger.info(f"[ImputeMissing] Dropped {len(self.drop_cols_)} columns. Shape now: {X.shape}")

        # Numeric imputation
        if self.num_imputer_:
            X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])

        # Categorical imputation
        if self.cat_imputer_:
            X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])

        if self.verbose:
            self.logger.info(f"[ImputeMissing] Transform completed. Final shape: {X.shape}")

        self.feature_names_out = [col for col in X.columns if col not in self.drop_cols_]
        return X[self.feature_names_out]

    def _save_log(self):
        """Save imputation summary JSON."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log = {
            "missing_threshold": self.missing_threshold,
            "dropped_columns": self.drop_cols_,
            "fill_values_summary": {k: str(v) for k, v in self.fill_values_.items()}
        }
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=4)
        if self.verbose:
            self.logger.info(f"[ImputeMissing] Log saved to {self.log_path}")

    # Helper: for inspection
    def get_fill_values(self):
        """Return learned fill values per column."""
        return self.fill_values_
    
    def get_dropped_columns(self):
        """Return list of dropped columns."""
        return self.drop_cols_
