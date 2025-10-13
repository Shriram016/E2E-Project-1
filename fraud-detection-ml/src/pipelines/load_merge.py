# src/pipelines/load_merge.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union

class LoadMerge(BaseEstimator, TransformerMixin):
    """
    Loads and merges multiple data sources (transaction, identity, etc.)
    into a single unified DataFrame.

    Fits and transforms like an sklearn transformer:
        .fit(X, y) -> learns nothing (stateless)
        .transform(X) -> loads and merges sources

    Parameters
    ----------
    sources : dict
        Dictionary mapping logical names to file paths, e.g.
        {"transaction": "data/raw/train_transaction.parquet",
         "identity": "data/raw/train_identity.parquet"}
    join_key : str
        Column name to join on (e.g., "TransactionID").
    how : str
        Join type ('left', 'inner', etc.)
    validate : bool
        Whether to validate key uniqueness.
    """

    def __init__(
        self,
        sources: dict,
        join_key: str = "TransactionID",
        how: str = "left",
        validate: bool = True,
        verbose: bool = True,
    ):
        self.sources = sources
        self.join_key = join_key
        self.how = how
        self.validate = validate
        self.verbose = verbose
        self.merged_columns_ = None

    def fit(self, X=None, y=None):
        """Stateless transformer: nothing to learn."""
        return self

    def transform(self, X=None):
        """Loads and merges all configured sources."""
        if not isinstance(self.sources, dict):
            raise ValueError("sources must be a dict of {name: path}")

        dfs = {}
        for name, path in self.sources.items():
            if self.verbose:
                print(f"[LoadMerge] Loading {name} from {path}")
            dfs[name] = self._load_data(path)

        # Validate join key existence
        for name, df in dfs.items():
            if self.join_key not in df.columns:
                raise KeyError(f"[LoadMerge] {self.join_key} not found in {name}")

        # Sequentially merge all dataframes
        merged = None
        for i, (name, df) in enumerate(dfs.items()):
            if i == 0:
                merged = df
            else:
                merged = merged.merge(df, on=self.join_key, how=self.how)

        if self.validate:
            self._validate_merge(dfs, merged)

        self.merged_columns_ = merged.columns.tolist()

        if self.verbose:
            print(f"[LoadMerge] Final merged shape: {merged.shape}")

        return merged

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load parquet or CSV depending on extension."""
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _validate_merge(self, dfs: dict, merged: pd.DataFrame):
        """Optional validation checks after merge."""
        left_name = list(dfs.keys())[0]
        left_df = dfs[left_name]
        left_keys = set(left_df[self.join_key])
        merged_keys = set(merged[self.join_key])

        coverage = len(merged_keys) / len(left_keys) if left_keys else 0
        if self.verbose:
            print(f"[LoadMerge] Join coverage: {coverage:.2%}")

        if coverage < 0.95:
            print(f"[LoadMerge] WARNING: join coverage below 95%")

        # Duplicate key check
        dupes = merged[self.join_key].duplicated().sum()
        if dupes > 0:
            print(f"[LoadMerge] WARNING: {dupes} duplicate {self.join_key} values found.")

