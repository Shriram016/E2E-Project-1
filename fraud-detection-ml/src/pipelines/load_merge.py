# src/pipelines/load_merge.py
import pandas as pd

import test
from src.utils.logger import get_logger

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("load_merge")

def load_and_merge(
    sources: dict,
    join_key: str = "TransactionID",
    join_type: str = "left",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame|None:
    """
    Loads and merges two (or more) data sources into one DataFrame.

    Parameters
    ----------
    sources : dict
        Dict mapping names to file paths, e.g.
        {"transaction": "path/train_transaction.csv",
         "identity": "path/train_identity.csv"}
    join_key : str
        Column to join on (e.g. "TransactionID").
    how : str
        Join type ('left', 'inner', etc.)
    validate : bool
        Whether to validate join key coverage and duplicates.
    verbose : bool
        Whether to log progress.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    if not isinstance(sources, dict):
        raise ValueError("sources must be a dict of {name: path}")

    dfs = {}
    for name, path in sources.items():
        if verbose:
            logger.info(f"[load_and_merge] Loading {name} from {path}")
        dfs[name] = _load_data(path)

    # Validate join key existence
    for name, df in dfs.items():
        if join_key not in df.columns:
            raise KeyError(f"[load_and_merge] {join_key} not found in {name}")

    # Merge sequentially
    merged = pd.DataFrame()
    for i, (name, df) in enumerate(dfs.items()):
        logger.info(f"[load_and_merge] Merging {name} with shape {df.shape}")
        if i == 0:
            merged = df
        else:
            if merged is None:
                raise ValueError("Merged DataFrame is None during merging.")
            merged = merged.merge(df, on=join_key, how=join_type) # type: ignore

    if verbose:
        logger.info(f"[load_and_merge] Final merged shape: {merged.shape}")

    # Optional validation
    if validate:
        _validate_merge(left=dfs["transaction"],
                        right=dfs["identity"],
                        merged=merged,
                        join_key=join_key,)

    return merged


def _load_data(path: str) -> pd.DataFrame:
    """Load parquet or CSV depending on extension."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")




def _validate_merge(left, right, merged, join_key, verbose=True):
    """
    Minimal merge validation:
      ✅ Checks % of left rows that matched
      ✅ Checks duplicate join keys
      ✅ Logs NaN ratio in right-side columns
    """
    # 1️⃣ True join coverage
    match_rate = (
        left[[join_key]]
        .merge(right[[join_key]].drop_duplicates(), on=join_key, how="left", indicator=True)
        ["_merge"]
        .eq("both")
        .mean()
    )

    # 2️⃣ Duplicate key check
    dupes = merged[join_key].duplicated().sum()

    # 3️⃣ Right-side NaN ratio (rough health check)
    right_cols = [c for c in right.columns if c != join_key]
    nan_ratio = merged[right_cols].isna().mean().mean() if right_cols else 0

    if verbose:
        logger.info(f"[load_and_merge] match_rate={match_rate:.2%}, dupes={dupes}, right_NaN={nan_ratio:.2%}")
        if match_rate < 0.2:
            logger.warning("[load_and_merge] Low match rate — check join key.")
        if dupes > 0:
            logger.warning(f"[load_and_merge] {dupes} duplicate {join_key} values detected.")

    return match_rate, dupes, nan_ratio
