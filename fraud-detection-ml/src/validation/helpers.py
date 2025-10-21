import hashlib
import json
import numpy as np
import pandas as pd

def hash_array(arr: np.ndarray) -> str:
    """Compute SHA256 hash for deterministic check."""
    if isinstance(arr, pd.DataFrame):
        # Ensure deterministic order of columns and rows
        arr_sorted = arr.sort_index(axis=1).sort_index(axis=0)
        # Convert to float64 bytes for stability
        data_bytes = arr_sorted.to_numpy(dtype=np.float64, copy=False).tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    elif isinstance(arr, np.ndarray):
        return hashlib.sha256(arr.tobytes()).hexdigest()

def assert_allclose(a, b, rtol=1e-7, atol=1e-9):
    """Wrapper with clearer failure message."""
    try:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError("❌ Non-deterministic transform detected") from e


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)