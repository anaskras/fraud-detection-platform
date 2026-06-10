import numpy as np
import pandas as pd


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "TransactionAmt" in out.columns:
        out["TransactionAmt_log1p"] = np.log1p(out["TransactionAmt"].clip(lower=0))

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(-1)

    object_cols = out.select_dtypes(include=["object", "string"]).columns
    out[object_cols] = out[object_cols].fillna("missing")

    return out
