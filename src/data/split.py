import pandas as pd


def time_split(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    train_size: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_size)

    train_df = sorted_df.iloc[:split_idx].copy()
    val_df = sorted_df.iloc[split_idx:].copy()
    return train_df, val_df
