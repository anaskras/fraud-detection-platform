"""Merge raw CSV transaction + identity tables into Parquet via DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data.paths import resolve_path


def merge_data(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions in memory (use for small samples only)."""
    print(f"transactions shape: {transactions.shape}")
    print(f"identity shape: {identity.shape}")

    merged = transactions.merge(identity, on="TransactionID", how="left")

    print(f"merged shape: {merged.shape}")
    return merged


def merge_csv_to_parquet(
    transactions_csv: str | Path,
    identity_csv: str | Path,
    output_path: str | Path,
    *,
    compression: str = "zstd",
    memory_limit: str | None = "4GB",
) -> Path:
    """Left-join raw CSVs on TransactionID and write merged Parquet via DuckDB.

    Reads CSVs in a streaming pipeline without loading full tables into pandas.

    Args:
        transactions_csv: Path to ``{split}_transaction.csv``.
        identity_csv: Path to ``{split}_identity.csv``.
        output_path: Destination Parquet path.
        compression: Parquet codec (zstd, snappy, etc.).
        memory_limit: DuckDB memory cap; None for default.

    Returns:
        Path to the written Parquet file.
    """
    transactions_csv = Path(transactions_csv)
    identity_csv = Path(identity_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tx_sql = resolve_path(transactions_csv)
    id_sql = resolve_path(identity_csv)
    out_sql = resolve_path(output_path)
    codec = compression.upper()

    con = duckdb.connect()
    try:
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{tx_sql}', header = true) AS t
                LEFT JOIN read_csv_auto('{id_sql}', header = true) AS i
                USING (TransactionID)
            ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION {codec})
            """
        )

        row_count = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_sql}')"
        ).fetchone()[0]
    finally:
        con.close()

    print(
        f"{transactions_csv.name} + {identity_csv.name} -> {output_path.name}: "
        f"{row_count:,} rows"
    )
    return output_path


def merge_split(
    split: str,
    *,
    raw_dir: str | Path = "data/raw",
    merged_dir: str | Path = "data/merged",
    compression: str = "zstd",
    memory_limit: str | None = "4GB",
) -> Path:
    """Build ``data/merged/{split}_merged.parquet`` from raw CSVs for train or test.

    Args:
        split: ``train`` or ``test``.
        raw_dir: Directory with competition CSV files.
        merged_dir: Output directory for merged Parquet datasets.
        compression: Parquet codec passed to merge_csv_to_parquet.
        memory_limit: DuckDB memory cap.

    Returns:
        Path to ``{split}_merged.parquet``.
    """
    split = split.lower().strip()
    raw_dir = Path(raw_dir)
    merged_dir = Path(merged_dir)

    transactions_csv = raw_dir / f"{split}_transaction.csv"
    identity_csv = raw_dir / f"{split}_identity.csv"
    output_path = merged_dir / f"{split}_merged.parquet"

    return merge_csv_to_parquet(
        transactions_csv,
        identity_csv,
        output_path,
        compression=compression,
        memory_limit=memory_limit,
    )


def build_merged_datasets(
    *,
    raw_dir: str | Path = "data/raw",
    merged_dir: str | Path = "data/merged",
    splits: tuple[str, ...] = ("train", "test"),
    **kwargs,
) -> list[Path]:
    """Build merged Parquet files for all splits (default: train and test)."""
    return [merge_split(split, raw_dir=raw_dir, merged_dir=merged_dir, **kwargs) for split in splits]


if __name__ == "__main__":
    build_merged_datasets()
