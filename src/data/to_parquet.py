"""Convert large CSV files to Parquet using DuckDB (low memory, no full DataFrame)."""

from __future__ import annotations

from pathlib import Path

import duckdb


def _resolve_path(path: str | Path) -> str:
    """Return an absolute path string safe for DuckDB SQL (forward slashes)."""
    return Path(path).resolve().as_posix()


def csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    *,
    compression: str = "zstd",
    memory_limit: str | None = "4GB",
) -> None:
    """Convert one CSV to Parquet via DuckDB streaming COPY.

    DuckDB reads and writes in a streaming pipeline; the file is never materialized
    as a single in-memory pandas DataFrame.

    Args:
        csv_path: Source CSV path.
        parquet_path: Destination Parquet path (parent dirs are created).
        compression: Parquet codec (e.g. zstd, snappy, gzip, uncompressed).
        memory_limit: DuckDB memory cap (e.g. "2GB"); None to use DuckDB default.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    csv_sql = _resolve_path(csv_path)
    parquet_sql = _resolve_path(parquet_path)
    codec = compression.upper()

    con = duckdb.connect()
    try:
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")

        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv_auto('{csv_sql}', header = true)
            ) TO '{parquet_sql}' (FORMAT PARQUET, COMPRESSION {codec})
            """
        )

        row_count = con.execute(
            f"SELECT count(*) FROM read_parquet('{parquet_sql}')"
        ).fetchone()[0]
    finally:
        con.close()

    print(f"{csv_path.name} -> {parquet_path.name}: {row_count:,} rows")


def convert_raw_dir(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
    *,
    pattern: str = "*.csv",
    compression: str = "zstd",
    memory_limit: str | None = "4GB",
) -> list[Path]:
    """Convert every CSV in raw_dir to a Parquet file in out_dir.

    Args:
        raw_dir: Directory containing source CSVs.
        out_dir: Directory for output Parquet files (created if missing).
        pattern: Glob for files to convert (default: all CSVs).
        compression: Parquet codec passed to csv_to_parquet.
        memory_limit: DuckDB memory cap per file conversion.

    Returns:
        List of written Parquet paths.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for csv_path in sorted(raw_dir.glob(pattern)):
        parquet_path = out_dir / f"{csv_path.stem}.parquet"
        csv_to_parquet(
            csv_path,
            parquet_path,
            compression=compression,
            memory_limit=memory_limit,
        )
        written.append(parquet_path)
    return written


if __name__ == "__main__":
    convert_raw_dir()
