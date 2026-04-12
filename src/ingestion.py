from __future__ import annotations

import time
from typing import Any
from pathlib import Path

from altair import limit_rows
from pandas import DataFrame

from src.utils.logger import get_logger
import pandas as pd

import pyarrow.parquet as pq
import pyarrow.csv as p_csv

def ingest_csv_to_parquet(
    raw_dir: Path,
    output_path: Path,
    compression: str = "snappy",
    engine: str = "pyarrow",
    chunk_size_rows: int = 50_000,
    validate_schema: bool = True,
    required_columns: list[str] | None = None,
    skip_if_exists: bool = True,
    force: bool = False,
    logging_config: dict[str, Any] | None = None,

) -> Path:
    """
    Convert all CSV files in raw_dir into a single Parquet file.

    If multiple CSV files exist, they are concatenated in sorted order.
    All files must share the same schema (column names and types).

    Args:
        raw_dir:          Directory containing the raw CSV file(s).
        output_path:      Full path (including filename) for the Parquet output.
        compression:      Parquet codec: snappy | gzip | brotli | none.
        engine:           Parquet read/write engine: pyarrow | fastparquet.
        chunk_size_rows:  Rows per streaming batch (memory control).
        validate_schema:  If True, verify required_columns are present.
        required_columns: Column names that must be present post-conversion.
        skip_if_exists:   If True and output_path already exists, skip entirely.
        force:            Override skip_if_exists.
        logging_config:   Optional logging config dict.

    Returns:
        Path to the output Parquet file.

    Raises:
        FileNotFoundError:  If raw_dir contains no CSV files.
        ValueError:         If required columns are missing from the data.
        RuntimeError:       On read/write failures.
    """
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    log_cfg = logging_config
    logger = get_logger(__name__, log_cfg)

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    # ── Idempotency check ─────────────────────────────────────────────────
    if not force and skip_if_exists and output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info(
            "Ingested file already exists at '%s' (%.1f MB). Skipping ingest. "
            "(Use --force-ingest to re-ingest.)",
            output_path,
            size_mb,
        )
        return output_path

    # ── Discover source CSV files ──────────────────────────────────────────
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found in '{raw_dir}'. "
            "Run the download step first: python main.py --step download"
        )

    logger.info(
        "Ingesting %d CSV file(s) from '%s' -> '%s'",
        len(csv_files),
        raw_dir,
        output_path,
    )
    for cf in csv_files:
        size_mb = cf.stat().st_size / (1024 ** 2)
        logger.info("  [IN] %s (%.2f MB)", cf.name, size_mb)


    # ── Create output directory ────────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Open first file to read schema (header only) ──────────────────────
    read_options = pa_csv.ReadOptions(
        block_size=chunk_size_rows * 200,  # approximate bytes per block
        use_threads=True,
    )
    convert_options = pa_csv.ConvertOptions(
        auto_dict_encode=False,
        include_missing_columns=False,
    )
    parse_options = pa_csv.ParseOptions(
        delimiter=",",
        quote_char='"',
        double_quote=True,
        newlines_in_values=False,
    )

    start_time = time.monotonic()
    total_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for csv_path in csv_files:
            logger.info("  [READ] Processing '%s'...", csv_path.name)
            file_rows = 0

            with pa_csv.open_csv(
                csv_path,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            ) as reader:
                schema: pa.Schema = reader.schema
                logger.info("Schema: %d column(s) - %s", len(schema), schema.names)

                for batch in reader:
                    p_csv.write_csv(data=batch, output_file=csv_path.name)
                    output_csv = output_path / csv_path.name.replace('.csv', '.parquet')

                    # # Initialize writer on every file (schema drives the output)
                    # writer = pq.ParquetWriter(
                    #     output_csv,
                    #     schema=schema,
                    #     compression=compression,
                    #     writer_engine_version=engine,
                    #     use_dictionary=True,
                    #     write_page_index=True,
                    #     data_page_version='2.0',
                    #     store_schema=True,
                    # )
                    #
                    # writer.write_batch(batch)

                    csv = pd.read_csv(csv_path)
                    csv.to_parquet(path=output_csv, compression=compression)

                    file_rows += batch.num_rows
                    total_rows += batch.num_rows
                    logger.debug(
                        "  Batch written: %d rows | file total: %d | grand total: %d",
                        batch.num_rows,
                        file_rows,
                        total_rows,
                    )

            logger.info("  [OK]   '%s' - %d rows", csv_path.name, file_rows)

    finally:
        if writer is not None:
            writer.close()

    # ── Schema validation ──────────────────────────────────────────────────
    if validate_schema and required_columns:
        _validate_required_columns(output_path, required_columns, logger)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    input_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 ** 2)
    output_size_mb = output_path.stat().st_size / (1024 ** 2)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

    logger.info(
        "Ingest complete in %.1f s | rows=%d | csv=%.2f MB | parquet=%.2f MB "
        "| ratio=%.2fx | compression=%s",
        elapsed,
        total_rows,
        input_size_mb,
        output_size_mb,
        compression_ratio,
        compression,
    )

    return output_path

def load_datasets(expected_files: list, data_path: Path, logger: Any) -> dict:
    datasets = dict()
    logger.info(f"Loading datasets from '{data_path}'...")
    for _d in expected_files:
        logger.info(f"  [READ] Loading '{_d}'...")
        _dataset = pd.read_csv(data_path.as_uri() + '/' + _d)
        datasets[_d] = _dataset
        datasets.update({_d: _dataset})
        logger.info(f"  [OK] '{_d}'...")
    return datasets

def join_tables(dfs: dict, logger: Any) -> DataFrame:
    logger.info(f"  [JOIN] Joining between datasets: '{dfs.keys()}'")

    join_order_order_items = dfs.get('olist_orders_dataset.csv').set_index("order_id").join(dfs.get('olist_order_items_dataset.csv').set_index("order_id"), lsuffix="ord",
                                                                   rsuffix="itm", validate="many_to_many", how="inner")

    join_order_products = join_order_order_items.join(dfs.get('olist_products_dataset.csv').set_index("product_id"),
                                                      on='product_id', lsuffix="itm", rsuffix="prd",
                                                      validate="many_to_many", how="inner")

    join_order_order_items = join_order_products.join(dfs.get('olist_sellers_dataset.csv').set_index("seller_id"),
                                                 on='seller_id', lsuffix="prd", rsuffix="slr", validate="many_to_one",
                                                 how="inner")
    join_order_order_items = join_order_order_items.join(dfs.get('olist_customers_dataset.csv').set_index('customer_id'), on='customer_id',
                                                  rsuffix="cus", validate="many_to_one", how="inner")

    join_order_order_items = join_order_order_items[(join_order_order_items['order_status'] == 'delivered')]

    join_order_order_items.drop_duplicates(keep='last', inplace=True)

    logger.info(f"  [OK] Joining between datasets successful: '{join_order_order_items.columns}'...")

    return join_order_order_items

def _validate_required_columns(
    parquet_path: Path,
    required_columns: list[str],
    logger: Any,
) -> None:
    """
    Read only the Parquet footer metadata to verify required columns exist.

    This reads zero data rows — only the schema from the file footer.

    Raises:
        ValueError: If any required column is absent from the schema.
    """

    schema = pq.read_schema(str(parquet_path))
    actual_columns = set(schema.names)
    missing = [col for col in required_columns if col not in actual_columns]

    if missing:
        raise ValueError(
            f"Schema validation failed. Missing required column(s): {missing}\n"
            f"Actual columns: {sorted(actual_columns)}\n"
            "Fix: verify the dataset contains the expected columns, "
            "or update 'schema.required_columns' in config/data.yaml."
        )

    logger.info(
        "Schema validation OK - all %d required column(s) present.",
        len(required_columns),
    )

