from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.fs as pafs  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError

from .health_check import run_health_checks
from .input_source_tracer import trace_lazyframe_sources

Model = TypeVar("Model", bound=BaseModel)
DataFrame: TypeAlias = pl.DataFrame | pl.LazyFrame | pd.DataFrame


class Transform:
    """Class to represent a transformation input with an RID."""

    def __init__(self, rid: str):
        self.rid = rid

    def __repr__(self):
        return f"Transform('{self.rid}')"


# Custom exceptions
class PyrunnerError(Exception):
    """Base exception for pyrunner library."""


class ConfigurationError(PyrunnerError):
    """Raised when configuration files are invalid or missing."""


class TransformNotFoundError(PyrunnerError):
    """Raised when a transform ID is not found."""


class ModuleLoadError(PyrunnerError):
    """Raised when a transform module cannot be loaded."""


class DataLoadError(PyrunnerError):
    """Raised when input data cannot be loaded."""


class DataWriteError(PyrunnerError):
    """Raised when output data cannot be written."""


class DSMeta:
    """Metadata for a dataset with dot notation access."""

    def __init__(
        self, rid: str, base_path: str, ds_meta_data: dict[str, Any] | None = None
    ):
        self.transform_id = rid
        self.artifact_dir = f"{build_folder(rid, base_path)}/artifacts"
        # Needed to read artifact_dir when it is an s3:// path; None for local paths.
        self.storage_options = input_storage_options(rid)

        # Every field is None when DS_META has nothing for this dataset
        ds_meta_data = ds_meta_data or {}
        self.data_snapshot_id = ds_meta_data.get("Id")
        self.build_id = ds_meta_data.get("BuildId")
        self.row_count = ds_meta_data.get("RowCount")
        self.column_count = ds_meta_data.get("ColumnCount")
        self.file_size = ds_meta_data.get("FileSize")
        self.schema = ds_meta_data.get("Schema")
        self.creation_date = ds_meta_data.get("CreationDate")
        self.columns = self.schema["Columns"] if self.schema and "Columns" in self.schema else []


# Storage
#
# On DataSpace, builds read and write S3 (DS_BUILD_URI is set): the build's own folder with
# the credentials in the DS_S3_* variables, and each input build folder with the credentials
# in DS_INPUTS. Artifacts and meta are written to local folders and uploaded at the end.
# Without DS_BUILD_URI, everything lives below base_path/data/<transform id>/.
#
# The variables are not the AWS_* ones on purpose: those would also point the AWS clients
# of a transform's own code to the DataSpace storage.


def _s3_inputs() -> dict[str, dict[str, str]]:
    return json.loads(os.environ.get("DS_INPUTS") or "{}")


def _storage_options(credentials: dict[str, str] | None = None) -> dict[str, str]:
    """Polars storage options for the build's own folder, or for given input credentials."""
    if credentials is None:
        credentials = {
            "accessKeyId": os.environ["DS_S3_ACCESS_KEY_ID"],
            "secretAccessKey": os.environ["DS_S3_SECRET_ACCESS_KEY"],
            "sessionToken": os.environ["DS_S3_SESSION_TOKEN"],
        }
    endpoint = os.environ["DS_S3_ENDPOINT"]
    return {
        "aws_endpoint_url": endpoint,
        "aws_region": os.environ["DS_S3_REGION"],
        "aws_allow_http": str(endpoint.startswith("http://")).lower(),
        "aws_access_key_id": credentials["accessKeyId"],
        "aws_secret_access_key": credentials["secretAccessKey"],
        "aws_session_token": credentials["sessionToken"],
    }


def _s3_filesystem(storage_options: dict[str, str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(
        access_key=storage_options["aws_access_key_id"],
        secret_key=storage_options["aws_secret_access_key"],
        session_token=storage_options["aws_session_token"],
        endpoint_override=storage_options["aws_endpoint_url"],
        region=storage_options["aws_region"],
        allow_bucket_creation=False,
        check_directory_existence_before_creation=False,
    )


def build_folder(rid: str, base_path: str = "") -> str:
    """Folder of the build of a transform: this build's or an input's."""
    inputs = _s3_inputs()
    if rid in inputs:
        return inputs[rid]["uri"]
    if os.environ.get("DS_BUILD_URI") and rid == os.environ.get("TRANSFORM_ID"):
        return os.environ["DS_BUILD_URI"]
    return f"{base_path}/data/{rid}"


def storage_options(path: str) -> dict[str, str] | None:
    """Storage options to access an s3:// path of this build, its previous build or an input,
    for example a folder below ``ds_meta.artifact_dir``. None for local paths."""
    if not path.startswith("s3://"):
        return None
    for credentials in _s3_inputs().values():
        if path == credentials["uri"] or path.startswith(credentials["uri"] + "/"):
            return _storage_options(credentials)
    return _storage_options()


def input_storage_options(rid: str) -> dict[str, str] | None:
    """Storage options to read an input's build folder; None for local folders."""
    inputs = _s3_inputs()
    return _storage_options(inputs[rid]) if rid in inputs else None


def output_storage_options() -> dict[str, str] | None:
    """Storage options to write this build's folder; None for local folders."""
    return _storage_options() if os.environ.get("DS_BUILD_URI") else None


def upload_build_folders() -> None:
    """Upload the local artifact and meta folders to this build's S3 folder."""
    build_uri = os.environ.get("DS_BUILD_URI")
    if not build_uri:
        return
    fs = _s3_filesystem(_storage_options())
    for variable, name in (("ARTIFACT_FOLDER", "artifacts"), ("META_FOLDER", "meta")):
        local_folder = os.environ[variable]
        for root, _, files in os.walk(local_folder):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                relative = os.path.relpath(local_path, local_folder).replace(os.sep, "/")
                target = f"{build_uri.removeprefix('s3://')}/{name}/{relative}"
                with open(local_path, "rb") as src, fs.open_output_stream(target) as dst:
                    shutil.copyfileobj(src, dst)


# JSON Schema Models
class Config(BaseModel):
    python: str
    dfType: str
    packages: list[str]


class Connection(BaseModel):
    id: str
    path: str
    inputs: list[str]


# Load and validate JSON files
def load_json(file_path: str, model: type[Model]) -> Model | list[Model]:
    """Load and validate JSON files with proper error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            model(**data)
            if isinstance(data, dict)
            else [model(**item) for item in data]
        )
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {file_path}: {e}") from e
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration in {file_path}: {e}") from e


CONFIG_FILE = "_config.json"
CONNECTIONS_FILE = "_connections.json"

# Global configuration - will be loaded when needed
CONNECTIONS: list[Connection] | None = None
DF_TYPE: str | None = None
DEFAULT_DRAFT_READ_LIMIT = 1_000_000
NO_DRAFT_READ_LIMIT = -1
DS_META_DATA: dict[str, dict[str, Any]] | None = None


def load_ds_meta() -> None:
    """Load DS_META data from environment variable."""
    global DS_META_DATA
    try:
        ds_meta_json = os.environ.get("DS_META")
        if ds_meta_json:
            ds_meta_list = json.loads(ds_meta_json)
            # Convert list to dict keyed by TransformId for easy lookup
            DS_META_DATA = {item["TransformId"]: item for item in ds_meta_list}
        else:
            DS_META_DATA = {}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(
            f"Warning: Failed to parse DS_META environment variable: {e}",
            file=sys.stderr,
        )
        DS_META_DATA = {}


def load_configuration() -> None:
    """Load configuration files with error handling."""
    global CONNECTIONS, DF_TYPE
    config = load_json(CONFIG_FILE, Config)
    connections = load_json(CONNECTIONS_FILE, Connection)

    if not isinstance(config, Config):
        raise ConfigurationError(
            f"Invalid configuration in {CONFIG_FILE}: expected an object"
        )
    if not isinstance(connections, list):
        raise ConfigurationError(
            f"Invalid configuration in {CONNECTIONS_FILE}: expected a list"
        )

    CONNECTIONS = connections
    DF_TYPE = config.dfType


# Utility functions
def write_column_trace_file(
    lf: pl.LazyFrame, transform_id: str, base_path: str
) -> None:
    """Write column trace file with proper error handling."""
    try:
        # S3 build folders are named after the build, so the scan paths do not name the
        # transform: the inputs', and this transform's own previous dataset
        source_names = {
            f"{build_folder(rid)}/datasets/data.parquet": rid for rid in _s3_inputs()
        }
        if os.environ.get("DS_BUILD_URI") and os.environ.get("PREVIOUS_DATASET_PATH"):
            source_names[os.environ["PREVIOUS_DATASET_PATH"]] = transform_id
        origins = trace_lazyframe_sources(lf, source_names, storage_options)

        # On S3 builds meta is written to the local META_FOLDER and uploaded later.
        if os.environ.get("DS_BUILD_URI"):
            meta_folder = os.environ["META_FOLDER"]
        else:
            meta_folder = f"{base_path}/data/{transform_id}/meta"
        trace_file_path = f"{meta_folder}/columns.json"
        os.makedirs(os.path.dirname(trace_file_path), exist_ok=True)
        with open(trace_file_path, "w", encoding="utf-8") as f:
            json.dump(origins, f, indent=4)
    except Exception as e:  # noqa: BLE001 - lineage metadata is best-effort
        # Don't raise an error here since this is not a critical error
        print(f"Failed to write column trace file: {e}", file=sys.stderr)


def write_df_to_parquet(
    df: DataFrame,
    file_path: str,
    transform_id: str,
    base_path: str,
) -> None:
    """
    Writes a DataFrame to a Parquet file with automatic type detection.

    Args:
        df: Can be a Polars DataFrame, Polars LazyFrame, or Pandas DataFrame
        file_path: Path where to save the parquet file
        transform_id: ID of the transform for tracing
        base_path: Base path for the project
    """
    try:
        storage_options = output_storage_options()
        if storage_options is None:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check dataframe type and handle accordingly
        if isinstance(df, pl.LazyFrame):
            # If it's a LazyFrame, collect it first
            write_column_trace_file(df, transform_id, base_path)
            df.sink_parquet(file_path, storage_options=storage_options)
        elif isinstance(df, pl.DataFrame):
            # If it's a Polars DataFrame, write directly
            df.write_parquet(file_path, storage_options=storage_options)
        elif isinstance(df, pd.DataFrame):
            # If it's a Pandas DataFrame, use pyarrow
            table = pa.Table.from_pandas(df, preserve_index=False)
            if storage_options is None:
                pq.write_table(table, file_path)
            else:
                pq.write_table(
                    table,
                    file_path.removeprefix("s3://"),
                    filesystem=_s3_filesystem(storage_options),
                )
        else:
            raise TypeError(f"Unsupported dataframe type: {type(df)}")

    except Exception as e:
        raise DataWriteError(f"Failed to write Parquet file: {e}") from e


def path_to_name(path: str) -> str:
    """Extracts the filename (without extension) from a given path."""
    return Path(path).stem


def load_module(module_name: str, file_path: str) -> Any:
    """Dynamically loads a module from a given file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ModuleLoadError(f"Could not create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ModuleLoadError(f"Failed to load module from {file_path}: {e}") from e


def transform(*args: Any, **kwargs: Any) -> Any:
    """
    Acts as both a decorator and the main runner function for transformations.

    Decorator usage:
        @transform
        def my_func(input): ...

        @transform(input=Transform("rid-xxx"))
        def my_func(input): ...

    Runner usage:
        transform("transform_id", base_path="")
    """
    if len(args) == 1 and callable(args[0]):
        # Case: @transform
        func = args[0]
        func._is_pyrunner_transform = True
        return func
    if len(args) == 0 and kwargs:
        # Case: @transform(input=Transform("..."))
        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            func._is_pyrunner_transform = True  # type: ignore[attr-defined]
            func._pyrunner_metadata = kwargs  # type: ignore[attr-defined]
            return func

        return wrapper

    # Case: transform("id", "path") - Runner mode
    return _execute_transform(*args, **kwargs)


# Mark as the runner function to avoid picking it up during discovery
transform._is_pyrunner_runner = True  # type: ignore[attr-defined]


def _annotated_transform(annotation: Any, func: Callable[..., Any]) -> Transform | None:
    """The Transform a parameter annotation names, as DataSpace reads it from the source."""
    if isinstance(annotation, str):
        # from __future__ import annotations keeps every annotation as its source text. Only its
        # Transform(...) call is evaluated, the part DataSpace reads, so a frame type imported only
        # for type checking does not have to resolve. It is the module's own, already loaded code.
        call = next(
            (
                node
                for node in ast.walk(ast.parse(annotation, mode="eval"))
                if isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "Transform")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "Transform")
                )
            ),
            None,
        )
        if call is None:
            return None
        annotation = eval(compile(ast.Expression(call), "<annotation>", "eval"), func.__globals__)
    if isinstance(annotation, Transform):
        return annotation
    # Annotated[pl.LazyFrame, Transform("rid...")]
    return next(
        (meta for meta in getattr(annotation, "__metadata__", ()) if isinstance(meta, Transform)),
        None,
    )


def _execute_transform(transform_id: str, base_path: str = "") -> None:
    """Internal function to execute the transformation process."""
    try:
        _run_transform(transform_id, base_path)
    except BaseException:
        # Keep the artifacts and health report of failed builds for debugging.
        try:
            upload_build_folders()
        except Exception as e:  # noqa: BLE001 - the build error is the one to report
            print(f"Failed to upload artifacts: {e}", file=sys.stderr)
        raise

    try:
        upload_start = time.perf_counter()
        upload_build_folders()
    except Exception as e:
        raise DataWriteError(f"Failed to upload artifacts: {e}") from e
    if os.environ.get("DS_BUILD_URI"):
        print(f"Upload Time: {time.perf_counter() - upload_start:.2f} sec")

    print("--------- Build successful ---------")


def _run_transform(transform_id: str, base_path: str) -> None:
    # Load configuration and DS_META data
    load_configuration()
    load_ds_meta()

    if CONNECTIONS is None:
        raise ConfigurationError("Connections not loaded")

    connections = {path_to_name(conn.path): conn for conn in CONNECTIONS}
    transform_conn = next((c for c in CONNECTIONS if c.id == transform_id), None)

    if not transform_conn:
        raise TransformNotFoundError(
            f"Transform with ID '{transform_id}' not found."
        )

    module_name = path_to_name(transform_conn.path)
    module = load_module(module_name, transform_conn.path)

    # Find the transform function
    # 1. Look for a function named 'transform' (that isn't our own runner)
    # 2. Look for functions decorated with @transform
    main_func: Callable[..., Any] | None = None

    if (
        hasattr(module, "transform")
        and callable(module.transform)
        and not getattr(module.transform, "_is_pyrunner_runner", False)
    ):
        main_func = module.transform
    else:
        # Prefer decorated functions
        decorated_funcs = [
            obj
            for _, obj in inspect.getmembers(module)
            if callable(obj)
            and getattr(obj, "_is_pyrunner_transform", False) is True
        ]
        if decorated_funcs:
            # If multiple, pick the first one found.
            main_func = decorated_funcs[0]

    if not main_func:
        raise ModuleLoadError(
            f"Module {transform_conn.path} does not contain a 'transform' function or @transform decorated function."
        )

    # Inspect function signature
    sig = inspect.signature(main_func)
    params = sig.parameters

    # Determine mapping for each parameter
    metadata: dict[str, Any] = getattr(main_func, "_pyrunner_metadata", {})
    param_mapping: dict[str, str] = {}  # parameter_name -> RID or legacy_name
    explicit_params: set[str] = set()

    for name, param in params.items():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Check decorator metadata first
        if name in metadata:
            val = metadata[name]
            if isinstance(val, Transform):
                param_mapping[name] = val.rid
                explicit_params.add(name)
            else:
                param_mapping[name] = (
                    val  # Assume it's an RID string or legacy name
                )
        # Check type annotation
        elif (annotated := _annotated_transform(param.annotation, main_func)) is not None:
            param_mapping[name] = annotated.rid
            explicit_params.add(name)
        else:
            # Default to parameter name
            param_mapping[name] = name

    # A lazy query only runs when its output is written, so reading, transforming
    # and writing are timed together; health checks run afterwards on the written
    # file and get their own timer
    start_time = time.perf_counter()

    # Read input data
    data_dict = read_parquet_files(
        connections, param_mapping, base_path, transform_id, explicit_params
    )

    # Execute transformation
    transform_result = main_func(**data_dict)

    # Handle optional health_checks return
    if isinstance(transform_result, tuple) and len(transform_result) == 2:
        result, health_checks = transform_result
    else:
        result = transform_result
        health_checks = None

    # Write output
    output_path = f"{build_folder(transform_conn.id, base_path)}/datasets/data.parquet"
    write_df_to_parquet(result, output_path, transform_conn.id, base_path)
    print(f"Execution Time: {time.perf_counter() - start_time:.2f} sec")

    # Checks run on the written file: re-running the transform for every check is
    # slower, and its row order may differ from the file's
    if health_checks:
        checks_start = time.perf_counter()
        run_health_checks(output_path, health_checks, output_storage_options())
        print(f"Health Check Time: {time.perf_counter() - checks_start:.2f} sec")


def _read_parquet_head(
    file_path: str, limit: int, filesystem: pafs.FileSystem | None
) -> pd.DataFrame:
    """The first `limit` rows as pandas, read batch by batch instead of loading the whole file."""
    parquet = pq.ParquetFile(file_path, filesystem=filesystem)
    batches = []
    remaining = limit
    for batch in parquet.iter_batches(batch_size=min(limit, 65_536)):
        batches.append(batch.slice(0, remaining))
        remaining -= batches[-1].num_rows
        if remaining == 0:
            break
    return pa.Table.from_batches(batches, schema=parquet.schema_arrow).to_pandas()


def read_parquet_files(
    connections: dict[str, Connection],
    param_mapping: dict[str, str],
    base_path: str,
    transform_id: str | None = None,
    explicit_params: set[str] | None = None,
) -> dict[str, pl.LazyFrame | pd.DataFrame]:
    """Reads input Parquet files based on parameter mapping (RID or legacy name)."""
    if DF_TYPE is None:
        raise ConfigurationError(
            "Configuration not loaded. Call load_configuration() first."
        )

    data_dict: dict[str, pl.LazyFrame | pd.DataFrame] = {}

    # Create internal lookup for connections by ID (RID) and by legacy name (path stem)
    conn_by_id = {conn.id: conn for conn in connections.values()}
    conn_by_name = connections  # Already keyed by path stem

    # A draft build reads only the first rows of each input; a workspace can change how many, or turn the limit off
    # with -1 (_config.json draftReadLimit)
    draft_read_limit = None
    if os.environ.get("IS_DRAFT_BUILD", "").lower() == "true":
        raw_limit = os.environ.get("DRAFT_READ_LIMIT", str(DEFAULT_DRAFT_READ_LIMIT))
        try:
            draft_read_limit = int(raw_limit)
        except ValueError as e:
            raise ConfigurationError(
                f"DRAFT_READ_LIMIT must be a whole number of rows, got {raw_limit!r}."
            ) from e
    if draft_read_limit == NO_DRAFT_READ_LIMIT:
        draft_read_limit = None
    # pandas reads head(-n) as "all but the last n rows", so any other non-positive limit must not get that far
    elif draft_read_limit is not None and draft_read_limit < 1:
        raise ConfigurationError(
            f"DRAFT_READ_LIMIT must be a positive number of rows or {NO_DRAFT_READ_LIMIT} for no limit, got {draft_read_limit}."
        )

    for param_name, mapping_key in param_mapping.items():
        # Try finding by ID first (RID), then by name
        conn = conn_by_id.get(mapping_key) or conn_by_name.get(mapping_key)

        if not conn:
            # Fallback to mapping_key as RID only if it was explicitly provided via Transform()
            is_explicit = explicit_params is not None and param_name in explicit_params
            if not is_explicit:
                raise DataLoadError(
                    f"Input parameter '{param_name}' not found in dataset connections"
                )
            rid = mapping_key
        else:
            rid = conn.id

        file_path = f"{build_folder(rid, base_path)}/datasets/data.parquet"
        storage_options = input_storage_options(rid)
        try:
            if os.environ.get("DS_BUILD_URI") and storage_options is None:
                raise DataLoadError(
                    f"No access to the data of '{rid}'; is it an input of this transform?"
                )
            df: pl.LazyFrame | pd.DataFrame
            if DF_TYPE == "polars":
                df = pl.scan_parquet(file_path, storage_options=storage_options)
                if draft_read_limit is not None:
                    df = df.head(draft_read_limit)  # lazy: only these rows are ever read
            else:
                # pyarrow takes the S3 path without its scheme, together with the filesystem
                filesystem = _s3_filesystem(storage_options) if storage_options else None
                pandas_path = file_path.removeprefix("s3://")
                if draft_read_limit is not None:
                    df = _read_parquet_head(pandas_path, draft_read_limit, filesystem)
                else:
                    df = pd.read_parquet(pandas_path, engine="pyarrow", filesystem=filesystem)

            # Inject ds_meta attribute into the dataframe
            if transform_id is not None:
                # Get DS_META data for this RID (data snapshot)
                ds_meta_data = DS_META_DATA.get(rid) if DS_META_DATA else None
                df.ds_meta = DSMeta(rid, base_path, ds_meta_data)  # type: ignore[union-attr]

            data_dict[param_name] = df
        except Exception as e:
            raise DataLoadError(
                f"Failed to load {param_name}. Ensure the input dataset exists at {file_path}: {e}"
            ) from e
    return data_dict
