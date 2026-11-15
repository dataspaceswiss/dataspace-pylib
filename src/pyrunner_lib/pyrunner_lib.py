from __future__ import annotations

import ast
import fnmatch
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as pads  # type: ignore[import-untyped]
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

    def __init__(self, rid: str, ds_meta_data: dict[str, Any] | None = None):
        self.transform_id = rid

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


def _s3_filesystem(
    storage_options: dict[str, str], request_timeout: float | None = None
) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(
        access_key=storage_options["aws_access_key_id"],
        secret_key=storage_options["aws_secret_access_key"],
        session_token=storage_options["aws_session_token"],
        endpoint_override=storage_options["aws_endpoint_url"],
        region=storage_options["aws_region"],
        request_timeout=request_timeout,
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


def _filesystem(path: str) -> tuple[pafs.FileSystem, str]:
    """The filesystem of a path of this build, its previous build or an input, and the path in it."""
    options = storage_options(path)
    if options is None:
        return pafs.LocalFileSystem(), path
    return _s3_filesystem(options), path.removeprefix("s3://")


def _dataset_files(dataset_folder: str) -> list[str]:
    """The Parquet files of a dataset folder, relative to it."""
    fs, folder = _filesystem(dataset_folder)
    return sorted(
        info.path.removeprefix(folder + "/")
        for info in fs.get_file_info(pafs.FileSelector(folder, recursive=True, allow_not_found=True))
        if info.type == pafs.FileType.File and info.path.endswith(".parquet")
    )


# The API transforms use


class Artifacts:
    """The artifact folder of an input, or of the previous build. Nothing is fetched until asked for."""

    def __init__(self, folder: str):
        self._folder = folder
        self._local: dict[str, Path] = {}

    def uri(self, path: str = "") -> str:
        """Where a file or folder is stored, to read it in place together with ``storage_options``:
        ``pl.scan_csv(artifacts.uri("big.csv"), storage_options=artifacts.storage_options)``."""
        return f"{self._folder}/{path}".rstrip("/")

    @property
    def storage_options(self) -> dict[str, str] | None:
        """The Polars storage options with the credentials to read this folder; None for a local folder."""
        return storage_options(self._folder)

    def list(self, pattern: str = "*") -> list[str]:
        """The files below the folder whose path matches the pattern, as paths relative to the folder.
        ``*`` matches any characters, also ``/``: ``list("raw/*.csv")``."""
        fs, folder = _filesystem(self._folder)
        selector = pafs.FileSelector(folder, recursive=True, allow_not_found=True)
        files = (
            info.path.removeprefix(folder + "/")
            for info in fs.get_file_info(selector)
            if info.type == pafs.FileType.File
        )
        return sorted(file for file in files if fnmatch.fnmatchcase(file, pattern))

    def get(self, path: str = "") -> Path:
        """A local copy of a file or folder, for code that needs local files; the whole artifact folder
        without a path. Fetched on the first call and reused afterwards."""
        path = path.strip("/")
        if path not in self._local:
            fs, source = _filesystem(self.uri(path))
            source_type = fs.get_file_info(source).type
            if source_type == pafs.FileType.NotFound:
                raise FileNotFoundError(f"There is no artifact '{path}' in {self._folder}")
            if isinstance(fs, pafs.LocalFileSystem):
                self._local[path] = Path(source)
            else:
                target = os.path.join(tempfile.mkdtemp(prefix="ds-artifacts-"), os.path.basename(source))
                if source_type == pafs.FileType.Directory:
                    # copy_files only creates the subfolders
                    os.makedirs(target)
                pafs.copy_files(
                    source, target, source_filesystem=fs, destination_filesystem=pafs.LocalFileSystem()
                )
                self._local[path] = Path(target)
        return self._local[path]


class Input(pl.LazyFrame):
    """The type of an input, for code completion: ``def transform(orders: Input)``. An input is the
    LazyFrame of the upstream dataset with these two attributes; the class itself is never created."""

    artifacts: Artifacts
    ds_meta: DSMeta


class Append:
    """What ``Previous.append`` returns: the rows to add to the files of the previous dataset."""

    def __init__(self, frame: pl.LazyFrame | pl.DataFrame):
        if not isinstance(frame, pl.LazyFrame | pl.DataFrame):
            raise TypeError(f"append needs a Polars frame, not {type(frame).__name__}")
        self.frame = frame


class Previous:
    """The last successful committed build of this transform on this branch. A parameter with this
    type makes a transform incremental: ``def transform(orders, history: Previous)``."""

    def __init__(self, transform_id: str):
        previous = json.loads(os.environ["DS_PREVIOUS"]) if os.environ.get("DS_PREVIOUS") else None
        self.exists: bool = previous is not None
        self._transform_id = transform_id
        self._uri: str | None = previous["uri"] if previous else None
        self._meta = previous["meta"] if previous else None
        self._artifacts: Artifacts | None = None

    def _folder(self, name: str) -> str:
        if self._uri is None:
            raise PyrunnerError(
                "This transform has no previous build yet; check `exists` before using it."
            )
        return f"{self._uri}/{name}"

    @property
    def data(self) -> pl.LazyFrame | pd.DataFrame:
        """The previous dataset."""
        return _read_dataset(self._folder(DATASET_FOLDER_NAME), _draft_read_limit())

    @property
    def artifacts(self) -> Artifacts:
        """The artifacts of the previous build."""
        if self._artifacts is None:
            self._artifacts = Artifacts(self._folder("artifacts"))
        return self._artifacts

    @property
    def ds_meta(self) -> DSMeta:
        """Row count, schema and the other metadata of the previous dataset."""
        self._folder("")
        return DSMeta(self._transform_id, self._meta)

    def append(self, frame: pl.LazyFrame | pl.DataFrame) -> Append:
        """Return this from the transform to keep the files of the previous dataset as they are and add
        the frame's rows as new files. The previous rows are neither read nor written again, so the build
        only costs what the new rows cost. The frame must have the columns and types of the previous
        dataset. On the first build the frame is the dataset."""
        return Append(frame)


class Storage:
    """Direct access to the S3 folder of the running build, for what the rest of the API does not cover."""

    @property
    def uri(self) -> str:
        """The build's folder, ``s3://bucket/builds/<build id>``."""
        return os.environ["DS_BUILD_URI"]

    @property
    def options(self) -> dict[str, str]:
        """Polars storage options with the build's temporary credentials: they write the build's own
        folder and read its previous build."""
        return _storage_options()

    def filesystem(self) -> pafs.S3FileSystem:
        """A pyarrow filesystem with the same credentials; its paths are the uri without ``s3://``."""
        return _s3_filesystem(_storage_options())


class Build:
    """The running build: ``from pyrunner_lib import build``."""

    storage = Storage()

    @property
    def id(self) -> str:
        return os.environ["DS_BUILD_ID"]

    @property
    def transform_id(self) -> str:
        return os.environ["TRANSFORM_ID"]

    @property
    def branch(self) -> str:
        return os.environ["DS_BRANCH"]

    @property
    def is_draft(self) -> bool:
        """True for a build of code that is not committed; it reads only the first rows of its inputs."""
        return os.environ.get("IS_DRAFT_BUILD", "").lower() == "true"

    @property
    def artifacts(self) -> Path:
        """The local folder for the build's artifacts. What is in it when the transform ends is stored
        with the build, also after a failure."""
        folder = Path(os.environ["ARTIFACT_FOLDER"])
        folder.mkdir(parents=True, exist_ok=True)
        return folder


build = Build()


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
# A dataset is every Parquet file below the datasets folder of a build: one file or many, also in subfolders
DATASET_FOLDER_NAME = "datasets"
DATASET_GLOB = "**/*.parquet"
# The rows of each append are written into a folder of their own below the dataset folder
APPEND_FOLDER_PREFIX = "append-"
# The storage answers a copy only when it is done, and pyarrow gives up on a silent request after 3 seconds,
# which a file of about 1 GiB already takes. An hour covers a single file of several hundred GiB.
COPY_REQUEST_TIMEOUT_SECONDS = 3600
# An output is split into files of about this size, measured as the size of the data in memory
DATASET_PART_BYTES = 1024**3
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
            f"{build_folder(rid)}/{DATASET_FOLDER_NAME}/": rid for rid in _s3_inputs()
        }
        if os.environ.get("DS_BUILD_URI") and os.environ.get("DS_PREVIOUS"):
            previous = json.loads(os.environ["DS_PREVIOUS"])["uri"]
            source_names[f"{previous}/{DATASET_FOLDER_NAME}/"] = transform_id
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
    dataset_folder: str,
    transform_id: str,
    base_path: str,
    traced: pl.LazyFrame | None = None,
) -> None:
    """
    Writes a DataFrame as the Parquet files of a dataset folder.

    Args:
        df: Can be a Polars DataFrame, Polars LazyFrame, or Pandas DataFrame
        dataset_folder: Folder that receives the dataset's Parquet files
        transform_id: ID of the transform for tracing
        base_path: Base path for the project
        traced: The query the column lineage is read from when it is not df itself
    """
    try:
        storage_options = output_storage_options()
        if storage_options is None:
            # A local folder is written again by every run, and files of an earlier run would be read as
            # part of the dataset. A build folder in S3 is new.
            shutil.rmtree(dataset_folder, ignore_errors=True)
            os.makedirs(dataset_folder)

        if isinstance(df, pl.LazyFrame):
            write_column_trace_file(traced if traced is not None else df, transform_id, base_path)
        if isinstance(df, pl.LazyFrame | pl.DataFrame):
            # Polars splits the output into files and streams them out, so no file grows without bound
            df.lazy().sink_parquet(
                pl.PartitionBy(dataset_folder, approximate_bytes_per_file=DATASET_PART_BYTES),
                storage_options=storage_options,
            )
        elif isinstance(df, pd.DataFrame):
            # A pandas DataFrame is in memory as a whole, so it is written as one file
            table = pa.Table.from_pandas(df, preserve_index=False)
            file_path = f"{dataset_folder}/00000000.parquet"
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


def write_append(
    frame: pl.LazyFrame | pl.DataFrame,
    previous: Previous,
    dataset_folder: str,
    transform_id: str,
    base_path: str,
) -> None:
    """Writes a dataset as the files of the previous dataset plus the frame's rows as new files."""
    if not previous.exists:
        write_df_to_parquet(frame, dataset_folder, transform_id, base_path)
        return

    previous_folder = previous._folder(DATASET_FOLDER_NAME)
    previous_data = pl.scan_parquet(
        f"{previous_folder}/{DATASET_GLOB}", storage_options=storage_options(previous_folder)
    )
    previous_schema, schema = previous_data.collect_schema(), frame.lazy().collect_schema()
    if schema != previous_schema:
        raise DataWriteError(
            "The appended rows must have the columns and types of the previous dataset.\n"
            f"Previous dataset: {dict(previous_schema)}\nAppended rows: {dict(schema)}"
        )

    try:
        # The previous files are copied as they are; in S3 the storage copies them, they do not pass
        # through the build
        source_fs, source = _filesystem(previous_folder)
        files = _dataset_files(previous_folder)
        if isinstance(source_fs, pafs.LocalFileSystem):
            shutil.rmtree(dataset_folder, ignore_errors=True)
            target_fs, target = source_fs, dataset_folder
        else:
            target_fs = _s3_filesystem(_storage_options(), COPY_REQUEST_TIMEOUT_SECONDS)
            target = dataset_folder.removeprefix("s3://")
        for file in files:
            if isinstance(target_fs, pafs.LocalFileSystem):
                os.makedirs(os.path.dirname(f"{target}/{file}"), exist_ok=True)
            target_fs.copy_file(f"{source}/{file}", f"{target}/{file}")
    except Exception as e:
        raise DataWriteError(f"Failed to copy the previous dataset: {e}") from e

    # Each append gets the next folder, so the files keep the order in which they were added
    appends = {file.split("/")[0] for file in files if file.startswith(APPEND_FOLDER_PREFIX)}
    append_folder = f"{dataset_folder}/{APPEND_FOLDER_PREFIX}{len(appends) + 1:06d}"
    # The lineage of the whole dataset: its columns come from the frame and from the dataset itself
    traced = pl.concat([previous_data, frame]) if isinstance(frame, pl.LazyFrame) else None
    write_df_to_parquet(frame, append_folder, transform_id, base_path, traced)


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


def _is_previous(annotation: Any) -> bool:
    """Whether a parameter annotation asks for the previous build: Previous, or Previous in a union."""
    if isinstance(annotation, str):
        # from __future__ import annotations keeps the annotation as its source text
        return any(
            (isinstance(node, ast.Name) and node.id == "Previous")
            or (isinstance(node, ast.Attribute) and node.attr == "Previous")
            for node in ast.walk(ast.parse(annotation, mode="eval"))
        )
    return annotation is Previous or Previous in typing.get_args(annotation)


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
    previous = Previous(transform_id)
    previous_params: list[str] = []

    for name, param in params.items():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # The transform's own previous build is no input dataset
        if _is_previous(param.annotation):
            previous_params.append(name)
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
    transform_result = main_func(**data_dict, **dict.fromkeys(previous_params, previous))

    # Handle optional health_checks return
    if isinstance(transform_result, tuple) and len(transform_result) == 2:
        result, health_checks = transform_result
    else:
        result = transform_result
        health_checks = None

    # Write output
    output_folder = f"{build_folder(transform_conn.id, base_path)}/{DATASET_FOLDER_NAME}"
    if isinstance(result, Append):
        write_append(result.frame, previous, output_folder, transform_conn.id, base_path)
    else:
        write_df_to_parquet(result, output_folder, transform_conn.id, base_path)
    print(f"Execution Time: {time.perf_counter() - start_time:.2f} sec")

    # Checks run on the written file: re-running the transform for every check is
    # slower, and its row order may differ from the file's
    if health_checks:
        checks_start = time.perf_counter()
        run_health_checks(
            f"{output_folder}/{DATASET_GLOB}", health_checks, output_storage_options()
        )
        print(f"Health Check Time: {time.perf_counter() - checks_start:.2f} sec")


def _draft_read_limit() -> int | None:
    """How many rows a draft build reads of each dataset; None for all of them, as in every other build.
    A workspace can change the number, or turn the limit off with -1 (_config.json draftReadLimit)."""
    if os.environ.get("IS_DRAFT_BUILD", "").lower() != "true":
        return None
    raw_limit = os.environ.get("DRAFT_READ_LIMIT", str(DEFAULT_DRAFT_READ_LIMIT))
    try:
        limit = int(raw_limit)
    except ValueError as e:
        raise ConfigurationError(
            f"DRAFT_READ_LIMIT must be a whole number of rows, got {raw_limit!r}."
        ) from e
    if limit == NO_DRAFT_READ_LIMIT:
        return None
    # pandas reads head(-n) as "all but the last n rows", so any other non-positive limit must not get that far
    if limit < 1:
        raise ConfigurationError(
            f"DRAFT_READ_LIMIT must be a positive number of rows or {NO_DRAFT_READ_LIMIT} for no limit, got {limit}."
        )
    return limit


def _read_dataset(dataset_folder: str, read_limit: int | None) -> pl.LazyFrame | pd.DataFrame:
    """The dataset of a folder as the workspace's dataframe type; with a limit only its first rows."""
    options = storage_options(dataset_folder)
    if DF_TYPE == "polars":
        frame = pl.scan_parquet(f"{dataset_folder}/{DATASET_GLOB}", storage_options=options)
        return frame.head(read_limit) if read_limit is not None else frame  # lazy: only these rows are read

    fs, folder = _filesystem(dataset_folder)
    files = [f"{folder}/{file}" for file in _dataset_files(dataset_folder)]
    if not files:
        raise FileNotFoundError(f"There is no Parquet file below {dataset_folder}")
    dataset = pads.dataset(files, format="parquet", filesystem=fs)
    table = dataset.head(read_limit) if read_limit is not None else dataset.to_table()
    return table.to_pandas()


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

    draft_read_limit = _draft_read_limit()

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

        dataset_folder = f"{build_folder(rid, base_path)}/{DATASET_FOLDER_NAME}"
        storage_options = input_storage_options(rid)
        try:
            if os.environ.get("DS_BUILD_URI") and storage_options is None:
                raise DataLoadError(
                    f"No access to the data of '{rid}'; is it an input of this transform?"
                )
            df = _read_dataset(dataset_folder, draft_read_limit)

            # An input is its dataset with its metadata and its artifacts
            if transform_id is not None:
                ds_meta_data = DS_META_DATA.get(rid) if DS_META_DATA else None
                df.ds_meta = DSMeta(rid, ds_meta_data)  # type: ignore[union-attr]
                df.artifacts = Artifacts(f"{build_folder(rid, base_path)}/artifacts")  # type: ignore[union-attr]

            data_dict[param_name] = df
        except Exception as e:
            raise DataLoadError(
                f"Failed to load {param_name}. Ensure the input dataset exists at {dataset_folder}: {e}"
            ) from e
    return data_dict
