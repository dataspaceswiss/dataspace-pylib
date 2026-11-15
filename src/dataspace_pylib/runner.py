"""Runs the transform of a build and is the API transforms import."""

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
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as pads  # type: ignore[import-untyped]
import pyarrow.fs as pafs  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError

from .health_check import run_health_checks
from .input_source_tracer import trace_lazyframe_sources

DataFrame: TypeAlias = pl.DataFrame | pl.LazyFrame | pd.DataFrame


class Transform:
    """Names an input of another workspace by its RID, as a parameter annotation:
    ``def transform(orders: Transform("rid.transform.…"))``."""

    def __init__(self, rid: str):
        self.rid = rid

    def __repr__(self):
        return f"Transform('{self.rid}')"


class DataSpaceError(Exception):
    """Base of the errors of this library."""


class ConfigurationError(DataSpaceError):
    """Raised when configuration files or the build description are invalid or missing."""


class TransformNotFoundError(DataSpaceError):
    """Raised when a transform is not found."""


class ModuleLoadError(DataSpaceError):
    """Raised when a transform module cannot be loaded."""


class DataLoadError(DataSpaceError):
    """Raised when input data cannot be loaded."""


class DataWriteError(DataSpaceError):
    """Raised when output data cannot be written."""


# The build description
#
# Everything a build needs to know comes as one document: from DataSpace as a file in the build container,
# from `dataspace-pylib run` for a build on a developer's machine. The code below is the same for both; a
# folder is an s3:// URI with temporary credentials or a local path.
#
# The credentials are deliberately not the AWS_* environment variables: those would also point the AWS
# clients of a transform's own code to the DataSpace storage.


class Credentials(BaseModel):
    accessKeyId: str
    secretAccessKey: str
    sessionToken: str


class Location(BaseModel):
    """A build folder; in S3 with the temporary credentials for it."""

    uri: str
    credentials: Credentials | None = None


class Dataset(Location):
    """The build folder of a dataset this build reads, with the facts about that dataset."""

    meta: dict[str, Any]


class StorageEndpoint(BaseModel):
    endpoint: str
    region: str


class BuildDescription(BaseModel):
    version: Literal[1]
    id: str
    transformId: str
    branch: str
    draft: bool
    # Rows a draft build reads of each dataset when the workspace sets draftReadLimit; -1 for all
    draftReadLimit: int | None = None
    local: bool
    # Where the S3 folders are; None when every folder is local
    storage: StorageEndpoint | None = None
    output: Location
    inputs: dict[str, Dataset]
    previous: Dataset | None = None
    # Local folders: what is in them when the transform ends belongs to the build
    artifactFolder: str
    metaFolder: str


_description: BuildDescription | None = None


def _build() -> BuildDescription:
    if _description is None:
        raise DataSpaceError(
            "No build is running. Run a transform with `python -m dataspace_pylib run <transform>`."
        )
    return _description


def _s3_filesystem(
    credentials: Credentials, request_timeout: float | None = None
) -> pafs.S3FileSystem:
    storage = _build().storage
    if storage is None:
        raise ConfigurationError("The build description names an S3 folder but no storage.")
    return pafs.S3FileSystem(
        access_key=credentials.accessKeyId,
        secret_key=credentials.secretAccessKey,
        session_token=credentials.sessionToken,
        endpoint_override=storage.endpoint,
        region=storage.region,
        request_timeout=request_timeout,
        allow_bucket_creation=False,
        check_directory_existence_before_creation=False,
    )


def _credentials(path: str) -> Credentials | None:
    """The credentials for an s3:// path: those of the build folder it lies in. None for a local path."""
    if not path.startswith("s3://"):
        return None
    build = _build()
    locations = [build.output, *build.inputs.values(), *([build.previous] if build.previous else [])]
    for location in locations:
        if path == location.uri or path.startswith(location.uri + "/"):
            if location.credentials is None:
                break
            return location.credentials
    raise DataLoadError(f"This build has no access to {path}")


def storage_options(path: str) -> dict[str, str] | None:
    """Polars storage options to read or write a path of this build; None for a local path."""
    credentials = _credentials(path)
    if credentials is None:
        return None
    storage = _build().storage
    if storage is None:
        raise ConfigurationError("The build description names an S3 folder but no storage.")
    return {
        "aws_endpoint_url": storage.endpoint,
        "aws_region": storage.region,
        "aws_allow_http": str(storage.endpoint.startswith("http://")).lower(),
        "aws_access_key_id": credentials.accessKeyId,
        "aws_secret_access_key": credentials.secretAccessKey,
        "aws_session_token": credentials.sessionToken,
    }


def upload_output_folders() -> None:
    """Upload the local artifact and meta folders into the build folder, when that is in S3. On a local
    build they already are in it."""
    build = _build()
    credentials = _credentials(build.output.uri)
    if credentials is None:
        return
    fs = _s3_filesystem(credentials)
    for local_folder, name in ((build.artifactFolder, "artifacts"), (build.metaFolder, "meta")):
        for root, _, files in os.walk(local_folder):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                relative = os.path.relpath(local_path, local_folder).replace(os.sep, "/")
                target = f"{build.output.uri.removeprefix('s3://')}/{name}/{relative}"
                with open(local_path, "rb") as src, fs.open_output_stream(target) as dst:
                    shutil.copyfileobj(src, dst)


def _filesystem(path: str) -> tuple[pafs.FileSystem, str]:
    """The filesystem of a path of this build, its previous build or an input, and the path in it."""
    credentials = _credentials(path)
    if credentials is None:
        return pafs.LocalFileSystem(), path
    return _s3_filesystem(credentials), path.removeprefix("s3://")


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


class Meta:
    """Facts about a dataset: of an input, or of the previous build."""

    def __init__(self, meta: dict[str, Any]):
        self.transform_id: str = meta["transformId"]
        self.build_id: str = meta["buildId"]
        # None for a dataset built on this machine
        self.snapshot_id: str | None = meta["snapshotId"]
        self.row_count: int = meta["rowCount"]
        self.column_count: int = meta["columnCount"]
        self.size_bytes: int = meta["sizeBytes"]
        self.created_at: datetime = datetime.fromisoformat(meta["createdAt"])
        # Column name to type
        self.columns: dict[str, str] = meta["columns"]


class Input(pl.LazyFrame):
    """The type of an input, for code completion: ``def transform(orders: Input)``. An input is the
    LazyFrame of the upstream dataset with these two attributes; the class itself is never created."""

    artifacts: Artifacts
    meta: Meta


class Append:
    """What ``Previous.append`` returns: the rows to add to the files of the previous dataset."""

    def __init__(self, frame: pl.LazyFrame | pl.DataFrame):
        if not isinstance(frame, pl.LazyFrame | pl.DataFrame):
            raise TypeError(f"append needs a Polars frame, not {type(frame).__name__}")
        self.frame = frame


class Previous:
    """The last successful committed build of this transform on this branch. A parameter with this
    type makes a transform incremental: ``def transform(orders, history: Previous)``."""

    def __init__(self) -> None:
        self._artifacts: Artifacts | None = None

    @property
    def exists(self) -> bool:
        """False on the first build of the transform."""
        return _build().previous is not None

    def _dataset(self) -> Dataset:
        previous = _build().previous
        if previous is None:
            raise DataSpaceError(
                "This transform has no previous build yet; check `exists` before using it."
            )
        return previous

    @property
    def data(self) -> pl.LazyFrame | pd.DataFrame:
        """The previous dataset."""
        return _read_dataset(f"{self._dataset().uri}/{DATASET_FOLDER_NAME}")

    @property
    def artifacts(self) -> Artifacts:
        """The artifacts of the previous build."""
        if self._artifacts is None:
            self._artifacts = Artifacts(f"{self._dataset().uri}/artifacts")
        return self._artifacts

    @property
    def meta(self) -> Meta:
        """Row count, columns and the other facts about the previous dataset."""
        return Meta(self._dataset().meta)

    def append(self, frame: pl.LazyFrame | pl.DataFrame) -> Append:
        """Return this from the transform to keep the files of the previous dataset as they are and add
        the frame's rows as new files. The previous rows are neither read nor written again, so the build
        only costs what the new rows cost. The frame must have the columns and types of the previous
        dataset. On the first build the frame is the dataset."""
        return Append(frame)


class Storage:
    """Direct access to the folder of the running build, for what the rest of the API does not cover."""

    @property
    def uri(self) -> str:
        """The build's folder: ``s3://bucket/builds/<build id>`` on DataSpace, a local path on a local run."""
        return _build().output.uri

    @property
    def options(self) -> dict[str, str] | None:
        """Polars storage options with the build's temporary credentials, which write the build's own
        folder and read its previous build. None on a local run."""
        return storage_options(_build().output.uri)

    def filesystem(self) -> pafs.FileSystem:
        """A pyarrow filesystem with the same credentials; its paths are the uri without ``s3://``."""
        return _filesystem(_build().output.uri)[0]


class Build:
    """The running build: ``from dataspace_pylib import build``."""

    storage = Storage()

    @property
    def id(self) -> str:
        return _build().id

    @property
    def transform_id(self) -> str:
        return _build().transformId

    @property
    def branch(self) -> str:
        return _build().branch

    @property
    def is_draft(self) -> bool:
        """True for a build of code that is not committed; it reads only the first rows of its inputs."""
        return _build().draft

    @property
    def is_local(self) -> bool:
        """True for a build on a developer's machine (`dataspace-pylib run`), False on DataSpace."""
        return _build().local

    @property
    def artifacts(self) -> Path:
        """The local folder for the build's artifacts. What is in it when the transform ends is stored
        with the build, also after a failure."""
        folder = Path(_build().artifactFolder)
        folder.mkdir(parents=True, exist_ok=True)
        return folder


build = Build()


# The workspace's configuration files


class Config(BaseModel):
    python: str
    dfType: Literal["polars", "pandas"]
    packages: list[str]


class Connection(BaseModel):
    id: str
    path: str
    inputs: list[str]


CONFIG_FILE = "_config.json"
CONNECTIONS_FILE = "_connections.json"


def load_json(file_path: str) -> Any:
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {file_path}: {e}") from e


def load_configuration() -> tuple[Config, list[Connection]]:
    """The workspace's _config.json and _connections.json, read from the current folder."""
    try:
        config = Config(**load_json(CONFIG_FILE))
    except (ValidationError, TypeError) as e:
        raise ConfigurationError(f"Invalid configuration in {CONFIG_FILE}: {e}") from e
    try:
        connections = [Connection(**item) for item in load_json(CONNECTIONS_FILE)]
    except (ValidationError, TypeError) as e:
        raise ConfigurationError(f"Invalid configuration in {CONNECTIONS_FILE}: {e}") from e
    return config, connections


def path_to_name(path: str) -> str:
    """The name of a transform: its file name without the extension."""
    return Path(path).stem


# Datasets

DEFAULT_DRAFT_READ_LIMIT = 1_000_000
NO_DRAFT_READ_LIMIT = -1
# A dataset is every Parquet file below the datasets folder of a build: one file or many, also in subfolders
DATASET_FOLDER_NAME = "datasets"
DATASET_GLOB = "**/*.parquet"
# An output is split into files of about this size, measured as the size of the data in memory
DATASET_PART_BYTES = 1024**3
# The rows of each append are written into a folder of their own below the dataset folder
APPEND_FOLDER_PREFIX = "append-"
# The storage answers a copy only when it is done, and pyarrow gives up on a silent request after 3 seconds,
# which a file of about 1 GiB already takes. An hour covers a single file of several hundred GiB.
COPY_REQUEST_TIMEOUT_SECONDS = 3600

# The workspace's dataframe type; set when a build starts
_df_type: str | None = None


def _draft_read_limit() -> int | None:
    """How many rows a draft build reads of each dataset; None for all of them, as in every other build.
    A workspace can change the number, or turn the limit off with -1 (_config.json draftReadLimit)."""
    build = _build()
    if not build.draft:
        return None
    limit = DEFAULT_DRAFT_READ_LIMIT if build.draftReadLimit is None else build.draftReadLimit
    if limit == NO_DRAFT_READ_LIMIT:
        return None
    # pandas reads head(-n) as "all but the last n rows", so any other non-positive limit must not get that far
    if limit < 1:
        raise ConfigurationError(
            f"draftReadLimit must be a positive number of rows or {NO_DRAFT_READ_LIMIT} for no limit, got {limit}."
        )
    return limit


def _read_dataset(dataset_folder: str) -> pl.LazyFrame | pd.DataFrame:
    """The dataset of a folder as the workspace's dataframe type; in a draft build only its first rows."""
    read_limit = _draft_read_limit()
    if _df_type == "polars":
        frame = pl.scan_parquet(
            f"{dataset_folder}/{DATASET_GLOB}", storage_options=storage_options(dataset_folder)
        )
        return frame.head(read_limit) if read_limit is not None else frame  # lazy: only these rows are read

    fs, folder = _filesystem(dataset_folder)
    files = [f"{folder}/{file}" for file in _dataset_files(dataset_folder)]
    if not files:
        raise FileNotFoundError(f"There is no Parquet file below {dataset_folder}")
    dataset = pads.dataset(files, format="parquet", filesystem=fs)
    table = dataset.head(read_limit) if read_limit is not None else dataset.to_table()
    return table.to_pandas()


def write_column_lineage(lf: pl.LazyFrame) -> None:
    """Writes which input columns each output column comes from. Lineage is extra information: a query
    it cannot be read from does not fail the build."""
    build = _build()
    try:
        # Build folders are named after the build, so the scan paths do not name the transform: the
        # inputs', and this transform's own previous dataset
        source_names = {
            f"{dataset.uri}/{DATASET_FOLDER_NAME}/": rid for rid, dataset in build.inputs.items()
        }
        if build.previous is not None:
            source_names[f"{build.previous.uri}/{DATASET_FOLDER_NAME}/"] = build.transformId
        origins = trace_lazyframe_sources(lf, source_names, storage_options)

        os.makedirs(build.metaFolder, exist_ok=True)
        with open(os.path.join(build.metaFolder, "columns.json"), "w", encoding="utf-8") as f:
            json.dump(origins, f, indent=4)
    except Exception as e:  # noqa: BLE001 - lineage metadata is best-effort
        print(f"Failed to write the column lineage: {e}", file=sys.stderr)


def write_dataset(
    df: DataFrame, dataset_folder: str, traced: pl.LazyFrame | None = None
) -> None:
    """Writes a dataframe as the Parquet files of a dataset folder. `traced` is the query the column
    lineage is read from when it is not df itself."""
    try:
        options = storage_options(dataset_folder)
        if options is None:
            os.makedirs(dataset_folder)

        if isinstance(df, pl.LazyFrame):
            write_column_lineage(traced if traced is not None else df)
        if isinstance(df, pl.LazyFrame | pl.DataFrame):
            # Polars splits the output into files and streams them out, so no file grows without bound
            df.lazy().sink_parquet(
                pl.PartitionBy(dataset_folder, approximate_bytes_per_file=DATASET_PART_BYTES),
                storage_options=options,
            )
        elif isinstance(df, pd.DataFrame):
            # A pandas DataFrame is in memory as a whole, so it is written as one file
            fs, folder = _filesystem(dataset_folder)
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False), f"{folder}/00000000.parquet", filesystem=fs
            )
        else:
            raise TypeError(f"Unsupported dataframe type: {type(df)}")
    except Exception as e:
        raise DataWriteError(f"Failed to write the dataset: {e}") from e


def write_append(frame: pl.LazyFrame | pl.DataFrame, dataset_folder: str) -> None:
    """Writes a dataset as the files of the previous dataset plus the frame's rows as new files."""
    previous = _build().previous
    if previous is None:
        write_dataset(frame, dataset_folder)
        return

    previous_folder = f"{previous.uri}/{DATASET_FOLDER_NAME}"
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
        # The previous files are copied as they are. In S3 the storage copies them, with the build's own
        # credentials, which read the previous build: the files do not pass through the build.
        files = _dataset_files(previous_folder)
        credentials = _credentials(dataset_folder)
        if credentials is None:
            fs: pafs.FileSystem = pafs.LocalFileSystem()
        else:
            fs = _s3_filesystem(credentials, COPY_REQUEST_TIMEOUT_SECONDS)
        source, target = previous_folder.removeprefix("s3://"), dataset_folder.removeprefix("s3://")
        for file in files:
            if credentials is None:
                os.makedirs(os.path.dirname(f"{target}/{file}"), exist_ok=True)
            fs.copy_file(f"{source}/{file}", f"{target}/{file}")
    except Exception as e:
        raise DataWriteError(f"Failed to copy the previous dataset: {e}") from e

    # Each append gets the next folder, so the files keep the order in which they were added
    appends = {file.split("/")[0] for file in files if file.startswith(APPEND_FOLDER_PREFIX)}
    append_folder = f"{dataset_folder}/{APPEND_FOLDER_PREFIX}{len(appends) + 1:06d}"
    # The lineage of the whole dataset: its columns come from the frame and from the dataset itself
    traced = pl.concat([previous_data, frame]) if isinstance(frame, pl.LazyFrame) else None
    write_dataset(frame, append_folder, traced)


# Running a transform


def load_module(module_name: str, file_path: str) -> Any:
    """Dynamically loads a module from a given file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ModuleLoadError(f"Could not create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        # Registered while it runs, as an imported module is: dataclasses and pickling look their module up
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
        return module
    except Exception as e:
        raise ModuleLoadError(f"Failed to load module from {file_path}: {e}") from e


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


def _read_inputs(
    func: Callable[..., Any], connections: list[Connection]
) -> dict[str, pl.LazyFrame | pd.DataFrame | Previous]:
    """The arguments of a transform function: an input dataset for each parameter, named by the
    parameter (the name of a transform of this workspace) or by a Transform("rid") annotation, and the
    previous build for a parameter of the type Previous."""
    build = _build()
    by_name = {path_to_name(connection.path): connection.id for connection in connections}
    arguments: dict[str, pl.LazyFrame | pd.DataFrame | Previous] = {}

    for name, parameter in inspect.signature(func).parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if _is_previous(parameter.annotation):
            arguments[name] = Previous()
            continue

        annotated = _annotated_transform(parameter.annotation, func)
        rid = annotated.rid if annotated is not None else by_name.get(name)
        if rid is None:
            raise DataLoadError(f"Parameter '{name}' names no transform of this workspace.")
        dataset = build.inputs.get(rid)
        if dataset is None:
            raise DataLoadError(
                f"This build was not given the data of '{rid}' (parameter '{name}'); is it an input of this transform?"
            )
        try:
            frame = _read_dataset(f"{dataset.uri}/{DATASET_FOLDER_NAME}")
        except Exception as e:
            raise DataLoadError(f"Failed to load '{name}' from {dataset.uri}: {e}") from e
        # An input is its dataset with the facts about it and its artifacts. Set past the frame's own
        # __setattr__: pandas would overwrite a column of that name instead.
        object.__setattr__(frame, "meta", Meta(dataset.meta))
        object.__setattr__(frame, "artifacts", Artifacts(f"{dataset.uri}/artifacts"))
        arguments[name] = frame
    return arguments


def run_build(description: BuildDescription) -> None:
    """Runs the transform of a build description; the current folder is the workspace's repository."""
    global _description
    _description = description
    try:
        try:
            _run_transform()
        except BaseException:
            # Keep the artifacts and health report of failed builds for debugging.
            try:
                upload_output_folders()
            except Exception as e:  # noqa: BLE001 - the build error is the one to report
                print(f"Failed to upload artifacts: {e}", file=sys.stderr)
            raise

        upload_start = time.perf_counter()
        try:
            upload_output_folders()
        except Exception as e:
            raise DataWriteError(f"Failed to upload artifacts: {e}") from e
        if _credentials(description.output.uri) is not None:
            print(f"Upload Time: {time.perf_counter() - upload_start:.2f} sec")
    finally:
        _description = None

    print("--------- Build successful ---------")


def _run_transform() -> None:
    global _df_type
    build = _build()
    config, connections = load_configuration()
    _df_type = config.dfType

    connection = next((c for c in connections if c.id == build.transformId), None)
    if connection is None:
        raise TransformNotFoundError(f"Transform with ID '{build.transformId}' not found.")

    # Not under the transform's own name, which could be that of a module the code imports
    module = load_module(f"dataspace_transform_{path_to_name(connection.path)}", connection.path)
    func = getattr(module, "transform", None)
    if not callable(func):
        raise ModuleLoadError(f"{connection.path} does not contain a function named 'transform'.")

    # A lazy query only runs when its output is written, so reading, transforming
    # and writing are timed together; health checks run afterwards on the written
    # dataset and get their own timer
    start_time = time.perf_counter()
    result = func(**_read_inputs(func, connections))

    # A transform returns its dataframe, or the dataframe and its health checks
    health_checks = None
    if isinstance(result, tuple) and len(result) == 2:
        result, health_checks = result

    output_folder = f"{build.output.uri}/{DATASET_FOLDER_NAME}"
    if isinstance(result, Append):
        write_append(result.frame, output_folder)
    else:
        write_dataset(result, output_folder)
    print(f"Execution Time: {time.perf_counter() - start_time:.2f} sec")

    # Checks run on the written dataset: re-running the transform for every check is
    # slower, and its row order may differ from the dataset's
    if health_checks:
        checks_start = time.perf_counter()
        run_health_checks(
            f"{output_folder}/{DATASET_GLOB}",
            health_checks,
            build.metaFolder,
            storage_options(output_folder),
        )
        print(f"Health Check Time: {time.perf_counter() - checks_start:.2f} sec")
