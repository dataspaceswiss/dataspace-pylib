"""Builds on a developer's machine: `python -m dataspace_pylib run <transform>` in a clone of a workspace.

A local build works like one on DataSpace. Every run gets a build folder of its own, its inputs are the
latest builds of the upstream transforms, and its previous build is its own last successful one. Only the
place differs: the folders are below .dataspace/ in the repository instead of in S3.

    .dataspace/<transform>/builds/0001/   one folder per run: datasets/, artifacts/, meta/, and the file
                                          SUCCESS once the build has succeeded
    .dataspace/<transform>/mock/          hand-made data for a transform that is not built here: Parquet
                                          files in datasets/, and artifacts/ if the code reads any
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads  # type: ignore[import-untyped]

from .runner import (
    DATASET_FOLDER_NAME,
    BuildDescription,
    DataLoadError,
    Dataset,
    Location,
    TransformNotFoundError,
    load_configuration,
    path_to_name,
    run_build,
)

STORE = ".dataspace"
# The file that marks a build folder as a successful build
SUCCESS = "SUCCESS"
# Successful builds kept per transform; older ones are removed after a run
BUILDS_KEPT = 3


def _dataset_files(build_folder: Path) -> list[Path]:
    return sorted((build_folder / DATASET_FOLDER_NAME).rglob("*.parquet"))


def _builds(transform_folder: Path) -> list[Path]:
    """The build folders of a transform, oldest first."""
    builds = transform_folder / "builds"
    if not builds.is_dir():
        return []
    folders = (
        folder
        for folder in builds.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    )
    return sorted(folders, key=lambda folder: int(folder.name))


def _dataset(rid: str, build_folder: Path) -> Dataset:
    """A local build folder as a dataset a build reads, with the facts about it read from its files."""
    files = _dataset_files(build_folder)
    data = pads.dataset([str(file) for file in files], format="parquet")
    meta: dict[str, Any] = {
        "transformId": rid,
        "buildId": f"local-{build_folder.name}",
        "snapshotId": None,
        "rowCount": data.count_rows(),
        "columnCount": len(data.schema.names),
        "sizeBytes": sum(file.stat().st_size for file in files),
        "createdAt": datetime.fromtimestamp(
            build_folder.stat().st_mtime, timezone.utc
        ).isoformat(),
        "columns": {field.name: str(field.type) for field in data.schema},
    }
    return Dataset(uri=str(build_folder), meta=meta)


def _succeeded(transform_folder: Path) -> list[Path]:
    """The successful builds of a transform, oldest first."""
    return [build for build in _builds(transform_folder) if (build / SUCCESS).exists()]


def _latest(rid: str, name: str, store: Path) -> Dataset | None:
    """The newest successful local build of a transform, else its mock data, else None."""
    succeeded = _succeeded(store / name)
    if succeeded:
        return _dataset(rid, succeeded[-1])
    mock = store / name / "mock"
    return _dataset(rid, mock) if _dataset_files(mock) else None


def _git_branch() -> str:
    try:
        head = Path(".git/HEAD").read_text().strip()
    except OSError:
        return "local"
    return (
        head.removeprefix("ref: refs/heads/") if head.startswith("ref: ") else "local"
    )


def run_local(transform: str) -> None:
    """Builds a transform, given by its name, its file or its RID, in the repository of the current folder."""
    _, connections = load_configuration()
    names = {connection.id: path_to_name(connection.path) for connection in connections}
    connection = next(
        (c for c in connections if transform in (c.id, c.path, path_to_name(c.path))),
        None,
    )
    if connection is None:
        raise TransformNotFoundError(
            f"There is no transform '{transform}'. This workspace has: {', '.join(sorted(names.values()))}"
        )
    name = names[connection.id]
    store = Path(STORE).resolve()

    inputs: dict[str, Dataset] = {}
    for rid in connection.inputs:
        # A transform of another workspace has no name here; its data is found by its RID
        input_name = names.get(rid, rid)
        dataset = _latest(rid, input_name, store)
        if dataset is None:
            how = (
                f"run it first (`dataspace-pylib run {input_name}`), or "
                if rid in names
                else ""
            )
            raise DataLoadError(
                f"There is no local data for the input '{input_name}': {how}put Parquet files into "
                f"{os.path.join(STORE, input_name, 'mock', DATASET_FOLDER_NAME)}"
            )
        inputs[rid] = dataset

    builds = _builds(store / name)
    number = int(builds[-1].name) + 1 if builds else 1
    build_folder = store / name / "builds" / f"{number:04d}"
    for folder in ("artifacts", "meta"):
        (build_folder / folder).mkdir(parents=True)
    # Mock data is not a build, so it is never the previous build
    succeeded = _succeeded(store / name)
    previous = succeeded[-1] if succeeded else None

    print(f"Building {name} into {os.path.relpath(build_folder)}")
    run_build(
        BuildDescription(
            version=1,
            id=f"local-{number:04d}",
            transformId=connection.id,
            branch=_git_branch(),
            draft=False,
            local=True,
            output=Location(uri=str(build_folder)),
            inputs=inputs,
            previous=_dataset(connection.id, previous) if previous else None,
            artifactFolder=str(build_folder / "artifacts"),
            metaFolder=str(build_folder / "meta"),
        )
    )

    (build_folder / SUCCESS).touch()

    # Keep the last successful builds and what came after the oldest of them
    oldest_kept = int(_succeeded(store / name)[-BUILDS_KEPT:][0].name)
    for old in _builds(store / name):
        if int(old.name) < oldest_kept:
            shutil.rmtree(old)
