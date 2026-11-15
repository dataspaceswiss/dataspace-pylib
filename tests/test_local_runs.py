"""Builds on a developer's machine: the same runner as on DataSpace, with the folders below .dataspace/."""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from dataspace_pylib import (
    DataLoadError,
    DataSpaceError,
    HealthCheckFailure,
    TransformNotFoundError,
    build,
)
from dataspace_pylib.local import run_local


class Workspace:
    """A clone of a workspace: _config.json, _connections.json and one file per transform."""

    def __init__(self, root: Path, df_type: str = "polars"):
        self.root = root
        self.connections: list[dict] = []
        (root / "_config.json").write_text(
            json.dumps({"python": "3.11", "dfType": df_type, "packages": []})
        )
        self._save()

    def _save(self):
        (self.root / "_connections.json").write_text(json.dumps(self.connections))

    def add(self, name: str, code: str, inputs: tuple[str, ...] = ()):
        """A transform; an input is the name of another transform here, or the RID of one elsewhere."""
        rids = {c["path"].removesuffix(".py"): c["id"] for c in self.connections}
        self.connections = [c for c in self.connections if c["path"] != f"{name}.py"]
        self.connections.append(
            {
                "id": f"rid.transform.{name}",
                "path": f"{name}.py",
                "inputs": [rids.get(i, i) for i in inputs],
            }
        )
        self._save()
        (self.root / f"{name}.py").write_text(code)

    def builds(self, name: str) -> list[str]:
        return sorted(
            p.name for p in (self.root / ".dataspace" / name / "builds").iterdir()
        )

    def data(self, name: str, build: str | None = None) -> pl.DataFrame:
        folder = (
            self.root
            / ".dataspace"
            / name
            / "builds"
            / (build or self.builds(name)[-1])
        )
        return pl.read_parquet(folder / "datasets" / "**" / "*.parquet")

    def folder(self, name: str) -> Path:
        return self.root / ".dataspace" / name / "builds" / self.builds(name)[-1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Workspace(tmp_path)


SOURCE = "import polars as pl\ndef transform():\n    return pl.LazyFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})\n"


def test_a_transform_reads_the_latest_build_of_its_input(workspace):
    workspace.add("source", SOURCE)
    workspace.add(
        "double",
        "import polars as pl\n"
        "def transform(source):\n"
        "    assert source.meta.row_count == 3 and source.meta.columns == {'id': 'int64', 'value': 'int64'}\n"
        "    assert source.meta.transform_id == 'rid.transform.source' and source.meta.snapshot_id is None\n"
        "    return source.with_columns(double=pl.col('value') * 2)\n",
        inputs=("source",),
    )

    run_local("source")
    run_local("double")

    assert workspace.builds("source") == ["0001"]
    assert workspace.data("double")["double"].to_list() == [20, 40, 60]
    lineage = json.loads(
        (workspace.folder("double") / "meta" / "columns.json").read_text()
    )
    assert lineage["double"] == [
        {"dataset_rid": "rid.transform.source", "column": "value"}
    ]

    # A changed input is a new build of it, and the next build downstream reads that one
    workspace.add("source", SOURCE.replace("[10, 20, 30]", "[1, 2, 3]"))
    run_local("source.py")
    run_local("rid.transform.double")
    assert workspace.builds("source") == ["0001", "0002"]
    assert workspace.data("double")["double"].to_list() == [2, 4, 6]


def test_every_run_is_a_build_and_the_previous_one_is_the_last_that_succeeded(
    workspace,
):
    workspace.add("source", SOURCE)
    workspace.add(
        "history",
        "from pathlib import Path\n"
        "import polars as pl\n"
        "from dataspace_pylib import Previous, build\n"
        "def transform(source, earlier: Previous):\n"
        "    assert build.is_local and not build.is_draft and build.id.startswith('local-')\n"
        "    runs = int(earlier.artifacts.get('runs.txt').read_text()) + 1 if earlier.exists else 1\n"
        "    (build.artifacts / 'runs.txt').write_text(str(runs))\n"
        "    if Path('fail').exists():\n"
        "        raise ValueError('this run fails')\n"
        "    return earlier.append(source.with_columns(run=pl.lit(runs)))\n",
        inputs=("source",),
    )
    run_local("source")

    run_local("history")
    run_local("history")
    (workspace.root / "fail").touch()
    with pytest.raises(ValueError, match="this run fails"):
        run_local("history")
    (workspace.root / "fail").unlink()
    run_local("history")

    # The failed third build is kept for a look at it, but it is nobody's previous build
    assert workspace.builds("history") == ["0001", "0002", "0003", "0004"]
    assert not (workspace.root / ".dataspace/history/builds/0003/SUCCESS").exists()
    assert sorted(workspace.data("history")["run"].unique().to_list()) == [1, 2, 3]
    files = sorted(
        str(p.relative_to(workspace.folder("history") / "datasets"))
        for p in (workspace.folder("history") / "datasets").rglob("*.parquet")
    )
    assert files == [
        "00000000.parquet",
        "append-000001/00000000.parquet",
        "append-000002/00000000.parquet",
    ]
    assert workspace.data("history", "0001").height == 3


def test_only_the_last_successful_builds_are_kept(workspace):
    workspace.add("source", SOURCE)

    for _ in range(5):
        run_local("source")

    assert workspace.builds("source") == ["0003", "0004", "0005"]


def test_an_input_that_was_never_built_needs_mock_data(workspace):
    workspace.add("source", SOURCE)
    workspace.add(
        "copy", "def transform(source):\n    return source\n", inputs=("source",)
    )

    with pytest.raises(
        DataLoadError,
        match=r"run it first .*or put Parquet files into .dataspace/source/mock/datasets",
    ):
        run_local("copy")

    mock = workspace.root / ".dataspace" / "source" / "mock"
    (mock / "datasets").mkdir(parents=True)
    pl.DataFrame({"id": [7]}).write_parquet(mock / "datasets" / "anything.parquet")
    run_local("copy")
    assert workspace.data("copy")["id"].to_list() == [7]

    # Once the input is built here, its build counts and the mock data no longer does
    run_local("source")
    run_local("copy")
    assert workspace.data("copy")["id"].to_list() == [1, 2, 3]


def test_an_input_of_another_workspace_is_mocked_by_its_rid(workspace):
    rid = "rid.transform.11111111-2222-3333-4444-555555555555"
    workspace.add(
        "uses_other",
        "from __future__ import annotations\n"
        "import polars as pl\n"
        "from dataspace_pylib import Transform\n"
        f"def transform(orders: Transform('{rid}')):\n"
        "    return orders.select(total=pl.col('amount').sum())\n",
        inputs=(rid,),
    )
    mock = workspace.root / ".dataspace" / rid / "mock" / "datasets"
    mock.mkdir(parents=True)
    pl.DataFrame({"amount": [1, 2]}).write_parquet(mock / "data.parquet")

    run_local("uses_other")

    assert workspace.data("uses_other")["total"].to_list() == [3]


def test_artifacts_are_in_the_build_folder_and_downstream_reads_them_in_place(
    workspace,
):
    workspace.add(
        "source",
        "import polars as pl\n"
        "from dataspace_pylib import build\n"
        "def transform():\n"
        "    (build.artifacts / 'raw').mkdir()\n"
        "    (build.artifacts / 'raw' / 'notes.txt').write_text('notes')\n"
        "    assert build.storage.options is None\n"
        "    return pl.LazyFrame({'id': [1]})\n",
    )
    workspace.add(
        "reader",
        "import polars as pl\n"
        "from dataspace_pylib import Input\n"
        "def transform(source: Input):\n"
        "    assert source.artifacts.list() == ['raw/notes.txt'] and source.artifacts.list('*.csv') == []\n"
        "    return pl.LazyFrame({'notes': [source.artifacts.get('raw/notes.txt').read_text()],\n"
        "                         'path': [str(source.artifacts.get('raw'))]})\n",
        inputs=("source",),
    )

    run_local("source")
    run_local("reader")

    row = workspace.data("reader").row(0, named=True)
    assert row["notes"] == "notes"
    # Nothing is copied on a local run
    assert Path(row["path"]) == workspace.folder("source") / "artifacts" / "raw"


def test_health_checks_write_their_report_and_a_failed_one_fails_the_build(workspace):
    workspace.add(
        "checked",
        "import polars as pl\n"
        "from dataspace_pylib import Check\n"
        "def transform():\n"
        "    return pl.LazyFrame({'id': [1, 1]}), [Check('id').unique(severity='fail')]\n",
    )

    with pytest.raises(HealthCheckFailure):
        run_local("checked")

    report = json.loads(
        (workspace.folder("checked") / "meta" / "health_report.json").read_text()
    )
    assert report["summary"]["failed"] == 1
    assert not (workspace.folder("checked") / "SUCCESS").exists()


def test_a_large_output_is_split_into_files(workspace, monkeypatch):
    monkeypatch.setattr("dataspace_pylib.runner.DATASET_PART_BYTES", 1_000_000)
    # Each row is 8 bytes in memory, so the 8 MB of this frame fill eight files
    workspace.add(
        "big",
        "import polars as pl\ndef transform():\n    return pl.DataFrame({'a': range(1_000_000)})\n",
    )

    run_local("big")

    assert len(list((workspace.folder("big") / "datasets").iterdir())) == 8
    assert workspace.data("big")["a"].to_list() == list(range(1_000_000))


def test_a_pandas_workspace_reads_and_writes_pandas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace = Workspace(tmp_path, "pandas")
    workspace.add(
        "source",
        "import pandas as pd\ndef transform():\n    return pd.DataFrame({'id': [1, 2]})\n",
    )
    workspace.add(
        "plus",
        "import pandas as pd\n"
        "def transform(source):\n"
        "    assert isinstance(source, pd.DataFrame) and source.meta.row_count == 2\n"
        "    return source.assign(plus=source['id'] + 1)\n",
        inputs=("source",),
    )

    run_local("source")
    run_local("plus")

    assert workspace.data("plus")["plus"].to_list() == [2, 3]


@pytest.mark.parametrize(
    ("code", "error", "message"),
    [
        (
            "def transform(nobody):\n    return nobody\n",
            DataLoadError,
            "names no transform",
        ),
        ("def other():\n    pass\n", DataSpaceError, "function named 'transform'"),
        (
            "def transform():\n    return 'text'\n",
            DataSpaceError,
            "Unsupported dataframe type",
        ),
    ],
)
def test_mistakes_in_a_transform_are_named(workspace, code, error, message):
    workspace.add("wrong", code)

    with pytest.raises(error, match=message):
        run_local("wrong")


def test_a_transform_that_does_not_exist_lists_the_ones_that_do(workspace):
    workspace.add("source", SOURCE)

    with pytest.raises(TransformNotFoundError, match="This workspace has: source"):
        run_local("sauce")


def test_the_api_needs_a_running_build():
    with pytest.raises(DataSpaceError, match="No build is running"):
        _ = build.artifacts


def test_command_line(workspace):
    workspace.add("source", SOURCE)

    ran = subprocess.run(
        [sys.executable, "-m", "dataspace_pylib", "run", "source"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert (
        ran.returncode == 0
        and "Building source into .dataspace/source/builds/0001" in ran.stdout
    )
    assert workspace.data("source")["id"].to_list() == [1, 2, 3]

    missing = subprocess.run(
        [sys.executable, "-m", "dataspace_pylib", "run", "sauce"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert missing.returncode == 1 and missing.stderr.startswith(
        "TransformNotFoundError: There is no transform 'sauce'"
    )

    (workspace.root / "build.json").write_text(json.dumps({"version": 2}))
    newer = subprocess.run(
        [sys.executable, "-m", "dataspace_pylib", "build", "build.json"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert (
        newer.returncode == 1
        and "is not one this version of dataspace-pylib can read" in newer.stderr
    )
