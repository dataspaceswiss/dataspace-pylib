"""Runs transforms against a real S3 server with the credentials DataSpace hands to builds.

Skipped unless DS_TEST_S3_ENDPOINT points to an S3 server with STS (e.g. RustFS) whose root
credentials are DS_TEST_S3_ACCESS_KEY and DS_TEST_S3_SECRET_KEY. The bucket must exist.
"""

import io
import json
import os
import subprocess
import sys
import uuid

import polars as pl
import pyarrow.fs as pafs
import pytest

from pyrunner_lib import (
    DataLoadError,
    DataWriteError,
    Previous,
    PyrunnerError,
    transform,
)
from pyrunner_lib import build as running_build
from pyrunner_lib.pyrunner_lib import Artifacts

ENDPOINT = os.environ.get("DS_TEST_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(not ENDPOINT, reason="DS_TEST_S3_ENDPOINT is not set")
BUCKET = os.environ.get("DS_TEST_S3_BUCKET", "dataspace")
REGION = "us-east-1"


@pytest.fixture
def s3():
    boto3 = pytest.importorskip("boto3")
    root = {
        "endpoint_url": ENDPOINT,
        "aws_access_key_id": os.environ["DS_TEST_S3_ACCESS_KEY"],
        "aws_secret_access_key": os.environ["DS_TEST_S3_SECRET_KEY"],
        "region_name": REGION,
    }
    return boto3.client("s3", **root), boto3.client("sts", **root)


def scoped_credentials(sts, read_prefixes, write_prefixes):
    """Same session policy as the backend's StorageService.CreateScopedCredentials."""
    arn = f"arn:aws:s3:::{BUCKET}"
    statements = []
    if read_prefixes:
        statements.append(
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [f"{arn}/{p}*" for p in read_prefixes],
            }
        )
    if write_prefixes:
        statements.append(
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                "Resource": [f"{arn}/{p}*" for p in write_prefixes],
            }
        )
    statements.append(
        {
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": [arn],
            "Condition": {
                "StringLike": {
                    "s3:prefix": [f"{p}*" for p in read_prefixes + write_prefixes]
                }
            },
        }
    )
    response = sts.assume_role(
        RoleArn="arn:aws:iam::000000000000:role/dataspace",
        RoleSessionName="dataspace",
        DurationSeconds=900,
        Policy=json.dumps({"Version": "2012-10-17", "Statement": statements}),
    )
    return response["Credentials"]


@pytest.fixture
def build(s3, tmp_path, monkeypatch):
    """A build of rid.transform.output with the input rid.transform.input, as the backend sets it up."""
    client, sts = s3
    input_build = f"builds/rid.build.{uuid.uuid4()}/"
    output_build = f"builds/rid.build.{uuid.uuid4()}/"
    previous_build = f"builds/rid.build.{uuid.uuid4()}/"
    # The input is a dataset of two files; the previous build still has the single file of older builds
    for name, rows in (
        ("00000000.parquet", {"id": [1, 2], "value": [10, 20]}),
        ("00000001.parquet", {"id": [3], "value": [30]}),
    ):
        buffer = io.BytesIO()
        pl.DataFrame(rows).write_parquet(buffer)
        client.put_object(
            Bucket=BUCKET, Key=f"{input_build}datasets/{name}", Body=buffer.getvalue()
        )
    buffer = io.BytesIO()
    pl.DataFrame({"id": [0], "value": [0]}).write_parquet(buffer)
    client.put_object(
        Bucket=BUCKET,
        Key=f"{previous_build}datasets/data.parquet",
        Body=buffer.getvalue(),
    )

    own = scoped_credentials(sts, [previous_build], [output_build])
    inp = scoped_credentials(sts, [input_build], [])
    code = tmp_path / "code"
    code.mkdir()
    (code / "_config.json").write_text(
        json.dumps({"python": "3.11", "dfType": "polars", "packages": []})
    )
    (code / "_connections.json").write_text(
        json.dumps(
            [
                {"id": "rid.transform.input", "path": "input.py", "inputs": []},
                {
                    "id": "rid.transform.output",
                    "path": "output.py",
                    "inputs": ["rid.transform.input"],
                },
            ]
        )
    )
    monkeypatch.chdir(code)
    env = {
        "TRANSFORM_ID": "rid.transform.output",
        "DS_BUILD_URI": f"s3://{BUCKET}/{output_build}".rstrip("/"),
        "DS_INPUTS": json.dumps(
            {
                "rid.transform.input": {
                    "uri": f"s3://{BUCKET}/{input_build}".rstrip("/"),
                    "accessKeyId": inp["AccessKeyId"],
                    "secretAccessKey": inp["SecretAccessKey"],
                    "sessionToken": inp["SessionToken"],
                }
            }
        ),
        "DS_BUILD_ID": output_build.split("/")[1],
        "DS_BRANCH": "main",
        "DS_PREVIOUS": json.dumps(
            {
                "uri": f"s3://{BUCKET}/{previous_build}".rstrip("/"),
                "meta": {"RowCount": 1},
            }
        ),
        "ARTIFACT_FOLDER": str(tmp_path / "out" / "artifacts"),
        "META_FOLDER": str(tmp_path / "out" / "meta"),
        "DS_S3_ENDPOINT": ENDPOINT,
        "DS_S3_REGION": REGION,
        "DS_S3_ACCESS_KEY_ID": own["AccessKeyId"],
        "DS_S3_SECRET_ACCESS_KEY": own["SecretAccessKey"],
        "DS_S3_SESSION_TOKEN": own["SessionToken"],
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    os.makedirs(env["ARTIFACT_FOLDER"])
    return client, code, input_build, output_build


def dataset(client, build_prefix):
    """The dataset of a build: every Parquet file below its datasets folder, in the order of their names."""
    listing = client.list_objects_v2(Bucket=BUCKET, Prefix=f"{build_prefix}datasets/")
    names = sorted(
        o["Key"] for o in listing["Contents"] if o["Key"].endswith(".parquet")
    )
    return pl.concat(
        pl.read_parquet(
            io.BytesIO(client.get_object(Bucket=BUCKET, Key=name)["Body"].read())
        )
        for name in names
    )


def keys(client, prefix):
    return sorted(
        o["Key"][len(prefix) :]
        for o in client.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get(
            "Contents", []
        )
    )


def test_polars_build_reads_input_and_writes_dataset_artifacts_and_lineage(build):
    client, code, _, output_build = build
    (code / "output.py").write_text(
        "import os, polars as pl\n"
        "def transform(input):\n"
        "    open(os.path.join(os.environ['ARTIFACT_FOLDER'], 'index.html'), 'w').write('<html><head></head></html>')\n"
        "    return input.with_columns(double=pl.col('value') * 2)\n"
    )

    transform("rid.transform.output")

    assert keys(client, output_build) == [
        "artifacts/index.html",
        "datasets/00000000.parquet",
        "meta/columns.json",
    ]
    data = dataset(client, output_build)
    assert data["double"].to_list() == [20, 40, 60]
    lineage = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{output_build}meta/columns.json")[
            "Body"
        ].read()
    )
    assert lineage["double"] == [
        {"dataset_rid": "rid.transform.input", "column": "value"}
    ]


def test_incremental_build_reads_the_previous_dataset(build):
    client, code, _, output_build = build
    (code / "output.py").write_text(
        "import polars as pl\n"
        "from pyrunner_lib import Previous\n"
        "def transform(input, history: Previous):\n"
        "    assert history.exists and history.ds_meta.row_count == 1\n"
        "    return pl.concat([history.data, input])\n"
    )

    transform("rid.transform.output")

    assert dataset(client, output_build)["id"].to_list() == [0, 1, 2, 3]
    # The previous dataset is the transform's own
    lineage = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{output_build}meta/columns.json")[
            "Body"
        ].read()
    )
    assert lineage["id"] == [
        {"dataset_rid": "rid.transform.input", "column": "id"},
        {"dataset_rid": "rid.transform.output", "column": "id"},
    ]


APPEND = (
    "from pyrunner_lib import Previous\n"
    "def transform(input, history: Previous):\n"
    "    return history.append(input)\n"
)


def test_append_keeps_the_previous_files_and_adds_the_new_rows(build):
    client, code, _, output_build = build
    previous = json.loads(os.environ["DS_PREVIOUS"])["uri"].removeprefix(
        f"s3://{BUCKET}/"
    )
    (code / "output.py").write_text(APPEND)

    transform("rid.transform.output")

    # The previous file is there unchanged, byte for byte; the input's rows are a new file
    assert keys(client, output_build) == [
        "datasets/append-000001/00000000.parquet",
        "datasets/data.parquet",
        "meta/columns.json",
    ]
    copied = client.get_object(
        Bucket=BUCKET, Key=f"{output_build}datasets/data.parquet"
    )
    original = client.get_object(Bucket=BUCKET, Key=f"{previous}/datasets/data.parquet")
    assert copied["Body"].read() == original["Body"].read()
    assert sorted(dataset(client, output_build)["id"].to_list()) == [0, 1, 2, 3]
    lineage = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{output_build}meta/columns.json")[
            "Body"
        ].read()
    )
    assert lineage["id"] == [
        {"dataset_rid": "rid.transform.input", "column": "id"},
        {"dataset_rid": "rid.transform.output", "column": "id"},
    ]


def test_append_copies_with_a_timeout_that_large_files_need(build, monkeypatch):
    _, code, _, _ = build
    (code / "output.py").write_text(APPEND)
    timeouts = []
    create = pafs.S3FileSystem
    monkeypatch.setattr(
        pafs,
        "S3FileSystem",
        lambda **options: timeouts.append(options["request_timeout"])
        or create(**options),
    )

    transform("rid.transform.output")

    # pyarrow's own limit of 3 seconds fails the copy of a file above about 1 GiB
    assert 3600 in timeouts


def test_previous_is_found_when_annotations_are_kept_as_text(build):
    client, code, _, output_build = build
    (code / "output.py").write_text("from __future__ import annotations\n" + APPEND)

    transform("rid.transform.output")

    assert sorted(dataset(client, output_build)["id"].to_list()) == [0, 1, 2, 3]


def test_each_append_gets_the_next_folder(build):
    client, code, _, output_build = build
    previous = json.loads(os.environ["DS_PREVIOUS"])["uri"].removeprefix(
        f"s3://{BUCKET}/"
    )
    buffer = io.BytesIO()
    pl.DataFrame({"id": [-1], "value": [-1]}).write_parquet(buffer)
    client.put_object(
        Bucket=BUCKET,
        Key=f"{previous}/datasets/append-000001/00000000.parquet",
        Body=buffer.getvalue(),
    )
    (code / "output.py").write_text(APPEND)

    transform("rid.transform.output")

    assert keys(client, f"{output_build}datasets/") == [
        "append-000001/00000000.parquet",
        "append-000002/00000000.parquet",
        "data.parquet",
    ]


def test_append_on_the_first_build_writes_the_rows(build, monkeypatch):
    client, code, _, output_build = build
    monkeypatch.delenv("DS_PREVIOUS")
    (code / "output.py").write_text(
        "from pyrunner_lib import Previous\n"
        "def transform(input, history: Previous):\n"
        "    assert not history.exists\n"
        "    return history.append(input)\n"
    )

    transform("rid.transform.output")

    assert keys(client, f"{output_build}datasets/") == ["00000000.parquet"]
    assert dataset(client, output_build)["id"].to_list() == [1, 2, 3]


def test_append_of_other_columns_fails(build):
    client, code, _, output_build = build
    (code / "output.py").write_text(
        "import polars as pl\n"
        "from pyrunner_lib import Previous\n"
        "def transform(input, history: Previous):\n"
        "    return history.append(input.with_columns(extra=pl.lit(1)))\n"
    )

    with pytest.raises(
        DataWriteError, match="columns and types of the previous dataset"
    ):
        transform("rid.transform.output")

    assert keys(client, f"{output_build}datasets/") == []


def test_previous_build_that_does_not_exist_cannot_be_read(build, monkeypatch):
    monkeypatch.delenv("DS_PREVIOUS")
    history = Previous("rid.transform.output")

    assert not history.exists
    with pytest.raises(PyrunnerError, match="no previous build"):
        _ = history.data


def test_health_checks_read_the_written_dataset_and_upload_the_report(build):
    client, code, _, output_build = build
    (code / "output.py").write_text(
        "from pyrunner_lib.health_check import Check\n"
        "def transform(input):\n"
        "    return input, [Check('value').valid_range(min_val=15, severity='warn')]\n"
    )

    transform("rid.transform.output")

    report = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{output_build}meta/health_report.json")[
            "Body"
        ].read()
    )
    assert report["summary"]["failed"] == 1


def test_draft_build_reads_the_first_rows(build, monkeypatch):
    client, code, _, output_build = build
    monkeypatch.setenv("IS_DRAFT_BUILD", "true")
    monkeypatch.setenv("DRAFT_READ_LIMIT", "2")
    (code / "_config.json").write_text(
        json.dumps({"python": "3.11", "dfType": "pandas", "packages": []})
    )
    (code / "output.py").write_text("def transform(input):\n    return input\n")

    transform("rid.transform.output")

    data = dataset(client, output_build)
    assert data["id"].to_list() == [1, 2]


def test_pandas_build_reads_and_writes_s3(build):
    client, code, _, output_build = build
    (code / "_config.json").write_text(
        json.dumps({"python": "3.11", "dfType": "pandas", "packages": []})
    )
    (code / "output.py").write_text(
        "def transform(input):\n    input['double'] = input['value'] * 2\n    return input\n"
    )

    transform("rid.transform.output")

    data = dataset(client, output_build)
    assert data["double"].to_list() == [20, 40, 60]


def test_failed_build_still_uploads_artifacts(build):
    client, code, _, output_build = build
    (code / "output.py").write_text(
        "import os\n"
        "def transform(input):\n"
        "    open(os.path.join(os.environ['ARTIFACT_FOLDER'], 'debug.txt'), 'w').write('state')\n"
        "    raise ValueError('boom')\n"
    )

    with pytest.raises(Exception, match="boom"):
        transform("rid.transform.output")

    assert keys(client, output_build) == ["artifacts/debug.txt"]


def test_build_cannot_write_to_its_input(build):
    client, code, input_build, _ = build
    bucket_key = f"{BUCKET}/{input_build}datasets/00000000.parquet"
    (code / "output.py").write_text(
        "import os, polars as pl\n"
        "def transform(input):\n"
        f"    input.collect().write_parquet('s3://{bucket_key}', storage_options=input.artifacts.storage_options)\n"
        "    return input\n"
    )

    with pytest.raises(OSError, match="lacked the necessary privileges"):
        transform("rid.transform.output")

    data = dataset(client, input_build)
    assert data.columns == ["id", "value"]


def test_input_that_was_not_granted_is_not_readable(build, monkeypatch):
    _, code, _, _ = build
    monkeypatch.setenv("DS_INPUTS", "{}")
    (code / "output.py").write_text("def transform(input):\n    return input\n")

    with pytest.raises(DataLoadError, match="No access"):
        transform("rid.transform.output")


def test_build_reads_a_csv_from_the_artifacts_of_its_input(build):
    """Polars 1.41 to 1.43 hang in the lineage trace of a CSV scanned from S3, unless its schema is resolved first."""
    client, code, input_build, output_build = build
    client.put_object(
        Bucket=BUCKET,
        Key=f"{input_build}artifacts/raw/batch.csv",
        Body=b"id,name\n1,a\n2,b\n",
    )
    (code / "output.py").write_text(
        "import polars as pl\n"
        "def transform(input):\n"
        "    artifacts = input.artifacts\n"
        "    return pl.scan_csv(artifacts.uri('raw/batch.csv'), storage_options=artifacts.storage_options)\n"
    )

    # In a process of its own: a hanging build could not be stopped inside this one
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pyrunner_lib import transform; transform('rid.transform.output')",
        ],
        check=True,
        timeout=60,
    )

    data = dataset(client, output_build)
    assert data["name"].to_list() == ["a", "b"]


def test_artifacts_of_an_input_are_listed_and_fetched(build):
    client, code, input_build, output_build = build
    client.put_object(
        Bucket=BUCKET, Key=f"{input_build}artifacts/raw/notes.txt", Body=b"notes"
    )
    client.put_object(
        Bucket=BUCKET, Key=f"{input_build}artifacts/raw/batch.csv", Body=b"a\n1\n"
    )
    client.put_object(
        Bucket=BUCKET, Key=f"{input_build}artifacts/reference.bin", Body=b"\x00\x01"
    )
    (code / "output.py").write_text(
        "import os\n"
        "from pyrunner_lib import build\n"
        "def transform(input):\n"
        "    artifacts = input.artifacts\n"
        "    assert artifacts.list() == ['raw/batch.csv', 'raw/notes.txt', 'reference.bin']\n"
        "    assert artifacts.list('raw/*.csv') == ['raw/batch.csv']\n"
        "    assert artifacts.get('raw/notes.txt').read_text() == 'notes'\n"
        "    assert artifacts.get('raw/notes.txt') is artifacts.get('raw/notes.txt')\n"
        "    assert sorted(p.name for p in artifacts.get('raw').iterdir()) == ['batch.csv', 'notes.txt']\n"
        "    folder = artifacts.get()\n"
        "    assert sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()) == "
        "['raw/batch.csv', 'raw/notes.txt', 'reference.bin']\n"
        "    (build.artifacts / 'copy.txt').write_text(artifacts.get('raw/notes.txt').read_text())\n"
        "    assert build.id == os.environ['DS_BUILD_ID'] and build.branch == 'main' and not build.is_draft\n"
        "    return input\n"
    )

    transform("rid.transform.output")

    copy = client.get_object(Bucket=BUCKET, Key=f"{output_build}artifacts/copy.txt")
    assert copy["Body"].read() == b"notes"


def test_artifacts_that_do_not_exist(build):
    _, _, input_build, _ = build
    artifacts = Artifacts(f"s3://{BUCKET}/{input_build}artifacts")

    assert artifacts.list() == []
    with pytest.raises(FileNotFoundError):
        artifacts.get("missing.json")
    with pytest.raises(FileNotFoundError):
        artifacts.get()


def test_artifacts_of_the_previous_build_are_fetched(build):
    client, _, _, _ = build
    previous = json.loads(os.environ["DS_PREVIOUS"])["uri"].removeprefix(
        f"s3://{BUCKET}/"
    )
    client.put_object(Bucket=BUCKET, Key=f"{previous}/artifacts/state.json", Body=b"{}")

    # A folder with files only: nothing but the folder itself has to be created
    folder = Previous("rid.transform.output").artifacts.get()

    assert os.listdir(folder) == ["state.json"]


def test_artifacts_of_a_build_that_was_not_granted_cannot_be_fetched(build):
    client, _, _, _ = build
    foreign = f"builds/rid.build.{uuid.uuid4()}/"
    client.put_object(
        Bucket=BUCKET, Key=f"{foreign}artifacts/secret.txt", Body=b"secret"
    )

    with pytest.raises(OSError):
        Artifacts(f"s3://{BUCKET}/{foreign}artifacts").get()


def test_build_storage_gives_direct_access_to_the_build_folder(build):
    client, _, _, output_build = build

    pl.DataFrame({"x": [1]}).write_parquet(
        running_build.storage.uri + "/artifacts/direct.parquet",
        storage_options=running_build.storage.options,
    )

    assert keys(client, output_build) == ["artifacts/direct.parquet"]
