"""Runs transforms against a real S3 server with the credentials DataSpace hands to builds.

Skipped unless DS_TEST_S3_ENDPOINT points to an S3 server with STS (e.g. RustFS) whose root
credentials are DS_TEST_S3_ACCESS_KEY and DS_TEST_S3_SECRET_KEY. The bucket must exist.
"""

import io
import json
import os
import uuid

import polars as pl
import pytest

from pyrunner_lib import DataLoadError, transform

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
    buffer = io.BytesIO()
    pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).write_parquet(buffer)
    client.put_object(
        Bucket=BUCKET, Key=f"{input_build}datasets/data.parquet", Body=buffer.getvalue()
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
        "PREVIOUS_DATASET_PATH": f"s3://{BUCKET}/{previous_build}datasets/data.parquet",
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
        "datasets/data.parquet",
        "meta/columns.json",
    ]
    data = client.get_object(Bucket=BUCKET, Key=f"{output_build}datasets/data.parquet")[
        "Body"
    ].read()
    assert pl.read_parquet(io.BytesIO(data))["double"].to_list() == [20, 40, 60]
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
        "import os, polars as pl\n"
        "from pyrunner_lib import storage_options\n"
        "def transform(input):\n"
        "    previous = os.environ['PREVIOUS_DATASET_PATH']\n"
        "    return pl.concat([pl.scan_parquet(previous, storage_options=storage_options(previous)), input])\n"
    )

    transform("rid.transform.output")

    data = client.get_object(Bucket=BUCKET, Key=f"{output_build}datasets/data.parquet")[
        "Body"
    ].read()
    assert pl.read_parquet(io.BytesIO(data))["id"].to_list() == [0, 1, 2, 3]
    # The previous dataset is the transform's own, as its path said before builds lived in S3
    lineage = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{output_build}meta/columns.json")[
            "Body"
        ].read()
    )
    assert lineage["id"] == [
        {"dataset_rid": "rid.transform.input", "column": "id"},
        {"dataset_rid": "rid.transform.output", "column": "id"},
    ]


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

    data = client.get_object(Bucket=BUCKET, Key=f"{output_build}datasets/data.parquet")[
        "Body"
    ].read()
    assert pl.read_parquet(io.BytesIO(data))["id"].to_list() == [1, 2]


def test_pandas_build_reads_and_writes_s3(build):
    client, code, _, output_build = build
    (code / "_config.json").write_text(
        json.dumps({"python": "3.11", "dfType": "pandas", "packages": []})
    )
    (code / "output.py").write_text(
        "def transform(input):\n    input['double'] = input['value'] * 2\n    return input\n"
    )

    transform("rid.transform.output")

    data = client.get_object(Bucket=BUCKET, Key=f"{output_build}datasets/data.parquet")[
        "Body"
    ].read()
    assert pl.read_parquet(io.BytesIO(data))["double"].to_list() == [20, 40, 60]


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
    bucket_key = f"{BUCKET}/{input_build}datasets/data.parquet"
    (code / "output.py").write_text(
        "import os, polars as pl\n"
        "def transform(input):\n"
        f"    input.collect().write_parquet('s3://{bucket_key}', storage_options=input.ds_meta.storage_options)\n"
        "    return input\n"
    )

    with pytest.raises(OSError, match="lacked the necessary privileges"):
        transform("rid.transform.output")

    data = client.get_object(Bucket=BUCKET, Key=f"{input_build}datasets/data.parquet")[
        "Body"
    ].read()
    assert pl.read_parquet(io.BytesIO(data)).columns == ["id", "value"]


def test_input_that_was_not_granted_is_not_readable(build, monkeypatch):
    _, code, _, _ = build
    monkeypatch.setenv("DS_INPUTS", "{}")
    (code / "output.py").write_text("def transform(input):\n    return input\n")

    with pytest.raises(DataLoadError, match="No access"):
        transform("rid.transform.output")
