"""Tests for pyrunner_lib module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from pyrunner_lib.pyrunner_lib import (
    Config,
    ConfigurationError,
    Connection,
    DataLoadError,
    DataWriteError,
    ModuleLoadError,
    TransformNotFoundError,
    load_json,
    load_module,
    path_to_name,
    read_parquet_files,
    storage_options,
    transform,
    write_column_trace_file,
    write_df_to_parquet,
)


class TestConfig:
    """Test Config model."""
    
    def test_config_creation(self):
        """Test Config model creation."""
        config = Config(
            python="3.8",
            dfType="polars",
            packages=["polars", "pandas"]
        )
        assert config.python == "3.8"
        assert config.dfType == "polars"
        assert config.packages == ["polars", "pandas"]


class TestConnection:
    """Test Connection model."""
    
    def test_connection_creation(self):
        """Test Connection model creation."""
        conn = Connection(
            id="test_transform",
            path="/path/to/transform.py",
            inputs=["input1", "input2"]
        )
        assert conn.id == "test_transform"
        assert conn.path == "/path/to/transform.py"
        assert conn.inputs == ["input1", "input2"]


class TestLoadJson:
    """Test load_json function."""
    
    def test_load_json_success(self):
        """Test successful JSON loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "python": "3.8",
                "dfType": "polars",
                "packages": ["polars"]
            }, f)
            f.flush()
            
            try:
                result = load_json(f.name, Config)
                assert isinstance(result, Config)
                assert result.python == "3.8"
                assert result.dfType == "polars"
            finally:
                os.unlink(f.name)
    
    def test_load_json_file_not_found(self):
        """Test JSON file not found error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_json("nonexistent.json", Config)
    
    def test_load_json_invalid_json(self):
        """Test invalid JSON error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError, match="Invalid JSON"):
                    load_json(f.name, Config)
            finally:
                os.unlink(f.name)
    
    def test_load_json_validation_error(self):
        """Test validation error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "data"}, f)
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError, match="Invalid configuration"):
                    load_json(f.name, Config)
            finally:
                os.unlink(f.name)


class TestPathToName:
    """Test path_to_name function."""
    
    def test_path_to_name(self):
        """Test path to name conversion."""
        assert path_to_name("/path/to/file.py") == "file"
        assert path_to_name("file.py") == "file"
        assert path_to_name("/path/to/file") == "file"


class TestWriteColumnTraceFile:
    """Test write_column_trace_file function."""
    
    def test_write_column_trace_file_success(self):
        """Test successful column trace file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple LazyFrame
            df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            
            with patch('pyrunner_lib.pyrunner_lib.trace_lazyframe_sources') as mock_trace:
                mock_trace.return_value = {"a": ["source1"], "b": ["source2"]}
                
                write_column_trace_file(df, "test_transform", temp_dir)
                
                # Check that the file was created
                trace_file = Path(temp_dir) / "data" / "test_transform" / "meta" / "columns.json"
                assert trace_file.exists()
                
                # Check file contents
                with open(trace_file) as f:
                    data = json.load(f)
                    assert data == {"a": ["source1"], "b": ["source2"]}
                mock_trace.assert_called_once_with(df, {}, storage_options)
    

class TestWriteDfToParquet:
    """Test write_df_to_parquet function."""

    @pytest.mark.parametrize(
        "df",
        [
            pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        ],
        ids=["polars lazy", "polars eager", "pandas"],
    )
    def test_write_creates_the_dataset_folder(self, df, tmp_path):
        folder = tmp_path / "data" / "rid" / "datasets"

        with patch("pyrunner_lib.pyrunner_lib.write_column_trace_file") as mock_trace:
            write_df_to_parquet(df, str(folder), "test_transform", str(tmp_path))

        assert [p.name for p in folder.iterdir()] == ["00000000.parquet"]
        assert pl.read_parquet(folder / "*.parquet").to_dict(as_series=False) == {"a": [1, 2, 3], "b": [4, 5, 6]}
        assert mock_trace.call_count == (1 if isinstance(df, pl.LazyFrame) else 0)

    def test_write_splits_a_large_output_into_files(self, tmp_path):
        folder = tmp_path / "datasets"

        # Each row is 8 bytes in memory, so the 8 MB of this frame fill eight files
        with patch("pyrunner_lib.pyrunner_lib.DATASET_PART_BYTES", 1_000_000):
            write_df_to_parquet(pl.DataFrame({"a": range(1_000_000)}), str(folder), "test_transform", str(tmp_path))

        assert len(list(folder.iterdir())) == 8
        assert pl.read_parquet(folder / "*.parquet")["a"].to_list() == list(range(1_000_000))

    def test_write_replaces_the_files_of_an_earlier_local_run(self, tmp_path):
        folder = tmp_path / "datasets"
        folder.mkdir()
        pl.DataFrame({"a": [9]}).write_parquet(folder / "data.parquet")

        write_df_to_parquet(pl.DataFrame({"a": [1]}), str(folder), "test_transform", str(tmp_path))

        assert pl.read_parquet(folder / "*.parquet")["a"].to_list() == [1]

    def test_write_unsupported_type(self, tmp_path):
        with pytest.raises(DataWriteError, match="Failed to write Parquet file"):
            write_df_to_parquet("not a dataframe", str(tmp_path / "datasets"), "test_transform", str(tmp_path))

    def test_write_error(self, tmp_path):
        blocked = tmp_path / "file"
        blocked.write_text("not a folder")

        with pytest.raises(DataWriteError, match="Failed to write Parquet file"):
            write_df_to_parquet(pl.DataFrame({"a": [1]}), str(blocked / "datasets"), "test_transform", str(tmp_path))


class TestLoadModule:
    """Test load_module function."""
    
    def test_load_module_success(self):
        """Test successful module loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def transform(df):
    return df
""")
            f.flush()
            
            try:
                module = load_module("test_module", f.name)
                assert hasattr(module, 'transform')
                assert callable(module.transform)
            finally:
                os.unlink(f.name)
    
    def test_load_module_file_not_found(self):
        """Test module loading with non-existent file."""
        with pytest.raises(ModuleLoadError, match="Failed to load module"):
            load_module("test_module", "nonexistent.py")
    
    def test_load_module_syntax_error(self):
        """Test module loading with syntax error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax")
            f.flush()
            
            try:
                with pytest.raises(ModuleLoadError, match="Failed to load module"):
                    load_module("test_module", f.name)
            finally:
                os.unlink(f.name)


class TestReadParquetFiles:
    """Test read_parquet_files function."""
    
    def test_read_parquet_files_polars(self):
        """Test reading parquet files with Polars."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            parquet_path = os.path.join(temp_dir, "data.parquet")
            df.write_parquet(parquet_path)
            
            # Create connection
            conn = Connection(
                id="test_conn",
                path="/path/to/transform.py",
                inputs=["input1"]
            )
            connections = {"input1": conn}
            
            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'), patch('pyrunner_lib.pyrunner_lib.pl.scan_parquet') as mock_scan:
                mock_scan.return_value = pl.LazyFrame({"a": [1, 2, 3]})
                    
                result = read_parquet_files(connections, {"input1": "input1"}, temp_dir)
                    
                assert "input1" in result
                mock_scan.assert_called_once()
    
    def test_read_parquet_files_pandas(self, tmp_path):
        """pandas reads every Parquet file of the dataset folder, also in subfolders, and nothing else."""
        folder = tmp_path / "data" / "test_conn" / "datasets"
        (folder / "year=2026").mkdir(parents=True)
        pd.DataFrame({"a": [1, 2]}).to_parquet(folder / "data.parquet")
        pd.DataFrame({"a": [3]}).to_parquet(folder / "year=2026" / "00000000.parquet")
        (folder / "data.csv").write_text("a\n99\n")
        connections = {"input1": Connection(id="test_conn", path="/path/to/transform.py", inputs=[])}

        with patch("pyrunner_lib.pyrunner_lib.DF_TYPE", "pandas"):
            result = read_parquet_files(connections, {"input1": "input1"}, str(tmp_path))

        assert sorted(result["input1"]["a"]) == [1, 2, 3]

    def test_read_parquet_files_missing_connection(self):
        """Test reading parquet files with missing connection."""
        connections = {}
        
        with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'), pytest.raises(DataLoadError, match="Input parameter 'missing' not found"):
            read_parquet_files(connections, {"missing": "missing"}, "/tmp")

    def test_read_parquet_files_draft_build_limit(self):
        """A draft build cuts each input to DRAFT_READ_LIMIT rows; other builds read everything."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rid = "test-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            pl.DataFrame({"a": range(10)}).write_parquet(os.path.join(temp_dir, "data", rid, "datasets", "data.parquet"))
            connections = {"input1": Connection(id=rid, path="test.py", inputs=[])}
            draft = {"IS_DRAFT_BUILD": "true", "DRAFT_READ_LIMIT": "3"}

            for df_type in ("polars", "pandas"):
                with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', df_type):
                    with patch.dict(os.environ, draft):
                        result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                        assert len(result.collect() if df_type == "polars" else result) == 3

                    with patch.dict(os.environ, {}, clear=True):
                        result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                        assert len(result.collect() if df_type == "polars" else result) == 10

            # Without a configured limit a draft build reads DEFAULT_DRAFT_READ_LIMIT rows
            with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true"}, clear=True), patch(
                'pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'
            ), patch('pyrunner_lib.pyrunner_lib.DEFAULT_DRAFT_READ_LIMIT', 4):
                result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                assert len(result.collect()) == 4

    def test_read_parquet_files_draft_limit_minus_one_reads_everything(self):
        """-1 turns the draft limit off."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rid = "test-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            pl.DataFrame({"a": range(10)}).write_parquet(os.path.join(temp_dir, "data", rid, "datasets", "data.parquet"))
            connections = {"input1": Connection(id=rid, path="test.py", inputs=[])}
            for df_type in ("polars", "pandas"):
                with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true", "DRAFT_READ_LIMIT": "-1"}), patch(
                    'pyrunner_lib.pyrunner_lib.DF_TYPE', df_type
                ), patch('pyrunner_lib.pyrunner_lib.DEFAULT_DRAFT_READ_LIMIT', 3):
                    result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                    assert len(result.collect() if df_type == "polars" else result) == 10

    def test_read_parquet_files_draft_limit_must_be_positive(self):
        """Zero and negatives other than -1 are refused; pandas would read head(-n) as all but the last n rows."""
        connections = {"input1": Connection(id="rid", path="test.py", inputs=[])}
        for limit in ("0", "-2", "abc", ""):
            with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true", "DRAFT_READ_LIMIT": limit}), patch(
                'pyrunner_lib.pyrunner_lib.DF_TYPE', 'pandas'
            ), pytest.raises(ConfigurationError):
                read_parquet_files(connections, {"input1": "rid"}, "/tmp")

    def test_read_parquet_files_pandas_draft_reads_only_the_limited_rows(self):
        """A pandas draft build reads batch by batch across row groups and never loads the whole file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rid = "test-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            pl.DataFrame({"a": range(100), "b": [str(i) for i in range(100)]}).write_parquet(
                os.path.join(temp_dir, "data", rid, "datasets", "data.parquet"), row_group_size=7
            )
            connections = {"input1": Connection(id=rid, path="test.py", inputs=[])}

            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'pandas'), patch(
                'pandas.read_parquet', side_effect=AssertionError("loaded the whole file")
            ):
                with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true", "DRAFT_READ_LIMIT": "20"}):
                    result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                    assert list(result["a"]) == list(range(20))

                with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true", "DRAFT_READ_LIMIT": "500"}):
                    result = read_parquet_files(connections, {"input1": rid}, temp_dir)["input1"]
                    assert len(result) == 100 and list(result.columns) == ["a", "b"]

    def test_read_parquet_files_explicit_rid_fallback(self):
        """Test that explicit RIDs can fall back while others cannot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file for the "explicit" RID
            df = pl.DataFrame({"a": [1, 2, 3]})
            rid = "explicit-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            df.write_parquet(os.path.join(temp_dir, "data", rid, "datasets", "data.parquet"))
            
            connections = {} # Empty connections
            
            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                # 1. Explicit RID should work even if not in connections
                param_mapping = {"df": rid}
                explicit_params = {"df"}
                result = read_parquet_files(connections, param_mapping, temp_dir, explicit_params=explicit_params)
                assert "df" in result
                
                # 2. Non-explicit RID should fail if not in connections
                param_mapping = {"other": "some-rid"}
                with pytest.raises(DataLoadError, match="Input parameter 'other' not found"):
                    read_parquet_files(connections, param_mapping, temp_dir, explicit_params=set())
    
    def test_read_parquet_files_configuration_not_loaded(self):
        """Test reading parquet files without configuration."""
        with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', None), pytest.raises(ConfigurationError, match="Configuration not loaded"):
            read_parquet_files({}, {}, "/tmp")


class TestTransform:
    """Test transform function."""
    
    def test_transform_success(self):
        """Test successful transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            # Create test transform module
            transform_file = os.path.join(temp_dir, "transform.py")
            with open(transform_file, 'w') as f:
                f.write("""
def transform(input1):
    return input1
""")
            
                with open(connections_file, 'w') as f:
                    json.dump([{
                        "id": "test_transform",
                        "path": transform_file,
                        "inputs": ["input1"]
                    }], f)
            
            # Mock the file paths
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file), patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file), patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
                mock_read.return_value = {"input1": pl.LazyFrame({"a": [1, 2, 3]})}
                        
                with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet') as mock_write:
                    transform("test_transform", temp_dir)
                            
                    mock_read.assert_called_once()
                    mock_write.assert_called_once()
    
    def test_transform_not_found(self):
        """Test transform with non-existent transform ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            with open(connections_file, 'w') as f:
                json.dump([{
                    "id": "other_transform",
                    "path": "/path/to/transform.py",
                    "inputs": []
                }], f)
            
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file), patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file), pytest.raises(TransformNotFoundError, match="Transform with ID 'test_transform' not found"):
                transform("test_transform", temp_dir)
    
    def test_transform_module_no_transform_function(self):
        """Test transform with module missing transform function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            with open(connections_file, 'w') as f:
                json.dump([{
                    "id": "test_transform",
                    "path": "/path/to/transform.py",
                    "inputs": []
                }], f)
            
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file), patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file), patch('pyrunner_lib.pyrunner_lib.load_module') as mock_load:
                mock_module = MagicMock()
                # Remove transform attribute
                del mock_module.transform
                mock_load.return_value = mock_module
                        
                with pytest.raises(ModuleLoadError, match="does not contain a 'transform' function"):
                    transform("test_transform", temp_dir)


class TestStorageOptions:
    def test_picks_the_credentials_of_the_folder(self, monkeypatch):
        from pyrunner_lib.pyrunner_lib import storage_options

        monkeypatch.setenv("DS_S3_ENDPOINT", "http://storage:9000")
        monkeypatch.setenv("DS_S3_REGION", "us-east-1")
        monkeypatch.setenv("DS_S3_ACCESS_KEY_ID", "own-key")
        monkeypatch.setenv("DS_S3_SECRET_ACCESS_KEY", "own-secret")
        monkeypatch.setenv("DS_S3_SESSION_TOKEN", "own-token")
        monkeypatch.setenv("DS_INPUTS", json.dumps({"rid.transform.in": {
            "uri": "s3://dataspace/builds/rid.build.in",
            "accessKeyId": "in-key", "secretAccessKey": "in-secret", "sessionToken": "in-token",
        }}))

        assert storage_options("/local/folder") is None
        assert storage_options("s3://dataspace/builds/rid.build.own/x")["aws_allow_http"] == "true"
        assert storage_options("s3://dataspace/builds/rid.build.in/artifacts/a.tsv")["aws_access_key_id"] == "in-key"
        assert storage_options("s3://dataspace/builds/rid.build.in")["aws_access_key_id"] == "in-key"
        # A build whose id merely starts with the input's id is not the input
        assert storage_options("s3://dataspace/builds/rid.build.in2/x")["aws_access_key_id"] == "own-key"
        assert storage_options("s3://dataspace/builds/rid.build.own/x")["aws_access_key_id"] == "own-key"
