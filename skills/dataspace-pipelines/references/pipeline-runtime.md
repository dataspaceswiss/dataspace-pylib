# DataSpace Pipeline Runtime Reference

## Transform structure

Place business logic in Python transform files, conventionally below `src/`. Define one function named `transform`. DataSpace refuses to save a file that defines more than one.

Start the file with a docstring that says what the transform produces and from what, and keep it current with the logic.

The runner populates transform parameters with upstream data: a Polars `LazyFrame` per input when `_config.json` sets `"dfType": "polars"`, or an eager pandas `DataFrame` when it sets `"pandas"`. Return one of:

- `polars.LazyFrame` (preferred for optimization, streaming, and lineage)
- `polars.DataFrame`
- `pandas.DataFrame`

```python
import polars as pl


def transform(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.filter(pl.col("active"))
```

The runner automatically persists the returned dataframe as the transform's parquet dataset. Do not write that dataset manually.

### Lazy Polars first

Write transform logic as Polars lazy expressions and return the `LazyFrame`. Two things depend on it:

- **Performance.** Polars optimizes the whole lazy query, and the runner writes a returned `LazyFrame` with a streaming sink, so the dataset does not have to fit in memory.
- **Column lineage.** The runner derives which upstream columns feed each output column from the lazy query plan. A returned `polars.DataFrame` or `pandas.DataFrame` has no plan, so DataSpace records no column lineage for that transform.

Lineage is also lost for any column that passes through Python code inside the query, such as `map_elements` or `map_batches`, because Polars cannot see into the function. Prefer native expressions, and when no expression exists, apply the function only to the columns that need it.

Avoid collecting in the middle of a transform and continuing eagerly, converting to pandas, and looping over rows in Python. When a side task needs materialized data, such as a chart, collect a bounded projection for that task and still return the lazy frame.

### Let DataSpace keep the records

Do not add a bookkeeping layer to a pipeline. DataSpace records the following for every build and shows its history, without any code in the transform:

- status, start and end time, duration, who or which schedule started it, and the commit it ran;
- the exact input snapshots it read;
- logs and resource metrics;
- the resulting dataset's row count, column count, file size, and schema;
- column lineage and health check results.

Writing run logs, manifests, row-count or schema summaries, "last run" markers, or audit trails as JSON files, artifacts, or extra datasets duplicates this, can disagree with it, and leaves more code to maintain. Use what the platform provides instead:

| Need | Use |
| --- | --- |
| Facts about an input, such as its snapshot, row count, or columns | `input_df.meta` |
| What the previous run produced | A `Previous` parameter |
| A rule the data must satisfy | Health checks |
| A file a person or a downstream transform consumes | An artifact |

Write an artifact when it is a deliverable. Do not write one to remember what the pipeline did.

### Readable formatting

Transform code is mostly long chains of `filter`, `select`, `with_columns`, joins, and aggregations, and it is read far more often than it is written. Lay it out so the pipeline reads from top to bottom:

- Wrap a chain in parentheses and give each method call its own line.
- Give each expression its own line inside `select`, `with_columns`, and `agg`.
- Name an intermediate frame when a chain has distinct stages.

```python
import polars as pl


def transform(orders: pl.LazyFrame, customers: pl.LazyFrame) -> pl.LazyFrame:
    completed_orders = (
        orders.filter(pl.col("status") == "completed")
        .filter(pl.col("ordered_at") >= pl.date(2025, 1, 1))
        .select(
            "order_id",
            "customer_id",
            (pl.col("quantity") * pl.col("unit_price")).alias("revenue"),
        )
    )

    return (
        completed_orders.join(customers, on="customer_id", how="left")
        .group_by("customer_id", "segment")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("order_id").n_unique().alias("order_count"),
        )
        .sort("total_revenue", descending=True)
    )
```

A formatter or lint configuration in the repository, the project's own conventions, or an explicit request from the user takes precedence over this layout.

## Project configuration

`_config.json` at the repository root defines the build environment:

```json
{
  "confVersion": "1.0",
  "python": "3.11",
  "dfType": "polars",
  "packages": [
    "plotly",
    "git+https://github.com/dataspaceswiss/dataspace-pylib@<commit of feature/s3-storage>#egg=dataspace-pylib"
  ]
}
```

- `python` and `packages` define the image every build of the workspace runs in. A change to them makes the next build create a new image first, which takes noticeably longer than a normal build. Keep `dataspace-pylib` in `packages`, pinned to a tag.
- `dfType` decides the input type for every transform of the workspace. Keep `polars` unless the repository already uses `pandas`.
- `draftReadLimit` is optional: the number of rows a draft build reads from each input (1,000,000 when unset), or `-1` to read every row. It needs `dataspace-pylib` 0.6.1 or later; 0.6.0 always reads 1,000,000.

## Connection mapping

Use `_connections.json` to resolve transform RIDs to source files and upstream transforms.

```json
{
  "id": "rid.transform.uuid",
  "path": "src/path/to/transform.py",
  "inputs": ["rid.transform.upstream"]
}
```

Whenever a mapped Python file moves or is renamed, update its `path` in `_connections.json` in the same change. A stale path breaks execution.

DataSpace recognizes two input declaration forms.

Use a bare parameter for a transform of the same workspace; it is found by its name:

```python
def transform(input):
    return input
```

The runner resolves an unannotated parameter by matching its name to an upstream connection's filename stem. Renaming `src/source_clean.py` to `src/source/clean.py` changes that legacy name from `source_clean` to `clean` and can break downstream parameters even when the RID is unchanged. Search downstream transform signatures whenever a rename changes a stem.

Bind a parameter explicitly by RID with a parameter annotation:

```python
from dataspace_pylib import Transform


def transform(source_data: Transform("rid.transform.source")):
    return source_data
```

An input from a linked foreign workspace is always declared by its RID; the resolved connection preserves cross-workspace lineage. A parameter of the type `Previous` is not an input (see Incremental builds).

`Transform("WORKSPACE.transform")`, for example `orders: Transform("INSTACART.orders")`, is a planned DataSpace syntax and is not yet supported. Until it is implemented, use the foreign transform's RID in the annotation.

### How DataSpace reads input declarations

DataSpace works out a transform's inputs by parsing the syntax of the source file, without executing it. This happens when the file is saved in the DataSpace editor, and the result is stored as that transform's `inputs` in `_connections.json`. A build mounts exactly those inputs. The runner, in contrast, executes the module, so a declaration that computes its RID can run locally and still be wired to the wrong input, or to none, on the platform.

| Declaration | What DataSpace 1.13 reads |
| --- | --- |
| `name: Transform("rid.transform.x")`, also inside `Annotated[...]`, and `dataspace_pylib.Transform(...)` | Input bound to that RID. |
| `name: Transform(SOME_RID)` | Input bound to the value of `SOME_RID` when it is assigned exactly once in the same file, at module level, to a plain string. Otherwise the save is rejected. |
| `name: Transform("rid." + suffix)`, an f-string, or any other computed value | The save is rejected. |
| `name` | Input bound by name. |
| `name=Transform("rid.transform.x")` as a default value | Bound by name, not by RID. |

Write the RID as a plain string literal directly inside the call, import `Transform` with `from dataspace_pylib import Transform`, and give input parameters no default values. Older DataSpace versions silently bound constants, computed values, and some of the other forms by name, so that form is also the only one every version reads correctly.

Nothing re-derives the inputs when changes arrive over git. Whenever you add, remove, or rebind an input in code, make the same change to that transform's `inputs` in `_connections.json`, using the same RIDs.

The conventional entry point is:

```python
import dataspace_pylib

dataspace_pylib.main()
```

## The running build

Build data lives in S3, but a transform never handles S3 paths or credentials: inputs arrive as parameters, the output is the returned dataframe, and everything else comes from typed objects of `dataspace_pylib`.

```python
from dataspace_pylib import build

build.artifacts      # pathlib.Path: local folder for this build's artifacts
build.is_draft       # True for a build of uncommitted code
build.is_local       # True for a build on a developer's machine, False on DataSpace
build.id             # build RID
build.transform_id   # transform RID
build.branch         # branch name
build.storage        # direct S3 access, see below
```

Workspace secrets are environment variables named by the secret's key in the workspace settings. The build has no other environment variables to read: everything else comes from `build`, the inputs and a `Previous` parameter.

`build.storage` is for what the rest of the API does not cover, such as writing very large files straight to S3: `build.storage.uri` is the build's folder (`s3://…/builds/<build id>`), `build.storage.options` the Polars storage options with the build's temporary credentials, and `build.storage.filesystem()` a pyarrow filesystem with the same credentials. The credentials write the build's own folder and read its previous build; an input's folder is read with `input.artifacts.storage_options`. Nothing else is reachable. Files written outside `build.artifacts` and the build's S3 folder are lost when the build ends.

## Draft builds

A build of code that is not committed yet is a draft. Each input is cut to its first `draftReadLimit` rows (1,000,000 unless `_config.json` sets another number; `-1` reads every row). Its output is read only by other draft builds, never by a committed build, and it is never the previous build of a later run. Use a draft to try a change, and a committed build for results that count.

## Input metadata and upstream artifacts

An input is the LazyFrame of the upstream dataset with two attributes. Annotate it as `Input` to get code completion for them.

- `meta`: facts about the upstream dataset: `transform_id`, `build_id`, `snapshot_id`, `row_count`, `column_count`, `size_bytes`, `created_at` (a `datetime`) and `columns` (column name to type).
- `artifacts`: the artifact folder of the upstream build. Nothing is fetched until asked for.

```python
import json

import polars as pl
from dataspace_pylib import Input


def transform(upstream_data: Input) -> pl.LazyFrame:
    artifacts = upstream_data.artifacts
    names = artifacts.list("raw/*.csv")                            # paths relative to the folder; * also matches /
    summary = json.loads(artifacts.get("summary.json").read_text())  # one file, as a local pathlib.Path
    raw = artifacts.get("raw")                                     # a folder; artifacts.get() is the whole artifact folder
    big = pl.scan_csv(artifacts.uri("raw/big.csv"), storage_options=artifacts.storage_options)  # read in place
    return upstream_data
```

`get` returns a real local path, so `open`, `glob` and any library work with it; a path is fetched once per build. Fetch single files or subfolders when the folder is large, and let Polars read large files in place with `uri` and `storage_options`. Do not reconstruct another transform's artifact path.

## Artifacts and dashboards

Write artifacts such as JSON, CSV, or images into `build.artifacts`, a local folder. What is in it when the transform ends is stored with the build, also after the transform raised an error.

Create an interactive dashboard by writing `index.html` into it:

```python
import plotly.express as px
from dataspace_pylib import build


def transform(input_df):
    frame = (
        input_df.select("category", "value")
        .limit(10_000)
        .collect()
        .to_pandas()
    )
    figure = px.bar(frame, x="category", y="value")
    figure.write_html(build.artifacts / "index.html")
    return input_df
```

Keep any local dashboard materialization explicitly bounded and project only the required columns. For high-volume datasets, prefer server-side aggregation or an API-backed dashboard instead of collecting raw rows in the transform process.

If no `index.html` exists, DataSpace serves the artifact folder as a file listing.

For a full browser application, the transform build can compile React, Vite, or another frontend into `build.artifacts`. Keep `index.html` at the artifact root and make bundled asset URLs work from the nested hosted artifact path. Read [interactive-dashboards.md](interactive-dashboards.md) for the build, hosting, and DataSpace API query pattern.

## Health checks

Return health checks alongside the dataframe as `(dataframe, checks)`. Read [health-checks.md](health-checks.md) for supported checks, severity behavior, custom checks, report interpretation, and validation.

## Incremental builds

A parameter of the type `Previous` is the transform's own last successful, committed build on the same branch. The parameter can have any name, and it makes the transform incremental. `exists` is false on the first build; `data`, `artifacts` and `meta` are those of the previous build, shaped like an input's.

Add rows without touching the old ones:

```python
import polars as pl
from dataspace_pylib import Previous


def transform(new_data: pl.LazyFrame, history: Previous):
    return history.append(new_data)
```

`append` keeps the files of the previous dataset as they are and writes the new rows as additional files. The previous rows are neither read nor written again, so the build costs what the new rows cost, however large the dataset is. The new rows must have exactly the columns and types of the previous dataset, otherwise the build fails. On the first build the rows are the dataset. Health checks still run on the whole dataset.

Read and rewrite the dataset when old rows must change, for example to deduplicate or update by key:

```python
def transform(new_data: pl.LazyFrame, history: Previous) -> pl.LazyFrame:
    if not history.exists:
        return new_data
    return pl.concat([history.data, new_data]).unique("id", keep="last")
```

Choose merge and deduplication keys from the domain model; do not assume whole-row uniqueness is correct for every pipeline.

State that is not a dataset goes through artifacts: write it into `build.artifacts` and read the previous one with `history.artifacts.get("state.json")`.

A draft build's previous build is the last committed one, and its own output is never the previous build of a later run.

## Local execution

A clone of the workspace's repository builds on a developer's machine with the same library, from the repository root:

```bash
python -m dataspace_pylib run <transform>
```

`<transform>` is the transform's name (its file name without `.py`), its file, or its RID. The Python environment needs `dataspace-pylib` and the workspace's other packages from `_config.json`.

A local build works like one on DataSpace, only the place differs: the folders are below `.dataspace/` in the repository (ignored by git) instead of in S3.

- **Every run is a build of its own**, `.dataspace/<transform>/builds/0001/`, `0002/`, …, with `datasets/`, `artifacts/` and `meta/` (`columns.json`, `health_report.json`). A build that succeeded holds the file `SUCCESS`. The last three successful builds are kept.
- **An input is the latest successful local build of the upstream transform.** Build the upstream transforms first, in their order.
- **Mock data stands in for a transform that is not built here**: Parquet files in `.dataspace/<transform>/mock/datasets/`, and `mock/artifacts/` when the code reads artifacts. For a transform of another workspace the folder is named by its RID, `.dataspace/<rid>/mock/`. Mock data only counts while the transform has no successful local build.
- **The previous build** of a `Previous` parameter is the transform's own last successful local build, so incremental logic runs locally as on DataSpace: the first run has no previous build, each later one has.
- `build.is_local` is true, `build.is_draft` false, and every row of an input is read.

Set the workspace secrets the code needs as environment variables. A local build cannot read data from DataSpace or write to it.

## Official documentation

- [Documentation index](https://docs.dataspace.ch/llms.txt)
- [Transform function](https://docs.dataspace.ch/api-reference/transform-function.md)
- [Plots and dashboards](https://docs.dataspace.ch/platform/transformation/plot.md)
- [Health checks](https://docs.dataspace.ch/api-reference/health-checks.md)
