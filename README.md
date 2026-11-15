# dataspace-pylib

The Python side of [DataSpace](https://dataspace.ch): it runs a workspace's transforms and is the API they import.

```python
import polars as pl
from dataspace_pylib import Check, Input, Previous, build


def transform(orders: Input, history: Previous):
    notes = orders.artifacts.get("notes.txt").read_text()   # upstream artifacts, fetched when asked for
    (build.artifacts / "report.html").write_text("…")       # this build's artifacts
    new_rows = orders.filter(pl.col("amount") > 0)
    return history.append(new_rows), [Check("amount").no_nulls()]
```

Build a transform on your machine, in a clone of a workspace:

```bash
python -m dataspace_pylib run <transform>
```

Its data goes below `.dataspace/` in the repository; inputs are the latest local builds of the upstream transforms, or mock data.

## DataSpace Pipelines Skill

This repository includes the [`dataspace-pipelines`](https://github.com/dataspaceswiss/dataspace-pylib/tree/main/skills/dataspace-pipelines) skill for Codex and Claude Code. It describes the whole API.

### Codex

```text
$skill-installer Install dataspace-pipelines from
https://github.com/dataspaceswiss/dataspace-pylib/tree/main/skills/dataspace-pipelines
```

Invoke it with `$dataspace-pipelines`.

### Claude Code

```bash
npx skills add https://github.com/dataspaceswiss/dataspace-pylib/tree/main/skills/dataspace-pipelines \
  --agent claude-code --global
```

Invoke it with `/dataspace-pipelines`.
