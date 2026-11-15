"""`python -m dataspace_pylib`: runs a build.

build <file>      the build DataSpace describes in a file; this is what a build container runs
run <transform>   a build on this machine, in a clone of a workspace (see local.py)
"""

import argparse
import json
import sys

from pydantic import ValidationError

from .health_check import HealthCheckFailure
from .local import run_local
from .runner import BuildDescription, ConfigurationError, DataSpaceError, run_build


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataspace-pylib", description="Runs a DataSpace transform."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="run the build that a description file from DataSpace describes"
    )
    build.add_argument("description", help="path of the build description (JSON)")
    run = commands.add_parser(
        "run", help="build a transform on this machine, below .dataspace/"
    )
    run.add_argument("transform", help="name, file or RID of the transform")
    arguments = parser.parse_args()

    try:
        if arguments.command == "run":
            run_local(arguments.transform)
            return
        try:
            with open(arguments.description, encoding="utf-8") as f:
                description = BuildDescription(**json.load(f))
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as e:
            # The description holds credentials and this message ends up in the build's log, so a
            # validation error names its fields without their values
            problem = (
                "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in e.errors(include_input=False, include_url=False)
                )
                if isinstance(e, ValidationError)
                else str(e)
            )
            raise ConfigurationError(
                f"The build description {arguments.description} is not one this version of dataspace-pylib "
                f"can read; the workspace may need another version of it. {problem}"
            ) from None
        run_build(description)
    # The message says what is wrong; a traceback into this library would add nothing
    except (DataSpaceError, HealthCheckFailure) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
