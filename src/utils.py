from pathlib import Path
import sys


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def add_src_to_path() -> None:
    src = str(get_project_root()) + "/src"
    if src not in sys.path:
        sys.path.append(src)


# default behavior, may be subject to change
# append the directory with the source script files to path at runtime, so no further "src.<module>" calls
# have to be made. Since it is not saved, it doesn't pollute the PYTHONPATH globally.
# only works for projects which have the following form:
# project
# │   README.md
# │
# ├───src
# │   │   utils.py
# │   │   ...
# │   ...
add_src_to_path()
