from typing import List
from .models import Output


def write_output_file(path: str, data: List[Output]) -> None:
    """TODO"""
    result = "[\n"

    try:
        with open(path, "w") as f:
            f.write(result)

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occured while writing {path}."
        )
