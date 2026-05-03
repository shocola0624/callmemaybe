from typing import List
from .models import Output
import json


def write_output_file(path: str, data: List[Output]) -> None:
    """TODO"""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occurred while writing {path}."
        )
