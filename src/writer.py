from typing import List
from .models import Output
import json


def write_output_file(path: str, data: List[Output]) -> None:
    """TODO"""
    result = json.dump(data)

    try:
        with open(path, "w") as f:
            f.write(result)

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occured while writing {path}."
        )
