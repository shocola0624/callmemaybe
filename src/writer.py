from typing import List
from .models import Output
from pathlib import Path
import json


def write_output_file(file_path: str, data: List[Output]) -> None:
    """Write results to the output JSON file."""
    output_file = Path(file_path)

    # Create directory
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    except OSError as e:
        raise ValueError(
            f"Error: Failed to create output directory '{file_path}': {e}"
        )

    # Create output file and write result
    try:
        with output_file.open("w") as f:
            json.dump([d.model_dump() for d in data], f, indent=4)

    except OSError as e:
        raise ValueError(
            f"Error: Failed to write output file '{file_path}': {e}"
        )
