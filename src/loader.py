from .models import Prompt, FunctionDef
from typing import List
from pydantic import ValidationError
import json


def get_prompts_from_json(path: str) -> List[Prompt]:
    """Load and validate a list of prompt objects from a JSON file.

    Args:
        path: The file system path to the JSON file.

    Returns:
        A list of validated Prompt objects.

    Raises:
        ValueError: If the file is not a JSON, the file cannot be read,
            or the content fails JSON decoding and model validation.
    """

    if path[-5:] != ".json":
        raise ValueError(
            f"Error: {path} is not json file."
        )

    try:
        with open(path, "r") as f:
            raw_data = json.load(f)

        return [Prompt.model_validate(d) for d in raw_data]

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occurred while reading {path}."
        )

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Error ({path}): Invalid json format ({e})."
        )

    except (AttributeError, ValidationError):
        raise ValueError(
            f"Error ({path}): Invalid json format."
        )


def get_funcs_from_json(path: str) -> List[FunctionDef]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the function definition JSON file.

    Returns:
        A list of validated FunctionDef objects.

    Raises:
        ValueError: If the file is not a JSON, cannot be read, fails
            validation, or contains duplicate names or descriptions.
    """

    if path[-5:] != ".json":
        raise ValueError(
            f"Error: {path} is not json file."
        )

    try:
        with open(path, "r") as f:
            raw_data = json.load(f)

        funcs = [FunctionDef.model_validate(d) for d in raw_data]

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occurred while reading {path}."
        )

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Error ({path}): Invalid definition json format ({e})."
        )

    except ValidationError:
        raise ValueError(
            f"Error ({path}): Invalid definition json format."
        )

    names = []
    descs = []
    for f in funcs:
        if f.name in names:
            raise ValueError(
                "Error: Defined the same name several times."
            )
        names.append(f.name)

        if f.description in descs:
            raise ValueError(
                "Error: Defined the same description several times."
            )
        descs.append(f.description)

    return funcs
