from .models import Prompt, FunctionDef
from typing import List
from pydantic import ValidationError
import json


def get_prompts_from_json(path: str) -> List[Prompt]:
    """TODO"""

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
            f"Error ({path}): An error occured while reading {path}."
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
    """TODO"""

    if path[-5:] != ".json":
        raise ValueError(
            f"Error: {path} is not json file."
        )

    try:
        with open(path, "r") as f:
            raw_data = json.load(f)

        return [FunctionDef.model_validate(d) for d in raw_data]

    except OSError:
        raise ValueError(
            f"Error ({path}): An error occured while reading {path}."
        )

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Error ({path}): Invalid definition json format ({e})."
        )

    except ValidationError:
        raise ValueError(
            f"Error ({path}): Invalid definition json format."
        )
