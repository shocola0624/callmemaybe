from llm_sdk import Small_LLM_Model
from typing import Dict, List
import functools
import json


def get_next_token_id(
        logits: List[float]
) -> int:
    """Return the index of the maximum value from the provided logit list."""
    m = max(logits)
    return logits.index(m)


def get_token_ids(
        model: Small_LLM_Model,
        s: str,
        is_cached: bool = False
) -> List[int]:
    """Convert a string into a list of token IDs using encoding or a cache."""
    if is_cached or len(s) <= 1:
        return get_cached_token_ids(model, s)
    return model.encode(s)[0].tolist()


@functools.lru_cache
def get_cached_token_ids(
        model: Small_LLM_Model,
        s: str
) -> List[int]:
    """Retrieve and cache the token ID list for a specific string."""
    return model.encode(s)[0].tolist()


@functools.lru_cache
def get_token_from_id(
        model: Small_LLM_Model,
        token_id: int
) -> str:
    """Decode a single token ID into its corresponding string."""
    return model.decode([token_id])


def get_number_token_ids(
        model: Small_LLM_Model,
        param_value: str
) -> List[int]:
    """Retrieve token IDs for numeric characters based on the current input.

    It filters the vocabulary for digits, signs, or decimal points that
    can validly extend the current numeric string.
    """
    vocab = get_vocab_file(model)

    if len(param_value) == 0:
        return [
            v for k, v in vocab.items() if k.isdigit() or
            (k[0] == "-" and (len(k) == 1 or k[1:].isdigit()))
        ]

    if param_value.count(".") <= 0:
        return [
            v for k, v in vocab.items() if k.isdigit() or k == "."
        ]

    return [v for k, v in vocab.items() if k.isdigit()]


@functools.lru_cache
def get_vocab_file(
        model: Small_LLM_Model
) -> Dict[str, int]:
    """Load and cache the model's vocabulary mapping from a JSON file.

    Raises:
        ValueError: If the vocabulary file cannot be accessed or read.
    """
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r") as f:
            vocab: Dict[str, int] = json.load(f)

        return vocab

    except OSError:
        raise ValueError(
            "Error: Failed to read the vocabulary file."
        )
