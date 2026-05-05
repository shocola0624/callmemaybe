from llm_sdk import Small_LLM_Model
from typing import Dict, List
import functools
import json


def get_next_token_id(
        logits: List[float]
) -> int:
    """TODO"""
    m = max(logits)
    return logits.index(m)


def get_token_ids(
        model: Small_LLM_Model,
        s: str,
        is_cached: bool = False
) -> List[int]:
    """TODO"""
    if is_cached or len(s) <= 1:
        return get_cached_token_ids(model, s)
    return model.encode(s)[0].tolist()


@functools.lru_cache
def get_cached_token_ids(
        model: Small_LLM_Model,
        s: str
) -> List[int]:
    """TODO"""
    return model.encode(s)[0].tolist()


@functools.lru_cache
def get_token_from_id(
        model: Small_LLM_Model,
        token_id: int
) -> str:
    """TODO"""
    return model.decode([token_id])


def get_number_token_ids(
        model: Small_LLM_Model,
        param_value: str
) -> List[int]:
    """TODO"""
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
    """TODO"""
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r") as f:
            vocab: Dict[str, int] = json.load(f)

        return vocab

    except OSError:
        raise ValueError(
            "Error: Failed to read the vocabulary file."
        )
