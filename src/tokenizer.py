from llm_sdk import Small_LLM_Model
from typing import List
import functools


def get_next_token_id(
        logits: List[float]
) -> int:
    """TODO"""
    m = max(logits)
    return logits.index(m)


@functools.lru_cache
def get_token_ids(
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
