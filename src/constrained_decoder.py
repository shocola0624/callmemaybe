from .models import Prompt, FunctionDef, Output
from typing import List


def call_llm(
        prompts: List[Prompt],
        funcs: List[FunctionDef]
) -> List[Output]:
    """TODO"""
    pass
