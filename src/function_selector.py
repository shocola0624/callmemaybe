from .defines import MAX_TOKENS_FOR_EACH_CALL
from .models import FunctionDef
from .tokenizer import get_token_ids, get_next_token_id, get_token_from_id
from llm_sdk import Small_LLM_Model
from typing import List, Set
import numpy as np


def select_function(
        model: Small_LLM_Model,
        funcs: List[FunctionDef],
        user_prompt: str
) -> str:
    """TODO"""
    func_names = [f.name for f in funcs]
    func_name_token_ids = [get_token_ids(model, f) for f in func_names]
    func_descriptions = [f"- {f.name}: {f.description}" for f in funcs]

    # LLM generates function name enclosed in ""
    func_name = ""
    dq_token_id = get_token_ids(model, "\"")[0]

    # Build prompt
    descriptions = "\n".join(func_descriptions)
    prompt_for_func = (
        "You have access to the following functions:\n"
        f"{descriptions}\n"
        f"User request: {user_prompt}\n\n"
        "Which function should be called?\n"
        "Answer with the function name in double quotes only: \""
    )

    prompt_token_ids = get_token_ids(model, prompt_for_func)
    generated_token_ids: List[int] = []
    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits and mask them
        logits = model.get_logits_from_input_ids(prompt_token_ids)
        masked_logits = mask_func_name_logits(
            model, logits, func_name_token_ids, generated_token_ids
        )

        # Get next token id from logits
        next_token_id = get_next_token_id(masked_logits)
        if next_token_id == dq_token_id:
            break

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
        generated_token_ids.append(next_token_id)
        func_name += get_token_from_id(model, next_token_id)

    if func_name in func_names:
        return func_name
    else:
        raise ValueError(
            "Error: Reached to the max tokens.\n"
            f"Prompt: {user_prompt}"
            f"{func_name}"
        )


def mask_func_name_logits(
        model: Small_LLM_Model,
        logits: List[float],
        func_name_token_ids: List[List[int]],
        generated_token_ids: List[int]
) -> List[float]:
    """TODO"""
    current_len = len(generated_token_ids)
    masked_logits = [-np.inf] * len(logits)
    dq_token_id = get_token_ids(model, "\"")[0]

    # Register valid tokens
    valid_tokens: Set[int] = set()
    for i in func_name_token_ids:
        if generated_token_ids == i[:current_len]:
            if current_len < len(i):
                valid_tokens.add(i[current_len])
            elif current_len == len(i):
                valid_tokens.add(dq_token_id)

    # Mask
    for ids in valid_tokens:
        masked_logits[ids] = logits[ids]

    return masked_logits
