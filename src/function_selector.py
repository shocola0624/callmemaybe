from .defines import MAX_TOKENS_FOR_EACH_CALL
from .models import FunctionDef
from .prompt_builder import build_func_prompt
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
    # Function list
    func_names = [f.name for f in funcs]

    # Build prompt
    prompt_for_func = build_func_prompt(funcs, user_prompt)

    # LLM generates function name enclosed in ""
    dq_token_id = get_token_ids(model, "\"")[0]

    return select_designated_name(
        model, func_names, prompt_for_func, dq_token_id
    )


def select_designated_name(
        model: Small_LLM_Model,
        name_list: List[str],
        prompt: str,
        eos_token_id: int | None = None
) -> str:
    """TODO prompt to generate only a selected name"""
    selected_name = ""
    names_token_ids = [get_token_ids(model, n) for n in name_list]
    prompt_token_ids = get_token_ids(model, prompt)
    generated_token_ids: List[int] = []

    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits
        logits = model.get_logits_from_input_ids(prompt_token_ids)

        # Get next token id from logits
        masked_logits = mask_logits_by_names(
            model, logits, names_token_ids, generated_token_ids
        )
        next_token_id = get_next_token_id(masked_logits)

        # EOS token
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
        generated_token_ids.append(next_token_id)
        selected_name += get_token_from_id(model, next_token_id)

    if selected_name in name_list:
        return selected_name
    else:
        raise ValueError(
            "Error: Reached to the max tokens while "
            f"selecting a name from {name_list}."
        )


def mask_logits_by_names(
        model: Small_LLM_Model,
        logits: List[float],
        names_token_ids: List[List[int]],
        generated_token_ids: List[int]
) -> List[float]:
    """TODO"""
    current_len = len(generated_token_ids)
    masked_logits = [-np.inf] * len(logits)
    dq_token_id = get_token_ids(model, "\"")[0]

    # Register valid tokens
    valid_tokens: Set[int] = set()
    for i in names_token_ids:
        if generated_token_ids == i[:current_len]:
            if current_len < len(i):
                valid_tokens.add(i[current_len])
            elif current_len == len(i):
                valid_tokens.add(dq_token_id)

    # Mask logits
    for ids in valid_tokens:
        masked_logits[ids] = logits[ids]

    return masked_logits
