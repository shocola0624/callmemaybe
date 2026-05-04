from .defines import MAX_TOKENS_FOR_EACH_CALL, SYMBOL_MAP
from .models import FunctionDef
from .prompt_builder import build_param_prompt
from .tokenizer import get_token_ids, get_next_token_id, get_token_from_id, \
    get_number_token_ids
from llm_sdk import Small_LLM_Model
from typing import List
import numpy as np


def get_parameter_value(
        model: Small_LLM_Model,
        func: FunctionDef,
        parameter: str,
        user_prompt: str
) -> str:
    """TODO"""
    param_type = func.parameters[parameter].type

    # LLM generates parameter value enclosed in ""
    dq_token_id = get_token_ids(model, "\"")[0]

    # Build prompt
    prompt_for_param = build_param_prompt(func, parameter, user_prompt)

    generated_value = generate_parameter_value(
        model, param_type, prompt_for_param, dq_token_id
    )

    if param_type == "string":
        return normalize_symbol_word(generated_value)
    return generated_value


def generate_parameter_value(
        model: Small_LLM_Model,
        param_type: str,
        prompt: str,
        eos_token_id: int | None = None
) -> str:
    """TODO"""
    param_value = ""
    prompt_token_ids = get_token_ids(model, prompt)
    generated_token_ids: List[int] = []

    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits
        logits = model.get_logits_from_input_ids(prompt_token_ids)

        # Get next token id from logits
        masked_logits = mask_param_val_logits(
            model, logits, param_type
        )
        next_token_id = get_next_token_id(masked_logits)

        if eos_token_id is not None and next_token_id == eos_token_id:
            return param_value

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
        generated_token_ids.append(next_token_id)
        param_value += get_token_from_id(model, next_token_id)

        if param_value.count("\"") >= 1:
            parts = param_value.split("\"")
            return parts[0] if parts else ""

    raise ValueError(
        "Error: Reached to the max tokens.\n"
    )


def mask_param_val_logits(
        model: Small_LLM_Model,
        logits: List[float],
        value_type: str
) -> List[float]:
    """TODO"""
    if value_type == "string":
        return logits

    masked_logits = [-np.inf] * len(logits)
    dq_token_id = get_token_ids(model, "\"")[0]

    if value_type == "number":
        number_token_ids = get_number_token_ids(model)
        for i in number_token_ids:
            masked_logits[i] = logits[i]
        masked_logits[dq_token_id] = logits[dq_token_id]

    return masked_logits


def normalize_symbol_word(
        generated_value: str
) -> str:
    """TODO"""
    word_to_symbol = {i: k for k, v in SYMBOL_MAP.items() for i in v}
    try:
        w: str = word_to_symbol[generated_value]
        return w
    except KeyError:
        return generated_value
