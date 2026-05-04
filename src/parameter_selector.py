from .defines import MAX_TOKENS_FOR_EACH_CALL
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
    # LLM generates parameter value enclosed in ""
    param_value = ""
    param_type = func.parameters[parameter].type
    dq_token_id = get_token_ids(model, "\"")[0]

    # Category for string parameters
    # if param_type == "string":
    #     category = get_category_for_str(model, func, parameter, user_prompt)
    # else:
    #     category = ""

    # Build prompt
    prompt_for_param = build_param_prompt(func, parameter, user_prompt)

    prompt_token_ids = get_token_ids(model, prompt_for_param)
    generated_token_ids: List[int] = []
    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits and mask them
        logits = model.get_logits_from_input_ids(prompt_token_ids)
        masked_logits = mask_param_val_logits(
            model, logits, param_type
        )

        # Get next token id from logits
        next_token_id = get_next_token_id(masked_logits)
        if next_token_id == dq_token_id:
            return param_value

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
        generated_token_ids.append(next_token_id)
        param_value += get_token_from_id(model, next_token_id)

        if param_value.count("\"") >= 1:
            return param_value.split("\"")[0]

    raise ValueError(
        "Error: Reached to the max tokens.\n"
        f"Prompt: {user_prompt}\n"
        f"Parameter: {parameter}"
        f"{param_value}"
    )


def get_category_for_str(
        model: Small_LLM_Model,
        func: FunctionDef,
        parameter: str,
        user_prompt: str
) -> str:
    """TODO"""
    categories = [
        get_token_ids(model, c, True)
        for c in ("symbol", "literal", "regex")
    ]
    pass


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
