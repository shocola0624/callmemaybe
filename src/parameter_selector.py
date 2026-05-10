from .defines import MAX_TOKENS_FOR_EACH_CALL, SYMBOL_MAP
from .models import FunctionDef
from .prompt_builder import build_param_prompt
from .function_selector import select_designated_name
from .tokenizer import get_token_ids, get_next_token_id, get_token_from_id, \
    get_number_token_ids
from llm_sdk import Small_LLM_Model
from typing import Any, List
import numpy as np


# Reverse SYMBOL_MAP
WORD_TO_SYMBOL = {i: k for k, v in SYMBOL_MAP.items() for i in v}


def get_parameter_value(
        model: Small_LLM_Model,
        func: FunctionDef,
        parameter: str,
        user_prompt: str
) -> Any:
    """Predict the value of a parameter of the selected function using the SLM.

    The function constructs a targeted prompt for the SLM to extract the
    specified parameter value. It utilizes constrained decoding for consistent
    output and applies normalization to the result if the parameter type is
    defined as a string.

    Args:
        model: The SLM instance used for value generation.
        func: The selected function definition.
        parameter: The name of the parameter to be determined.
        user_prompt: One of raw prompts from the input.

    Returns:
        The generated and normalized value for the specified parameter.
    """
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
    elif param_type == "number":
        if "." in generated_value:
            return float(generated_value)
        else:
            return int(generated_value)
    elif param_type == "boolean":
        if generated_value == "true":
            return True
        else:
            return False
    return generated_value


def generate_parameter_value(
        model: Small_LLM_Model,
        param_type: str,
        prompt: str,
        eos_token_id: int
) -> str:
    """Generate a parameter value by applying type-based logit masking.

    The function iteratively predicts tokens while applying a mask to
    the logits based on the expected parameter type. It continues
    generating text until a closing quotation mark is detected, the
    EOS token is reached, or the maximum token limit is exceeded.

    Args:
        model: The SLM instance used for logit calculation.
        param_type: Data type of the parameter value.
        prompt: The input text used to guide the generation.
        eos_token_id: A token ID used to terminate generation.

    Returns:
        A string representing the extracted parameter value.
    """
    # boolean
    if param_type == "boolean":
        return select_designated_name(
            model, ["true", "false"], prompt, eos_token_id
        )

    param_value = ""
    prompt_token_ids = get_token_ids(model, prompt)

    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits
        logits = model.get_logits_from_input_ids(prompt_token_ids)

        # Get next token id from logits
        masked_logits = mask_param_val_logits(
            model, logits, param_type, param_value, eos_token_id
        )
        next_token_id = get_next_token_id(masked_logits)

        if next_token_id == eos_token_id:
            return param_value

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
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
        value_type: str,
        param_value: str,
        eos_token_id: int
) -> List[float]:
    """Restrict logit values based on the expected parameter data type.

    The function filters tokens by the required type, allowing full logits for
    strings but restricting numeric types to valid digits and the EOS token.

    Args:
        model: The SLM instance used for token mapping.
        logits: Raw logit values for the entire vocabulary.
        value_type: The required data type (e.g., "string" or "number").
        param_value: The string of tokens generated so far for the value.
        eos_token_id: The token ID used to signal the end of the value.

    Returns:
        A list of masked logits allowing only type-consistent tokens.
    """
    # string type
    if value_type == "string":
        return logits

    masked_logits = [-np.inf] * len(logits)

    # number type
    if value_type == "number":
        number_token_ids = get_number_token_ids(model, param_value)
        for i in number_token_ids:
            masked_logits[i] = logits[i]
        masked_logits[eos_token_id] = logits[eos_token_id]

    return masked_logits


def normalize_symbol_word(
        generated_value: str
) -> str:
    """Map predicted text labels back to their corresponding symbols.

    Args:
        generated_value: The text value generated by the model.

    Returns:
        The mapped symbol if a match exists, otherwise the original string.
    """
    try:
        w: str = WORD_TO_SYMBOL[generated_value]
        return w
    except KeyError:
        return generated_value
