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
    """Identify the target function name from the prompt using the SLM.

    Process:
        1. Build a prompt to guide the SLM toward selecting a function.
        2. Use specific token IDs to constrain the model's output.
        3. Extract the predicted function name from the designated options.

    Args:
        model: The SLM instance used for function selection.
        funcs: A list of available function definitions.
        user_prompt: One of raw prompts from the input.

    Returns:
        The name of the function selected by the model.
    """
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
    """Select one option from provided list using constrained decoding.

    The function converts valid names and the prompt into token IDs and
    iteratively calculates logits to predict the next token. It applies a mask
    to the logits to ensure only tokens forming valid names are generated,
    continuing until a complete name is formed or an EOS token is reached.

    Args:
        name_list: A list of valid names that the model can select.
        prompt: The input text used to guide the generation.
        eos_token_id: An optional token ID used to terminate generation.

    Returns:
        A string containing one of the names from the provided list.
    """
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
    """Apply a mask to logits to restrict token generation to valid names.

    The function identifies valid next tokens by comparing the currently
    generated sequence with the prefixes of all allowed names. It then
    initializes a mask with negative infinity and preserves only the
    logits corresponding to these valid tokens or a closing quotation
    mark once a name is complete, ensuring the model's output remains
    within the predefined set.

    Args:
        logits: The raw logit values for the entire vocabulary.
        names_token_ids: A list of tokenized valid names.
        generated_token_ids: The sequence of tokens generated in current call.

    Returns:
        A list of masked logits where only valid next tokens remain finite.
    """
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
