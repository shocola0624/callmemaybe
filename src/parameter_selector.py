from .defines import MAX_TOKENS_FOR_EACH_CALL
from .models import FunctionDef
from .tokenizer import get_token_ids, get_next_token_id, get_token_from_id
from llm_sdk import Small_LLM_Model
from typing import List


def get_parameter(
        model: Small_LLM_Model,
        user_prompt: str,
        func: FunctionDef,
        parameter: str
) -> str:
    """TODO"""
    # LLM generates parameter value enclosed in ""
    param_value = ""
    dq_token_id = get_token_ids(model, "\"")[0]

    # Build prompt
    parameters = "\n".join(
        f" - {name} ({pdef.type})"
        for name, pdef in func.parameters.items()
    )
    prompt_for_param = (
        f"You are a parameter extraction assistant.\n\n"
        f"Function: {func.name}\n"
        f"Description: {func.description}\n\n"
        f"Parameters:\n"
        f"{parameters}"
        f"\n\nUser request: {user_prompt}\n\n"
        f"Extract the value of '{parameter}' from the user request.\n"
        "Comprehend the semantic meaning of terms within the values and apply"
        " appropriate transformations where necessary (e.g., percent to %).\n"
        "Answer with the value in double quotes only: \""
    )

    prompt_token_ids = model.encode(prompt_for_param)[0].tolist()
    generated_token_ids: List[int] = []
    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits and mask them
        logits = model.get_logits_from_input_ids(prompt_token_ids)
        masked_logits = mask_param_val_logits(
            model, logits, generated_token_ids
        )

        # Get next token id from logits
        next_token_id = get_next_token_id(masked_logits)
        if next_token_id == dq_token_id:
            return param_value

        # Add next token id and letter
        prompt_token_ids.append(next_token_id)
        generated_token_ids.append(next_token_id)
        param_value += get_token_from_id(model, next_token_id)

        if param_value.count("\"") >= 0:
            return param_value.split("\"")[0]

    raise ValueError(
        "Error: Reached to the max tokens.\n"
        f"Prompt: {user_prompt}\n"
        f"Parameter: {parameter}"
        f"{param_value}"
    )


def mask_param_val_logits(
        model: Small_LLM_Model,
        logits: List[float],
        generated_token_ids: List[int]
) -> List[float]:
    """TODO"""
    return logits
