from .defines import MAX_TOKENS_FOR_EACH_CALL
from .models import FunctionDef
from .tokenizer import get_token_ids, get_next_token_id, get_token_from_id, \
    get_number_token_ids
from llm_sdk import Small_LLM_Model
from typing import List
import numpy as np


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
    prompt_for_param = build_param_prompt(func, parameter, user_prompt)

    prompt_token_ids = get_token_ids(model, prompt_for_param)
    generated_token_ids: List[int] = []
    for _ in range(MAX_TOKENS_FOR_EACH_CALL):
        # Calculate logits and mask them
        logits = model.get_logits_from_input_ids(prompt_token_ids)
        masked_logits = mask_param_val_logits(
            model, logits, func.parameters[parameter].type
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


def build_param_prompt(
    func: FunctionDef,
    parameter: str,
    user_prompt: str,
) -> str:
    """TODO"""
    sig = ", ".join(
        f"{n}:{p.type}" for n, p in func.parameters.items()
    )

    return (
        "Extract one parameter value from the user request. "
        "Output the value only, then close with a double quote.\n\n"
        "Rules:\n"
        "- Preserve the EXACT case from the request. Never capitalize.\n"
        "- number: digits only (decimals and minus allowed). "
        "Stop right after the number.\n"
        "- string whose name implies regex/pattern: "
        "output a minimal regex. Do NOT wrap in capture groups '()' "
        "unless the request explicitly says 'capture' or 'group'.\n"
        "- string otherwise: copy the literal text verbatim from the request.\n\n"
        "--- Examples ---\n\n"
        'Function: fn_add_numbers (a:number, b:number)\n'
        'Request: "What is the sum of 2 and 3?"\n'
        'Parameter "a": "2"\n'
        'Parameter "b": "3"\n\n'
        'Function: fn_multiply (x:number, y:number)\n'
        'Request: "Multiply 3.14 by -7"\n'
        'Parameter "x": "3.14"\n'
        'Parameter "y": "-7"\n\n'
        'Function: fn_greet (name:string)\n'
        'Request: "Greet alice"\n'
        'Parameter "name": "alice"\n\n'
        'Function: fn_greet (name:string)\n'
        'Request: "Greet Bob"\n'
        'Parameter "name": "Bob"\n\n'
        'Function: fn_reverse_string (s:string)\n'
        "Request: \"Reverse the string 'hello'\"\n"
        'Parameter "s": "hello"\n\n'
        # --- regex examples: ALL without capture groups ---
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Replace all numbers in \\"abc123\\" with NUMBERS"\n'
        'Parameter "source_string": "abc123"\n'
        'Parameter "regex": "\\d+"\n'
        'Parameter "replacement": "NUMBERS"\n\n'
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Replace all vowels in \'hello\' with asterisks"\n'
        'Parameter "source_string": "hello"\n'
        'Parameter "regex": "[aeiouAEIOU]"\n'
        'Parameter "replacement": "asterisks"\n\n'
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Replace digits in code42 with hashes"\n'
        'Parameter "source_string": "code42"\n'
        'Parameter "regex": "\\d+"\n'
        'Parameter "replacement": "hashes"\n\n'
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Replace all whitespace in \'a b c\' with dashes"\n'
        'Parameter "source_string": "a b c"\n'
        'Parameter "regex": "\\s+"\n'
        'Parameter "replacement": "dashes"\n\n'
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Replace all consonants in \'world\' with X"\n'
        'Parameter "source_string": "world"\n'
        'Parameter "regex": "[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]"\n'
        'Parameter "replacement": "X"\n\n'
        # --- explicit capture group example for the rare case ---
        'Function: fn_substitute (source_string:string, regex:string, '
        'replacement:string)\n'
        'Request: "Capture all digits in code42 as a group and replace with X"\n'
        'Parameter "source_string": "code42"\n'
        'Parameter "regex": "(\\d+)"\n'
        'Parameter "replacement": "X"\n\n'
        "--- Now extract ---\n\n"
        f"Function: {func.name} ({sig})\n"
        f'Request: "{user_prompt}"\n'
        f'Parameter "{parameter}": "'
    )
