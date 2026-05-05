from .models import FunctionDef
from typing import List


def build_func_prompt(
        funcs: List[FunctionDef],
        user_prompt: str
) -> str:
    """TODO"""
    func_descriptions = [f"- {f.name}: {f.description}" for f in funcs]
    descriptions = "\n".join(func_descriptions)
    return (
        "You have access to the following functions:\n"
        f"{descriptions}\n"
        f"User request: {user_prompt}\n\n"
        "Which function should be called?\n"
        "Answer with the function name in double quotes only: \""
    )


def build_param_prompt(
        func: FunctionDef,
        parameter: str,
        user_prompt: str
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
        "- string otherwise: copy the literal text verbatim from the request."
        "\n\n--- Examples ---\n\n"
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
        'Request: "Capture all digits in '
        'code42 as a group and replace with X"\n'
        'Parameter "source_string": "code42"\n'
        'Parameter "regex": "(\\d+)"\n'
        'Parameter "replacement": "X"\n\n'
        "--- Now extract ---\n\n"
        f"Function: {func.name} ({sig})\n"
        f'Request: "{user_prompt}"\n'
        f'Parameter "{parameter}": "'
    )
