from .models import Prompt, FunctionDef, Output
from .function_selector import select_function
from .parameter_selector import get_parameter
from .defines import RESET, RED, ORANGE, YELLOW, GREEN
from llm_sdk import Small_LLM_Model
from typing import Dict, List


def call_llm(
        prompts: List[Prompt],
        funcs: List[FunctionDef],
        model_name: str = "Qwen/Qwen3-0.6B"
) -> List[Output]:
    """TODO"""
    # Instance LLM class
    print(f"LLM Model: {ORANGE}{model_name}{RESET}")
    model = Small_LLM_Model(model_name)

    # Execute each prompt
    output_data = []
    for i, p in enumerate(prompts):
        print(f"\n======== Test {i+1} ========")
        output_data.append(
            generate_function_call(model, funcs, p.prompt)
        )

    print(f"\n{GREEN}All tests have completed successfully.{RESET}")

    return output_data


def generate_function_call(
        model: Small_LLM_Model,
        funcs: List[FunctionDef],
        user_prompt: str
) -> Output:
    """TODO"""
    # Print user prompt
    print(f"Prompt: {GREEN}{user_prompt}{RESET}")

    # Select the function
    func_name = select_function(model, funcs, user_prompt)
    func = next(f for f in funcs if f.name == func_name)
    print(f"Function: {RED}{func_name}{RESET}")

    # Select each parameter
    parameters: Dict[str, str] = dict()
    for p in func.parameters.keys():
        parameter_value = get_parameter(model, user_prompt, func, p)
        parameters[p] = parameter_value
        print(f"Parameter '{p}': {YELLOW}{parameter_value}{RESET}")

    return {
        "prompt": user_prompt,
        "name": func_name,
        "parameters": parameters
    }
