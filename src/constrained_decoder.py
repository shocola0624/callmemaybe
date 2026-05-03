from .models import Prompt, FunctionDef, Output
from .function_selector import select_function
from .parameter_selector import get_parameter
from llm_sdk import Small_LLM_Model
from typing import Any, Dict, List


def call_llm(
        prompts: List[Prompt],
        funcs: List[FunctionDef]
) -> List[Output]:
    """TODO"""
    model = Small_LLM_Model("Qwen/Qwen3-0.6B")

    output_data = []
    for p in prompts:
        output_data.append(
            generate_function_call(model, funcs, p.prompt)
        )

    return output_data


def generate_function_call(
        model: Small_LLM_Model,
        funcs: List[FunctionDef],
        user_prompt: str
) -> Output:
    """TODO"""
    func_name = select_function(model, funcs, user_prompt)
    func = next(f for f in funcs if f.name == func_name)

    parameters: Dict[str, Any] = dict()
    for p in func.parameters.keys():
        parameter_value = get_parameter(model, user_prompt, func, p)
        parameters[p] = parameter_value

    # return Output(
    #     prompt=user_prompt,
    #     name=func_name,
    #     parameters=parameters
    # )
    return {
        "prompt": user_prompt,
        "name": func_name,
        "parameters": parameters
    }
