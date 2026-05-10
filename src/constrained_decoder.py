from .models import Prompt, FunctionDef, Output
from .function_selector import select_function
from .parameter_selector import get_parameter_value
from .defines import RESET, RED, ORANGE, YELLOW, GREEN
from llm_sdk import Small_LLM_Model
from typing import Any, Dict, List


def call_llm(
        prompts: List[Prompt],
        funcs: List[FunctionDef],
        model_name: str = "Qwen/Qwen3-0.6B"
) -> List[Output]:
    """Execute function calling by predicting names and parameters using SLM.

    Process:
        1. Initialize an instance of the Small_LLM_Model class.
        2. Process each prompt to identify the appropriate function to call.
        3. Returns output.

    Args:
        prompts: A list of prompts from the input.
        funcs: A list of function definitions.
        model_name: The name of the language model to be used.
            Defaults to "Qwen/Qwen3-0.6B".

    Returns:
        A list containing the generated outputs for each prompt.
    """
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
    """Generate function call by sequentially predicting names and parameters.

    Process:
        1. Identify the most suitable function name from the provided list.
        2. Iterate through each parameter of the selected function.
        3. Use the SLM to determine the specific value for each parameter.
        4. Construct and return an Output object.

    Args:
        model: The SML instance used for inference.
        funcs: A list of available function definitions.
        user_prompt: One of raw prompts from the input.

    Returns:
        An Output object containing the function name and parameters.
    """
    # Print user prompt
    print(f"Prompt: {GREEN}{user_prompt}{RESET}")

    # Select the function
    func_name = select_function(model, funcs, user_prompt)
    func = next(f for f in funcs if f.name == func_name)
    print(f"Function: {RED}{func_name}{RESET}")

    # Select each parameter
    parameters: Dict[str, Any] = dict()
    for p in func.parameters.keys():
        parameter_value = get_parameter_value(model, func, p, user_prompt)
        parameters[p] = parameter_value
        print(f"Parameter '{p}': {YELLOW}{parameter_value}{RESET}")

    return Output(
        prompt=user_prompt,
        name=func_name,
        parameters=parameters
    )
