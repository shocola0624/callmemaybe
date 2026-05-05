import sys
import argparse
from .defines import DEF_PATH, INPUT_PATH, OUTPUT_PATH, \
    RESET, RED
from .loader import get_prompts_from_json, get_funcs_from_json
from .constrained_decoder import call_llm
from .writer import write_output_file


def call_me_maybe() -> None:
    """Entry point for the function calling tool.

    Parses command-line arguments, reads the function definitions and test
    prompts from JSON files, runs each prompt through the LLM with
    constrained decoding to produce a function call, and writes the
    structured results to the output file.

    The pipeline consists of four stages:

    1. Parse CLI arguments (--functions_definition, --input, --output),
       falling back to default paths in data/input/ and data/output/.
    2. Load and validate the input JSON files. Both files are required to
       be syntactically valid JSON that conforms to the expected schema.
    3. For each prompt, invoke the LLM to select a function and extract
       its parameter values via constrained decoding.
    4. Write the resulting list of function calls to the output JSON file,
       creating the parent directory if it does not exist.

    Errors are reported to stderr with a clear message rather than crashing
    with a stack trace. ValueError covers both user-facing input issues
    (missing or malformed files, invalid output path) and internal
    validation failures.

    Raises:
        SystemExit: Indirectly via argparse when CLI arguments are
            malformed (e.g. unknown flags). Caught ValueErrors do not
            propagate; they exit with a printed error message.
    """

    # 1. Argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", type=str, default=DEF_PATH)
    parser.add_argument("--input", type=str, default=INPUT_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)

    args = parser.parse_args()

    def_path = args.functions_definition
    input_path = args.input
    output_path = args.output

    try:
        if not output_path or output_path[-1] == "/":
            raise ValueError(
                f"Error: Invalid output file path ({output_path})."
            )

        # 2. Read and validate JSON
        prompts = get_prompts_from_json(input_path)
        funcs = get_funcs_from_json(def_path)

        # 3. LLM
        output_data = call_llm(prompts, funcs)

        # 4. Write the result into the output file
        write_output_file(output_path, output_data)

    except ValueError as e:
        print(f"{RED}=====  ERROR  ====={RESET}")
        print(e, file=sys.stderr)
