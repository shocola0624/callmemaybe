import sys
import argparse
from .defines import DEF_PATH, INPUT_PATH, OUTPUT_PATH
from .loader import get_prompts_from_json, get_funcs_from_json
from .constrained_decoder import call_llm
from .writer import write_output_file


def main() -> None:
    """TODO"""
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
        # 2. Read and validate JSON
        prompts = get_prompts_from_json(input_path)
        funcs = get_funcs_from_json(def_path)

        # 3. LLM
        output_data = call_llm(prompts, funcs)

        # 4. Write the result into the output file
        write_output_file(output_path, output_data)

    except ValueError as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
