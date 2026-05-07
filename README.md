*This project has been created as part of the 42 curriculum by skotera.*

# Call Me Maybe



## Description
**call me maybe** is a function-calling system that translates natural language prompts into structured JSON function calls using a small language model (Qwen3-0.6B).

Given a natural language request such as `"What is the sum of 2 and 3?"`, the system does not answer the question directly.
Instead, it identifies the appropriate function to call and extracts the required arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The core technique is **constrained decoding**: rather than prompting the model and hoping for structured output, the system intervenes at the logit level on every generation step, masking invalid tokens to `-inf` before sampling.
This enforces both the selection of a valid function name and the generation of type-correct parameter values, without relying on any external grammar-enforcement library.


## Implementation
JSON structure is assembled deterministically by the program — the model never emits braces, commas, or quotes.
The model's only job is to decide **which function to call** and **what values to assign to each parameter**. Constrained decoding ensures those decisions are drawn from the valid candidate set.

### Algorithm Explanation

#### Overview of Constrained Decoding


#### Function Calling

##### 1. Function Selection

##### 2. Parameter Value Selection


### Design Decisions
Brief is Betterより、0.6Bモデルで推論は難しいと判断。

### Performance Analysis

### Challenges Faced

### Testing Strategy

### Example Usage

### Bonus
- Performance optimizations (caching, batching)
- Visualization of the generation process



## Instruction

### Requirements
- Python 3.10+
- uv

### Makefile

#### Installation
```bash
make install
```
Install project dependencies using `uv`.

#### Execution
```bash
make run

make run-flag  # Run the main script with flags to specify file paths

make debug  # Run the main script in debug mode using pdb
```
Execute the main script of the project.
By default, the program will read input files from the data/input/ directory and write output to the data/output/ directory.
You can optionally specify custom paths using the `--input` and `--output` arguments.

**Flags:**
- `--functions_definition`
- `--input`
- `--output`

#### Other Commands
```bash
make clean

make fclean
```
Remove temporary files or caches.
`make fclean` also removes venv files in addition to the files removed by `make clean`.

```bash
make lint

make lint-strict
```
Execute flake8 and mypy command for static type checking.

## Resources

### Documentation
- [LLM Structured Output and Constrained Decoding Practical Guide: From JSON Schema to Production Application](https://www.youngju.dev/blog/llm/2026-03-07-llm-structured-output-constrained-decoding-json-schema.en)

**mypy**
- [The mypy configuration file](https://mypy.readthedocs.io/en/stable/config_file.html)

**Python documentation**
- [json — JSON encoder and decoder](https://docs.python.org/3.14/library/json.html)
- [argparse — Parser for command-line options, arguments and subcommands](https://docs.python.org/3.14/library/argparse.html)
- [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)

**Wikipedia**
- [Automaton theory](https://en.wikipedia.org/wiki/Automata_theory)

**Qwen**
- [Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)

### Academic papers
- [Brief Is Better: Non-Monotonic Chain-of-Thought Budget Effects in Function-Calling Language Agents](https://arxiv.org/abs/2604.02155)

### AI Usage
- Documentation assistant
- Prompt
- Collection and summarisation of academic papers