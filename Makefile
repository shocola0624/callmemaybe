VENV = .venv

SRC = src

DEF_PATH = data/input/functions_definition.json
INPUT_PATH = data/input/function_calling_tests.json
OUTPUT_PATH = data/output/function_calling_results.json

FLAGS = --functions_definition $(DEF_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

install:
	uv sync

run:
	uv run python -m $(SRC)

run-flag:
	uv run python -m $(SRC) $(FLAGS)

debug:
	uv run python -m pdb -m $(SRC) $(FLAGS)

clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' \
		-exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache

fclean: clean
	rm -rf $(VENV)

lint: $(VENV)
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(VENV)
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run run-flag debug clean fclean lint lint-strict