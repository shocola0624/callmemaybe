PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip
UV = $(VENV)/bin/uv

PDB = pdb
SRC = src

DEF_PATH = data/input/functions_definition.json
INPUT_PATH = data/input/function_calling_tests.json
OUTPUT_PATH = data/output/function_calling_results.json

FLAGS = --functions_definition $(DEF_PATH) \
--input $(INPUT_PATH) \
--output $(OUTPUT_PATH)

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install uv
	$(UV) sync

run: $(VENV)
	$(UV) run python -m $(SRC) $(FLAGS)

debug: $(VENV)
	$(UV) run python -m $(PDB) -m $(SRC) $(FLAGS)

clean:

lint: $(VENV)
	$(UV) run python -m flake8 .; \
	$(UV) run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(VENV)
	$(UV) run python -m flake8 .; \
	$(UV) run python -m mypy . --strict

.PHONY: install run debug clean lint lint-strict