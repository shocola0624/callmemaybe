VENV = .venv
PIP = $(VENV)/bin/pip
UV = $(VENV)/bin/uv

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install uv
	$(UV) sync

.PHONY: install