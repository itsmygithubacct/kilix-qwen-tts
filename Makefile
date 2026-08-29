PYTHON ?= python3
PYTHONPATH := src
export PYTHONPATH

.PHONY: check
check:
	$(PYTHON) tools/check_interface.py
	$(PYTHON) tools/check_surface.py
	$(PYTHON) -m unittest discover -s tests -v
