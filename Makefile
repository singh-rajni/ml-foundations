.PHONY: setup test run notebooks

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python scripts/run_reference_workflow.py

notebooks:
	bash scripts/execute_notebooks.sh
