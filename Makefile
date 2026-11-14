.PHONY: install check demo run build
install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'
check:
	.venv/bin/ruff check lucifer tests
	.venv/bin/pytest -q
demo:
	.venv/bin/lucifer demo
run:
	.venv/bin/lucifer run
build:
	.venv/bin/python -m build
