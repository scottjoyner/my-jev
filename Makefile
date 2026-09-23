.PHONY: install test lint train-example

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

train-example:
	python -m my_jev.train \
		--train examples/train.jsonl \
		--valid examples/valid.jsonl \
		--output runs/example \
		--epochs 1 \
		--freeze-backbone
