.PHONY: all
all: lint test

.PHONY: lint
lint:
	pre-commit run --all-files
	poetry run mypy

.PHONY: test
test:
	poetry run python -m unittest discover


.PHONY: clean
clean:
	git clean -d -f
