.PHONY: all
all: lint test

.PHONY: lint
lint:
	pre-commit run --all-files
	poetry run mypy

.PHONY: test
test:
	poetry run coverage run -m unittest discover
	poetry run coverage xml
	poetry run coverage html


.PHONY: clean
clean:
	git clean -d -f


.PHONY: setup
setup:
	poetry install --with dev
