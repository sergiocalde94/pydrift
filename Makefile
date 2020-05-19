.PHONY: install

install:
	poetry install

.PHONY: test

test:
	poetry run pytest --pyargs pydrift

.PHONY: check_style

check_style:
	poetry run flake8 --exclude=__init__.py
	poetry run flake8 --ignore F401 pydrift/__init__.py
