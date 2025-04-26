.PHONY: build
build: deps-prod

.PHONY: build-dev
build-dev: deps-dev

.PHONY: deps-prod
deps-prod:
	poetry install --only main

.PHONY: deps-dev
deps-dev:
	poetry install --with dev

.PHONY: start
start:
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: test
test:
	pytest tests/

.PHONY: lint
lint:
	ruff check .

.PHONY: format
format:
	ruff format .

.PHONY: lint-fix
lint-fix:
	ruff check --fix .
	ruff format .

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 