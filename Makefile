.PHONY: install test test-unit test-functional lint format typecheck build clean coverage

install:
	pip install -e ".[dev]"

test: test-unit

test-unit:
	pytest tests/unit/ -v --timeout=60

test-functional:
	pytest tests/functional/ -v -s --timeout=300

test-integration:
	pytest tests/integration/ -v -s --timeout=600 -m integration

test-performance:
	pytest tests/performance/ -v -s --timeout=600 -m performance

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

typecheck:
	mypy semantic_vector_router/ --ignore-missing-imports

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

coverage:
	pytest tests/unit/ --cov=semantic_vector_router --cov-report=term-missing --cov-report=html
