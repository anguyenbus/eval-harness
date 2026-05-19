.PHONY: install lint format test test-cov eval-parsing eval-rag clean

# Default target
all: install lint test

# Install dependencies using uv
install:
	uv sync

# Run linting (ruff)
lint:
	uv run ruff check .
	uv run black --check .

# Format code with black and ruff
format:
	uv run ruff format .
	uv run black .

# Run type checking with mypy
typecheck:
	uv run mypy src

# Run tests
test:
	uv run pytest -q

# Run tests with coverage report
test-cov:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# Run parsing evaluation (stub parser by default)
eval-parsing:
	uv run eval-parsing --dataset omnidocbench --parser stub

# Run RAG evaluation (stub RAG by default)
eval-rag:
	uv run eval-rag --dataset legalbench_rag --slice mini --rag stub

# Clean generated files
clean:
	rm -rf results/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run pre-commit checks (CI simulation)
ci: install lint typecheck test
