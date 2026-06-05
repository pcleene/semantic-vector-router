# Contributing to Semantic Vector Router

Thank you for your interest in contributing to SVR. This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/pcleene/semantic-vector-router.git
cd semantic-vector-router
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To work with a specific embedding provider, install the corresponding extra:

```bash
pip install -e ".[dev,voyage]"       # Voyage AI
pip install -e ".[dev,openai]"       # OpenAI
pip install -e ".[dev,cohere]"       # Cohere
pip install -e ".[dev,huggingface]"  # HuggingFace / sentence-transformers
pip install -e ".[dev,all-embedders]" # All providers
```

## Running Tests

### Unit tests (no network required)

```bash
pytest tests/unit/ -v
```

### Functional tests (requires real MongoDB Atlas)

Functional tests run against a live Atlas cluster. Create a `.env` file in the project root:

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-host>/<db>
VOYAGE_API_KEY=pa-...
```

Then run:

```bash
pytest tests/functional/ -v -s
```

### With coverage

```bash
pytest tests/unit/ --cov=semantic_vector_router --cov-report=term-missing
```

## Code Style

### Formatting and linting

| Tool  | Configuration                 |
|-------|-------------------------------|
| black | `--line-length 100`           |
| ruff  | Rules: E, F, I, N, W, UP     |
| mypy  | `--strict`                    |

All settings are defined in `pyproject.toml`. Run the full suite before submitting:

```bash
ruff check .
black --check .
mypy semantic_vector_router/ --ignore-missing-imports
```

### General rules

- **Python 3.9+** syntax. Use `from __future__ import annotations` when you need newer type hint syntax (e.g., `X | Y` unions, `list[int]` lowercase generics).
- **Line length**: 100 characters.
- **All public API** must have type annotations and Google-style docstrings.

Example docstring:

```python
async def search(
    self,
    query: str,
    partitions: list[str] | None = None,
    limit: int = 10,
) -> SearchResult:
    """Execute a vector search across specified partitions.

    Args:
        query: The natural-language search query.
        partitions: Optional partition names to restrict search scope.
        limit: Maximum number of results to return.

    Returns:
        SearchResult with hits, metadata, and timing info.

    Raises:
        SearchError: If the search index is not ready.
    """
```

## Project Conventions

### Logging

All modules use the project's structured logger, **not** the stdlib `logging.getLogger`:

```python
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)
```

### Async PyMongo

Async PyMongo calls require explicit `await`. Several methods return coroutines that must be awaited before calling `.to_list()`:

```python
# Correct
cursor = await collection.aggregate(pipeline)
results = await cursor.to_list(length=None)

# Correct
cursor = await collection.list_search_indexes()
indexes = await cursor.to_list(length=None)

# Also async -- just await directly
doc = await collection.find_one({"_id": doc_id})
count = await collection.count_documents(filter)
values = await collection.distinct("field")

# Closing the client is async too
await client.close()
```

### Test patterns

- **Unit tests** are fully mocked -- no network calls, no database access.
- **Functional tests** use real MongoDB Atlas with structured embeddings (orthogonal category centroids, 32 dims).
- **Mock at the use-site**, not the definition-site. When module B does `from A import foo`, patch `B.foo`, not `A.foo`.
- **CLI modules** use module-level imports (required for `@patch` to work in tests).

### CLI architecture

The CLI is built with Click. Each command group lives in its own module under `semantic_vector_router/cli/`. Shared helpers (`_get_backend()`, `_run_async()`, `handle_config_error`) are in `cli/helpers.py`.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality. Unit tests are required; functional tests are encouraged when the feature touches Atlas.
3. **Ensure all unit tests pass** (`pytest tests/unit/ -v`).
4. **Run the linters** (`ruff check .`, `black --check .`, `mypy`) and fix any issues.
5. **Submit a PR** with a clear title and description explaining the change, its motivation, and how it was tested.

### Commit messages

Use conventional-style prefixes:

- `feat:` -- new feature
- `fix:` -- bug fix
- `docs:` -- documentation only
- `test:` -- adding or updating tests
- `refactor:` -- code restructuring without behavior change
- `chore:` -- maintenance (deps, CI, tooling)

## Reporting Issues

Open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce (if applicable).
- Expected vs. actual behavior.
- Python version, OS, and SVR version (`svr --version` or `pip show semantic-vector-router`).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
