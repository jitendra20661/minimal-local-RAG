# Tests

This directory contains unit tests for the LAQ RAG system.

## Running Tests

Install test dependencies:
```bash
pip install pytest pytest-cov
```

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_config.py -v
```

## Test Structure

- `test_config.py` - Tests for configuration validation
- `test_pdf_processor.py` - Tests for PDF processing and Pydantic models

## Writing Tests

When adding new features, please add corresponding tests:
1. Create a new test file: `test_<module_name>.py`
2. Use pytest conventions (test classes start with `Test`, test methods start with `test_`)
3. Include docstrings describing what each test does
4. Use fixtures for shared setup code

## Note

Some tests require Ollama to be running. Tests that interact with external services should be marked with `@pytest.mark.integration` and can be skipped during local development.
