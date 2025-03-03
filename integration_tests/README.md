# Integration Tests

This directory contains integration tests for the DSPy Metric Learning package. These tests verify that different components of the system work together correctly.

## Test Categories

Tests are categorized using pytest markers:

- `integration`: All tests in this directory have this marker
- `slow`: Tests that take longer to run (e.g., full workflow tests)

## Running Tests

### Run all integration tests

```bash
python -m pytest integration_tests/ -v
```

### Run only slow integration tests

```bash
python -m pytest integration_tests/ -v -m "slow"
```

### Run all tests except slow ones

```bash
python -m pytest -v -m "not slow"
```

### Run only integration tests (across all directories)

```bash
python -m pytest -v -m "integration"
```

## Adding New Integration Tests

When adding new integration tests:

1. Use the `@pytest.mark.integration` decorator on test classes
2. Add the `@pytest.mark.slow` decorator to tests that take a long time to run
3. Use the provided test data directory structure for file operations
4. Clean up after your tests in the `tearDown` method

## Test Data Management

Integration tests use a local `test_data` directory to avoid interfering with user data. This directory is automatically created and cleaned up during tests.

## Mocking

The `MockLM` class provides a simple language model mock for testing. Extend this or create additional mocks as needed for your tests.
