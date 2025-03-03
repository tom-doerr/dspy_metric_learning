# Contributing to DSPy Metric Learning

Thank you for considering contributing to DSPy Metric Learning! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Any relevant logs or screenshots
6. Your environment (OS, Python version, etc.)

### Suggesting Enhancements

If you have an idea for an enhancement, please create an issue with:

1. A clear, descriptive title
2. A detailed description of the enhancement
3. Any relevant examples or mock-ups
4. Why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests to ensure they pass
5. Update documentation if necessary
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
3. Run tests to ensure everything is working:
   ```bash
   poetry run pytest
   ```

## Testing

Please ensure all tests pass before submitting a pull request:

```bash
python -m pytest
```

For integration tests:

```bash
python -m pytest integration_tests/
```

## Code Style

Please follow these guidelines:

- Use consistent indentation (4 spaces)
- Follow PEP 8 guidelines
- Write clear, descriptive docstrings
- Include type hints where appropriate

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
