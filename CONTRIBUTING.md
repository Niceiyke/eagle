# Contributing to Eagle API Framework

Thank you for your interest in contributing to Eagle! We appreciate your time and effort in making this project better.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes and commit them
5. Push your changes to your fork
6. Open a pull request

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **pytest** for testing

Run these commands before committing:

```bash
black .
isort .
mypy .
pytest
```

## Pull Request Guidelines

1. Keep PRs focused on a single feature or bugfix
2. Write clear, concise commit messages
3. Update the documentation if necessary
4. Make sure all tests pass
5. Ensure your code is properly formatted

## Reporting Issues

When reporting issues, please include:

- A clear description of the issue
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant error messages
- Your environment (Python version, OS, etc.)

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
