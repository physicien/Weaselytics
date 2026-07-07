# Contributing to Weaselytics

Thank you for your interest in contributing!

## Reporting issues

- Use the issue templates for bug reports and feature requests.
- Include a minimal, complete, and verifiable example.
- Include your Python version, OS, and Weaselytics version.

## Pull requests

- Open an issue first to discuss changes before submitting a PR.
- Keep changes focused — one PR per feature/bugfix.
- Follow the existing code style: 80-character line limit, Numpy-style
  docstrings, descriptive variable names.
- Write tests for new functionality and ensure all tests pass:

```
pytest weaselytics/tests/
```

- Run Ruff before submitting:

```
ruff check weaselytics/
```

## Code style

- Maximum line length: 80 characters.
- Docstrings follow the Numpy documentation style.
- Imports: standard library, third-party, local — each group separated
  by a blank line.
- No commented-out code; remove dead code.
- Prefer ``np.asarray(..., dtype=float)`` over implicit list coercion.

## Licensing

By contributing, you agree that your contributions will be licensed
under the project's BSD 3-Clause license.
