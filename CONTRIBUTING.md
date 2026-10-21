# Contributing to OpenMythos

## Setup

```bash
git clone https://github.com/openmythos/openmythos
cd openmythos
pip install -e ".[all,dev]"
pre-commit install
```

## Workflow

1. Fork the repo and create a feature branch from `main`.
2. Write tests BEFORE code (see `tests/`).
3. Run `make check` and `make test` locally — both must pass.
4. Open a PR with a clear description and link to any related issues.

## Code Standards

- Python 3.11+, type hints required on all public functions
- Black + Ruff formatting (enforced by pre-commit)
- All new features must have unit tests with ≥80% coverage
- No hardcoded credentials, secrets, or API keys (pre-commit will block)
- No new TODOs or stubs in `main`, `server`, `trainer`, or `benchmarks`

## Security Reporting

Report security issues privately to security@openmythos.ai — do not open a public issue.
