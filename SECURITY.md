# Security Configuration

This project has comprehensive security measures in place to prevent secrets and vulnerabilities from entering the codebase.

## Pre-commit Hooks

The following security checks run automatically before each commit:

1. **Code Formatting** - Black and isort ensure consistent code style
2. **Linting** - Ruff checks for code quality issues
3. **Type Checking** - Mypy ensures type safety
4. **Tests** - Full test suite runs to ensure functionality
5. **Secret Detection** - detect-secrets scans for potential credentials
6. **Vulnerability Scanning** - Trivy scans for security vulnerabilities
7. **General Checks** - File size limits, trailing whitespace, YAML validation

### Setup

To install pre-commit hooks:
```bash
just pre-commit-install
```

### Manual Run

To run all hooks manually:
```bash
just pre-commit-run
```

## Secret Detection

We use detect-secrets to prevent credentials from being committed:

- Baseline file: `.secrets.baseline`
- Update baseline: `just update-secrets-baseline`
- Audit findings: `just audit-secrets`  # pragma: allowlist secret

## Vulnerability Scanning

Trivy scans for:
- Dependency vulnerabilities
- Infrastructure as Code issues
- Exposed secrets
- License compliance

Run manually:
```bash
just security-scan
```

## GitHub Actions

Security scans also run automatically on:
- Every push to main/master/develop
- Every pull request
- Daily at 2 AM UTC

## Important Files to Never Commit

The following files are in `.gitignore` and should NEVER be committed:
- `.env` (except `.env.example`)
- `credentials.json`
- Any `.pem`, `.key`, `.cert` files
- Database files
- Log files

## Security Commands

```bash
just check           # Run tests and linting
just security-check  # Run full security suite
just security-scan   # Run Trivy scan
just pre-commit-run  # Run all pre-commit hooks
```

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly by emailing the maintainers directly rather than opening a public issue.
