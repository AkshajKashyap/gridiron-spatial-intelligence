# Contributing

## Project scope

Contributions should support reproducible, target-centric analysis of how
origin geometry relates to future target–defender separation dynamics. They
must preserve the project's documented data and claim boundaries.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run the complete data-free test suite and release verifier:

```bash
python -m pytest -q
python scripts/verify_release.py
```

## Coding and deterministic-output expectations

- Keep changes focused, typed where practical, and covered by synthetic,
  data-free tests.
- Preserve frozen schemas, chronological splits, leakage controls, and stable
  ordering unless a reviewed contract change explicitly replaces them.
- Given unchanged inputs, tracked aggregate outputs and reports must be
  byte-identical.
- Checksum changes require explicit review and an explanation tied to the
  affected source evidence.

## Data and artifact policy

- Do not commit NFL source data.
- Do not commit ignored Parquet files, row-level artifacts, predictions, or
  fitted model objects.
- Keep public evidence compact, aggregate-only, path-safe, and covered by the
  tracked-artifact policy.

## Pull requests

Pull requests should explain the bounded objective, tests run, affected
contracts or checksums, and any change to interpretation. New claims must be
supported by tracked aggregate evidence. Links and data-free verification
commands must work from a clean checkout.

## Frozen-test safeguard

Do not rerun the frozen evaluation for model selection. The frozen test was
executed once; later work must not use it to tune features, models,
comparators, or thresholds.

## Claim boundary

Contributions must not present nearest defenders as official coverage
assignments or claim completion probability, causal effects, complete
passing-window openness, full-field defensive control, betting value, or
production readiness without new data and separately reviewed evidence.
