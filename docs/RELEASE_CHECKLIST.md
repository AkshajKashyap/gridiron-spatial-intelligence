# Release Checklist

Target release: `gridiron-spatial-intelligence` `v0.1.0`.

Checked items have direct repository evidence. Pending items must remain open
until their release task is completed and verified.

## Environment and package

- [x] Fresh temporary environment installs with `python -m pip install -e ".[test]"`.
- [x] `gridiron_spatial` imports from the temporary environment.
- [x] Run the complete pytest suite in the final release environment.
- [x] Compile all tracked Python under `src/`, `scripts/`, and `tests/`.
- [x] `pip check` reports no broken requirements in the clean environment.
- [x] Distribution name, version `0.1.0`, Python requirement, and runtime
  dependencies agree with `pyproject.toml`.
- [x] Build and inspect the final wheel/sdist and import from the built package.
- [x] No lint/type checks are configured for `v0.1.0`.

## Data-free release surface

- [x] Verify the README quickstart through the clean GitHub Actions checkout
  and equivalent local data-free commands.
- [x] Packaging and release-evidence focused tests pass.
- [x] Full data-free release-verification command passes locally.
- [x] Data-free CI/release command sequence validates locally.
- [x] Evidence-manifest tests prove deterministic byte-identical output.
- [x] Rebuild the production manifest and preserve all nine source checksums.
- [x] Confirm raw NFL data, Parquet, pair-level rows, predictions, and fitted
  objects are not tracked.
- [x] Confirm only the nine approved compact evidence JSON files and release
  manifest are allowlisted.
- [x] Confirm the evidence manifest contains only relative safe paths.

## Frozen evidence

- [x] Frozen evaluator execution count remains `1`.
- [x] Frozen selections changed remains `0`.
- [x] Frozen comparators changed remains `0`.
- [x] Leakage validation remains `PASS`.
- [x] Frozen reconciliation mismatch count remains `0`.
- [x] Frozen-result checksum matches the release manifest.
- [x] Release instructions never recommend rerunning the frozen evaluator for
  selection.

## Documentation and portfolio review

- [x] README states the bounded research question, evidence, and limitations.
- [x] Architecture, model card, evaluation methodology, reproducibility,
  release checklist, and interview notes exist.
- [x] Documentation links use tracked relative paths or approved release
  artifacts.
- [x] No document claims official coverage, completion probability, causality,
  betting value, or production readiness.
- [x] Aggregate-only data-free demo smoke passes deterministically.
- [x] Post-governance GitHub Actions run `30733959567` passed for commit
  `3c221c9c77f2e27f8704960ffb3222c606573215` on Python 3.11 and Python 3.13.

## Repository and release

- [x] `git diff --check` passes for completed release-hardening tasks.
- [x] MIT license added.
- [x] Changelog added.
- [x] Citation metadata added.
- [x] Contribution guide added.
- [x] `v0.1.0` release notes prepared.
- [x] Package governance metadata validated.
- [x] Final post-governance CI passed.
- [x] The working tree was clean before verification and remained clean after
  deterministic verification and generated-output cleanup.
- [x] Version, release date, author, license, release notes, citation metadata,
  and changelog are mutually consistent.
- [x] Final tracked-artifact policy verification passed.
- [x] Final evidence and checksum verification passed without drift.
- [x] Final source and wheel distribution build passed.
- [x] No unrelated generated package metadata remains modified.
- [x] The `v0.1.0` and `0.1.0` tag names are available locally and remotely.
- [ ] Create the `v0.1.0` tag.
- [ ] Push the `v0.1.0` tag.
- [ ] Create the GitHub release.
- [ ] Verify the release page.
- [ ] Verify post-release URLs.

## Release decision

The final pre-tag gate is complete. Tag creation, pushing the tag, GitHub
release creation, release-page verification, and post-release URL checks
remain pending and must be performed explicitly.
