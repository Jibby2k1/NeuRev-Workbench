# Changelog

All notable user-facing changes to NeuRev Workbench are recorded here. The
project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
intends to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for
tagged software releases.

Scientific evidence has its own immutable experiment, run, capsule, claim, and
decision records. A software version change does not upgrade evidence, and an
experiment result is not retroactively rewritten into a software changelog.

## [Unreleased]

### Added

- Repository-wide, schema-validated research registry with portable evidence
  capsules and synchronized human, machine, and manuscript views.
- Publication-boundary policy and read-only repository audit.
- Native experiment, planning-decision, and planned-run scaffold templates with
  exclusive no-overwrite creation.
- Contributor, conduct, security, citation, issue, pull-request, and release
  process documentation.

### Changed

- Research navigation now distinguishes operational completion, scientific
  outcome, evidence tier, review state, claim state, and decision action.

### Security

- Public contribution and release guidance now explicitly excludes raw inputs,
  generated output trees, reviewer identities, private randomization material,
  credentials, and workstation paths.

[Unreleased]: https://github.com/Jibby2k1/NeuRev-Workbench/commits/HEAD
