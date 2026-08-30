# Release process

NeuRev releases software, documentation, public evidence metadata, and large
artifact packages through separate, explicit gates. A tagged software release
does not imply that every experiment passed, that a claim is supported, or that
private data may be published.

## Release surfaces

| Surface | Versioned by | Stored in Git | Required authority |
| --- | --- | --- | --- |
| Python software and documentation | Semantic version and Git tag | Yes | Reviewed source and tests |
| Research story views | Source fingerprint | Generated files only | Canonical `research/registry/` records |
| Public evidence capsule | Stable capsule ID and checksum | Yes, when sanitized | Validated run and bounded claim effects |
| Large figures, tables, media, or reproducibility bundle | Archive version and SHA-256 manifest | No | GitHub Release, Zenodo, or another durable store |
| Raw input or private review package | Data-governance identifier | Never | Authorized restricted storage |

`visibility` metadata is not access control. Anything committed to the public
repository or attached to a public release is public.

## Version policy

Use `MAJOR.MINOR.PATCH` for software:

- **MAJOR**: an incompatible public API, record contract, CLI, or artifact
  contract change;
- **MINOR**: backward-compatible functionality, experiment tooling, or a major
  new documented workflow;
- **PATCH**: backward-compatible fixes and documentation corrections.

Evidence retains stable experiment and capsule IDs across software releases.
Never mutate an already published archive or favorable result in place. Publish
a corrected successor, mark the previous capsule `superseded` or `withdrawn`
when warranted, and record the decision and reason.

## 1. Define the release candidate

1. Choose the exact commit, semantic version, and release scope.
2. Move relevant `CHANGELOG.md` entries from **Unreleased** into a dated version
   section; preserve negative and breaking changes.
3. Update `pyproject.toml` and `CITATION.cff` to the same version. Add
   `date-released` to the citation file only when the date is final.
4. Confirm the root license file exists and matches the license declared in
   `pyproject.toml` and `CITATION.cff`.
5. Identify every evidence capsule or external archive included. Record what is
   deliberately excluded.

## 2. Validate a clean checkout

Release validation must not rely on ignored local inputs, outputs, symlinks, or
the maintainer's environment. Create a fresh clone or worktree at the candidate
commit and run:

```bash
python3 -m venv .venv-neurobench
.venv-neurobench/bin/python -m pip install --upgrade pip
.venv-neurobench/bin/python -m pip install -e '.[dev]'
.venv-neurobench/bin/python -m pytest -q
.venv-neurobench/bin/python -m neurobench.research.registry build
.venv-neurobench/bin/python -m neurobench.research.registry check
make -C paper/overleaf_jnm story-check
```

Run focused smoke tests for documented CLIs and import the package from outside
the repository root. A clean-clone check establishes portability of maintained
source and committed metadata; it does not reproduce large experiments unless a
separate archive and protocol promise that capability.

## 3. Reconcile research records

For every capsule in scope:

- validate all experiment, run, decision, claim, and capsule records against
  their schemas and cross-record references;
- verify code commit, dirty-state declaration, configuration and input hashes,
  seeds, timestamps, output manifest, metric population, uncertainty, gates,
  review state, and interpretation boundaries;
- confirm required scientific-audit inventory and media validation passed, or
  retain the exact authorized opt-out and its consequence;
- verify candidate yield is not described as precision, sparse positives are not
  treated as exhaustive negatives, and source identity or transfer is not
  inferred beyond evidence;
- regenerate every synchronized research and manuscript view and confirm the
  stale-output check passes.

An inconclusive, negative, stopped, failed, or invalidated experiment may be
released when accurately recorded. Publication is not promotion.

## 4. Audit the public boundary

Inspect the candidate from the repository root:

```bash
.venv-neurobench/bin/python tools/audit_publication_boundary.py \
  --root . --fail-on warning
git status --short --branch
git diff --check
```

Then manually inspect the tracked file list, staged diff, rendered Markdown,
diagrams, manuscript, citation metadata, local links, archive licenses, and
attributions. Confirm there are no raw recordings, `Inputs/`, `Outputs/`, local
environment directories, credentials, absolute workstation paths, private
review responses, reviewer identities, randomization keys, identity maps, or
non-portable symlinks.

Automated scanning does not replace repository-host secret scanning or history
review. If restricted content entered Git history, stop the release and follow
`SECURITY.md`; do not merely add it to `.gitignore`.

## 5. Build external artifacts

Build large assets in a new ignored `Outputs/` directory. Create a deterministic
manifest containing relative path, media type, byte size, and lowercase SHA-256
for every file. Validate archive extraction into a temporary directory, media
decode where applicable, and agreement between archive, manifest, evidence
capsule, and release notes.

Do not commit ZIP, TAR, raw data, full output trees, or private review packages.
Upload only sanitized public bundles to a versioned durable store. Record its
HTTPS or DOI locator and checksum in the capsule or release notes. Keep private
administrator bundles in authorized restricted storage.

## 6. Tag and publish

1. Require the candidate commit and generated views to be clean and current.
2. Create an annotated, preferably signed `vMAJOR.MINOR.PATCH` tag.
3. Push the reviewed commit and tag through the normal protected-branch process.
4. Create GitHub release notes from the dated changelog section. Clearly
   separate software changes, scientific evidence, known limitations, and
   external assets.
5. Attach only validated public artifacts and their checksum manifest.
6. Verify the public repository, citation panel, release assets, hashes, links,
   and installation instructions from a signed-out or clean environment.

Publication to GitHub, Zenodo, PyPI, or any other service is an external state
change and requires explicit current authorization from the release owner.

## 7. Record and respond

After publication, record the release tag and durable artifact locators in the
appropriate canonical records, then rebuild generated views in a follow-up
change if needed. Do not edit a published tag or silently replace an asset.

For a packaging or documentation defect, publish a patch release. For an
incorrect evidence statement, issue a corrected or superseding capsule and
decision. For a security or privacy incident, follow `SECURITY.md`, coordinate
disclosure, rotate exposed credentials, and remove public access to affected
assets where the hosting service permits; retain an auditable incident record
without republishing sensitive content.
