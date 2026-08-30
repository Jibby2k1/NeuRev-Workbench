# Publication boundary

NeuRev Workbench is designed to make scientific methods and bounded evidence
public without publishing source recordings, participant or reviewer identity,
workstation details, credentials, or rebuildable output trees. This contract
applies before staging a file, opening a pull request, creating a release, or
sharing an annotation package.

## Artifact classes

| Class | Canonical location | Git policy | Examples |
|---|---|---|---|
| Maintained source | `neurobench/`, `tools/`, `tests/`, `schemas/` | Commit after review | Algorithms, schemas, non-mutating audit tools, tiny synthetic fixtures |
| Canonical research record | `research/` and curated `docs/` records | Commit after validation | Experiment protocols, claim and decision records, sanitized evidence capsules |
| Local input | `Inputs/` | Ignore completely, except optional `Inputs/README.md` | Raw movies, spreadsheets, labels, imports |
| Generated output | `Outputs/` | Ignore | Run directories, media, caches, review applications, full result tables |
| Curated public visual | `docs/assets/` or a documented manuscript figure directory | Commit only when small, attributed, and sanitized | Diagrams and data-grounded summary figures |
| Release payload | GitHub Releases, Zenodo, or another versioned artifact store | Do not commit archive files | Reproducibility bundles, manuscript ZIPs, large media |
| Private review material | Administrator-controlled storage | Never commit or publish | Reviewer responses, identity maps, randomization keys, adjudication notes |
| Local environment | `.venv*`, `venv/`, `env/`, `.env*`, `tmp/`, build trees | Ignore | Environment symlinks, credentials, caches, compiler products |

An ignored path that was tracked in an older commit remains tracked until a
deliberate index-only cleanup. `.gitignore` does not rewrite history. Any such
cleanup must be separately reviewed; this policy never authorizes deleting the
working copy or historical experiment outputs.

## Portable evidence contract

A committed evidence capsule is a compact, sanitized statement about a run,
not a copy of its output directory. It should contain:

- stable experiment and run identifiers;
- the code commit and an explicit dirty-state declaration;
- dataset and configuration identifiers or hashes, never private source paths;
- primary metrics, uncertainty, gate results, and validation status;
- an artifact-manifest checksum and a durable external locator when artifacts
  are released;
- the permitted claim scope, limitations, and reviewer or adjudication state.

All committed links and paths must be repository-relative or durable public
URLs. Do not commit `/home/...`, `/Users/...`, drive-letter user directories,
temporary paths, external symlink targets, or links that only resolve against a
local `Outputs/` tree. Public records may name a blinded reviewer role or opaque
review ID, but never a response payload, identity mapping, or private key.

Large generated media and tables remain in `Outputs/` while active. When they
must be shared, create a manifest with sizes and SHA-256 hashes, validate the
package, and publish the package outside Git history. Commit only the sanitized
manifest or evidence capsule and its durable locator.

## Blinded-review boundary

Reviewer-facing Phase A and Phase B packages are distributions, not source
records. They must be generated under `Outputs/`, scanned, and shared in the
declared order. Reviewer submissions and the administrator bundle—including
randomization keys, identity mappings, and adjudication material—remain private.
Only an aggregate, adjudicated, claim-bounded evidence capsule may cross into
the public repository.

## Read-only audit

Run the publication audit before staging or publication:

```bash
.venv-neurobench/bin/python tools/audit_publication_boundary.py --root .
```

The default CI policy exits nonzero on errors while still reporting warnings:

```bash
.venv-neurobench/bin/python tools/audit_publication_boundary.py \
  --root . --format json --fail-on error
```

For a release candidate, make warnings blocking too:

```bash
.venv-neurobench/bin/python tools/audit_publication_boundary.py \
  --root . --fail-on warning
```

The auditor inspects Git-tracked files and untracked, non-ignored candidates.
It never follows symlinks and never writes, moves, stages, or deletes files. It
reports:

- **errors** for tracked local-data/output paths, private payload names,
  likely credentials, absolute user-home paths, and non-portable symlinks;
- **warnings** for archive candidates, unexpectedly large files, files too
  large for bounded text inspection, and relative symlinks that still require
  a portability decision.

Machine-readable output includes stable finding codes and counts. The scanner
is deliberately conservative and is not a substitute for repository-host
secret scanning, history inspection, data-governance review, or human release
approval. Never print or paste a detected secret while resolving a finding.

## Publication checklist

Before merging or releasing:

1. Run the auditor and resolve every error; review every warning.
2. Confirm `git status --short --branch` and inspect the exact staged diff.
3. Verify that no `Inputs/`, `Outputs/`, private-review, archive, environment,
   temporary, or compiler-output path is staged.
4. Regenerate canonical story views, then run their stale-output checks.
5. Validate evidence schemas, scientific gates, local links, checksums, and a
   clean-checkout build.
6. Inspect rendered README, diagrams, documentation, and manuscript views.
7. Publish large packages through the selected release store and verify their
   recorded hashes; do not add the archives to Git.

Passing this checklist establishes publication hygiene and portability. It
does not upgrade scientific evidence, convert candidate yield into precision,
or authorize a broader biological claim.
