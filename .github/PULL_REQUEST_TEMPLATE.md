## Outcome

<!-- Lead with the user or scientific outcome, then summarize the implementation. -->

## Scope and records

- Program IDs:
- Claim IDs:
- Experiment / run / capsule / decision IDs:
- Out of scope:

## Scientific state

<!-- Check only what this PR actually changes. These dimensions are independent. -->

- [ ] No scientific state changes
- [ ] Experiment lifecycle changes
- [ ] Experiment outcome changes
- [ ] Evidence tier or review state changes
- [ ] Claim state or scope changes
- [ ] Decision action changes

Explain the evidence, limitations, and falsifiers for every checked state change:

## Validation

- [ ] Focused tests pass
- [ ] Relevant broader tests pass
- [ ] Research schemas and cross-record checks pass
- [ ] Generated research and manuscript views were rebuilt and are current
- [ ] Clean-checkout validation passes without ignored `Inputs/` or `Outputs/`
- [ ] Visual or media changes were rendered and inspected

Commands and results:

```text
<exact commands and bounded results>
```

## Publication boundary

- [ ] I inspected the exact diff and staged paths
- [ ] The publication-boundary audit passes at the required severity
- [ ] No raw inputs, output trees, credentials, private review material,
      reviewer identities, randomization keys, absolute workstation paths, or
      non-portable symlinks are included
- [ ] Large artifacts are external, checksummed, and linked through sanitized
      metadata rather than committed archives
- [ ] Third-party code, data, fonts, and visuals have compatible licenses and
      attribution

## Release and rollback

- Changelog entry:
- Compatibility impact:
- External assets or migrations:
- Safe rollback or supersession path:

## Reviewer notes

<!-- Call out uncertainty, deferred gates, performance cost, or areas needing domain review. -->
