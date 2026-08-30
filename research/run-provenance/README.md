# Portable run provenance

This directory contains small, sanitized, byte-exact execution records that a
registered run needs for clean-clone inspection. It is not an evidence-capsule
directory and does not turn an engineering run into a completed scientific
experiment.

Each run subdirectory may retain a resolved configuration, artifact index, or
similarly bounded manifest when the full run package remains under ignored
`Outputs/` storage. Files here must:

- use repository-relative or portable data URIs;
- exclude raw recordings, checkpoints, caches, reviewer identities, secrets,
  and workstation paths;
- preserve the exact bytes and SHA-256 recorded by the run record; and
- state incomplete scientific, review, or publication gates in the canonical
  run record rather than implying completion from artifact presence.

Scientific outcomes and claim effects belong in validated capsules under
`research/evidence/`. Generated story views remain downstream of the registry
and must not be edited by hand.
