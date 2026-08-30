# Security policy

## Supported versions

NeuRev Workbench has not yet published a stable tagged release. Until then,
security fixes target the latest default branch only.

| Version | Supported |
| --- | --- |
| Latest default branch | Yes |
| Historical commits and unversioned archives | No |

After the first tagged release, this table will identify supported release
lines. Research output archives are immutable evidence and are not patched in
place; a corrected archive receives a new version, checksum, and status record.

## Report a vulnerability privately

Use GitHub's private
[security-advisory form](https://github.com/Jibby2k1/NeuRev-Workbench/security/advisories/new).
Do not open a public issue for a suspected vulnerability, exposed credential,
restricted dataset, reviewer identity, private randomization key, or path that
reveals a contributor's workstation.

Include only what maintainers need to reproduce the problem safely:

- affected commit, version, component, and execution context;
- impact and realistic attack or exposure scenario;
- minimal reproduction using synthetic or non-sensitive data;
- any known mitigation;
- whether credentials or restricted content may already have been exposed.

Do not attach raw recordings, secrets, identity maps, private reviewer
responses, or large generated outputs. If the private advisory channel is not
available, contact the maintainer through a private method listed on the
[repository owner's profile](https://github.com/Jibby2k1) and ask for a secure
transfer channel before sending sensitive material.

Maintainers will acknowledge the report, assess scope and supported versions,
coordinate a fix and disclosure plan where appropriate, and credit reporters
who request attribution. Response timing depends on impact and maintainer
availability; please avoid public disclosure until a coordinated date is
agreed.

## In scope

Security reports may include arbitrary code execution, unsafe deserialization,
path traversal or output overwrite, command injection, credential exposure,
privacy-boundary bypass, malicious archive handling, dependency compromise, or
release-integrity failures.

Scientific disagreements, metric corrections, documentation errors, and
bounded evidence disputes are important but are normally not security issues.
Use the scientific-evidence issue form unless the report would expose
restricted data or an exploitable integrity weakness.

## Release integrity

Official release artifacts must follow `docs/RELEASE_PROCESS.md`: clean-checkout
validation, publication-boundary audit, immutable checksums, exact source tag,
and an external archive for large artifacts. Never trust an archive whose hash
does not match the release manifest.
