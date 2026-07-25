# Report Artifact Index

Reports are generated reader artifacts; their canonical data and source metadata
must remain beside them.

| Report | Canonical source | Generated reader | Status |
|---|---|---|---|
| Fish inverse-control experiment program | `fish_control_program_v1/artifact.json` | [HTML report](fish_control_program_v1/report.html) | verified desktop/mobile |
| Fish neural intent and inverse-control roadmap | `fish_inverse_control_roadmap/artifact.json` | [HTML report](fish_inverse_control_roadmap/report.html) | verified desktop/mobile |

Rules:

- edit `artifact.json` or supporting source files, not generated `report.html`;
- rebuild with the repository-approved portable report builder;
- keep source queries and chart notes beside the artifact;
- treat HTML as generated in code review and search;
- never use a screenshot as the only report deliverable.

