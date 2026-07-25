# Fish-control program audit: fish_inverse_control_v1

## Decision

- Planned experiments: `8`.
- Planned compute jobs: `68`.
- Recommended next experiment: `fc00_activation_annotation_panel_v1`.
- Readiness counts: `{"blocked": 5, "manual_action_required": 3}`.

This audit reports readiness; it does not authorize GPU or stimulation work.

## Stage gates

| Gate | Status | Evidence | Blockers |
|---|---|---|---|
| `activation_precision` | `insufficient_evidence` | Learnable-direct v3 tied frozen raw-direct at 0.6056 mean sparse-positive held-out recall.<br>Existing labels contain 79 burst rows across 27 unique ROIs. | No exhaustively reviewed spatial-temporal evaluation panel exists, so ordinary precision is not identifiable. |
| `intent_data` | `insufficient_evidence` | The existing 11-video latent smoke test is video-level and weakly separable.<br>Behavior-alignment utilities exist. | Frame-synchronized movement onset, direction, uncertainty, and leakage-guarded trial windows are not yet available as a frozen dataset. |
| `intent_signal` | `not_started` | The current ridge latent head reached 0.3636 accuracy, equal to the majority baseline. | Intent data gate has not passed. |
| `action_data` | `blocked` | Inverse-dynamics export and timestamp-alignment utilities exist. | Requested and measured stimulation actions, timing, target, amplitude, duration, sham status, and safety interventions are not populated. |
| `system_id` | `not_started` | None recorded | Action-data gate has not passed. |
| `simulator_control` | `not_started` | None recorded | System-identification gate has not passed. |
| `deployment` | `not_started` | Latency p50/p95/p99 instrumentation exists in the repository. | No simulator pass, shadow-mode evidence, approved action envelope, or command interlock exists. |

## Experiment queue

| Priority | Experiment | Stage | Jobs | Readiness | Decision value |
|---:|---|---|---:|---|---:|
| 1 | `fc00_activation_annotation_panel_v1` | measurement | 0 | `manual_action_required` | 17 |
| 2 | `fc01_frozen_detector_tournament_v1` | measurement | 6 | `blocked` | 17 |
|  | ↳ blocker | dependency |  | fc00_activation_annotation_panel_v1 is planned, not completed |  |
|  | ↳ blocker | input |  | missing required input: ../Outputs/FishControl/activation_annotation_panel_v1/benchmark_manifest.json |  |
| 3 | `fc02_structured_background_pu_v1` | measurement | 12 | `blocked` | 11 |
|  | ↳ blocker | dependency |  | fc01_frozen_detector_tournament_v1 is planned, not completed |  |
|  | ↳ blocker | gate |  | activation_precision is insufficient_evidence, not passed |  |
|  | ↳ blocker | input |  | missing required input: ../Outputs/FishControl/frozen_detector_tournament_v1/metrics.json |  |
| 4 | `fc03_intent_dataset_readiness_v1` | intent | 0 | `manual_action_required` | 17 |
| 5 | `fc04_intent_spatiotemporal_ablation_v1` | intent | 24 | `blocked` | 15 |
|  | ↳ blocker | dependency |  | fc03_intent_dataset_readiness_v1 is planned, not completed |  |
|  | ↳ blocker | gate |  | intent_data is insufficient_evidence, not passed |  |
|  | ↳ blocker | input |  | missing required input: ../Outputs/FishControl/intent_dataset_readiness_v1/intent_trials.parquet |  |
| 6 | `fc05_action_logging_readiness_v1` | system_identification | 0 | `manual_action_required` | 18 |
| 7 | `fc06_action_conditioned_system_id_v1` | system_identification | 8 | `blocked` | 12 |
|  | ↳ blocker | dependency |  | fc05_action_logging_readiness_v1 is planned, not completed |  |
|  | ↳ blocker | gate |  | action_data is blocked, not passed |  |
|  | ↳ blocker | input |  | missing required input: ../Outputs/FishControl/action_logging_readiness_v1/transitions.parquet |  |
| 8 | `fc07_uncertainty_mpc_sim_v1` | control | 18 | `blocked` | 11 |
|  | ↳ blocker | dependency |  | fc06_action_conditioned_system_id_v1 is planned, not completed |  |
|  | ↳ blocker | gate |  | system_id is not_started, not passed |  |
|  | ↳ blocker | input |  | missing required input: ../Outputs/FishControl/action_conditioned_system_id_v1/model_manifest.json |  |

## Hardware envelope

- CPU: Intel Core i9-14900K; 32 logical CPUs, recommended experiment cap 16.
- RAM: 78 GiB total, recommended experiment cap 48 GiB.
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12282 MiB; reserve 2200 MiB.
- Concurrent GPU jobs: 1.
