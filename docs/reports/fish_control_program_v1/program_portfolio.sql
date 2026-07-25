-- Canonical report snapshot for the fish inverse-control program portfolio.
-- Values mirror examples/fish_control_program.example.json and its generated audit.
WITH experiment_queue(priority, experiment_id, stage, planned_jobs, readiness, decision_value) AS (
  VALUES
    (1, 'fc00_activation_annotation_panel_v1', 'measurement', 0, 'manual_action_required', 17),
    (2, 'fc01_frozen_detector_tournament_v1', 'measurement', 6, 'blocked', 17),
    (3, 'fc02_structured_background_pu_v1', 'measurement', 12, 'blocked', 11),
    (4, 'fc03_intent_dataset_readiness_v1', 'intent', 0, 'manual_action_required', 17),
    (5, 'fc04_intent_spatiotemporal_ablation_v1', 'intent', 24, 'blocked', 15),
    (6, 'fc05_action_logging_readiness_v1', 'system_identification', 0, 'manual_action_required', 18),
    (7, 'fc06_action_conditioned_system_id_v1', 'system_identification', 8, 'blocked', 12),
    (8, 'fc07_uncertainty_mpc_sim_v1', 'control', 18, 'blocked', 11)
)
SELECT * FROM experiment_queue ORDER BY priority;

WITH experiment_queue(stage, planned_jobs) AS (
  VALUES
    ('Measurement', 0), ('Measurement', 6), ('Measurement', 12),
    ('Intent', 0), ('Intent', 24),
    ('System ID', 0), ('System ID', 8),
    ('Control', 18)
)
SELECT stage, SUM(planned_jobs) AS planned_jobs
FROM experiment_queue
GROUP BY stage
ORDER BY CASE stage
  WHEN 'Measurement' THEN 1
  WHEN 'Intent' THEN 2
  WHEN 'System ID' THEN 3
  ELSE 4
END;
