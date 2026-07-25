-- Reviewed historical checkpoints used in the report narrative.
SELECT *
FROM (
  VALUES
    ('Pre-label CFAR screens', '36 cascades + two 12-setting ROI sweeps', 'Candidate counts varied substantially; no truth labels', 'Diagnostic only; cannot rank detector quality'),
    ('Grid32 passive dynamics', '1,212 / 1,212 experiments', 'Best test MSE improvement 0.0002295', 'Forecasting baseline, not intent or control'),
    ('Grid128 Stage A', 'Stopped at 477 / 972; 467 metric directories reviewed', 'Best global improvement 0.0008963; active-cell 0.0014562', 'Useful passive dynamics evidence; Stage B not launched'),
    ('Latent intent smoke test', '11 video-level folds', 'Ridge accuracy 0.3636, equal to majority accuracy', 'No validated intent signal yet'),
    ('Learnable detector v1-v3', 'Four sparse-positive held-out windows', 'Best v2 recall 0.2051; v3 tied raw direct at 0.6056', 'Precision remains unidentified; redesign labels and objective')
) AS history(workstream, scale, result, implication);
