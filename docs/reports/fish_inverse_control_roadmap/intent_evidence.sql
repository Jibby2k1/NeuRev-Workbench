-- Reviewed latent intent-head evidence transcribed from:
-- Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/
-- latent_classifier_report.md
SELECT *
FROM (
  VALUES
    ('Balanced chance', 0.3333, NULL, NULL, 'reference'),
    ('Majority class', 0.3636, NULL, NULL, 'reference'),
    ('Ridge latent head', 0.3636, 0.3611, 0.3545, '11-video LOOV smoke test')
) AS evidence(method, accuracy, balanced_accuracy, macro_f1, scope);

