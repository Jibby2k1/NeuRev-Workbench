-- Reviewed activation evidence transcribed from:
-- Outputs/LearnableContrast/spon_ca_burst_learnable_direct_tuning_v3/metrics.json
-- and its documented v1/v2 comparator artifacts.
-- Rates are fractions. The burst labels are sparse/non-exhaustive, so these
-- values support held-out recall comparisons but not ordinary precision.
SELECT *
FROM (
  VALUES
    ('v1 contrast', 0.132763975, 'learned contrast'),
    ('v2 contrast', 0.205141477, 'learned contrast'),
    ('raw direct', 0.605615942, 'raw direct'),
    ('v3 tuned', 0.605615942, 'learnable direct')
) AS evidence(method, mean_recall, detector_family);

