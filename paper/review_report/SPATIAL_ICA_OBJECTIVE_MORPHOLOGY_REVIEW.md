# Spatial ICA objective morphology review

## Result

The automated protected-v1 analysis is complete at `17_spatial_ica_objective_morphology_v2`. It compares all 79 original sparse-positive observations (27 immutable sites) with deterministic translated controls using identical windows and control coordinates for Raw and spatial ICA. Repeated bursts are aggregated at site grain before bootstrap and sign-flip inference; BH correction spans 12 representation-by-metric tests.

Both Raw and spatial ICA show higher center-to-annulus contrast, sharper radial boundaries, smaller effective spatial radius, and smaller peak displacement at labeled coordinates than at translated controls (all BH q <= 0.0015). Temporal persistence does not differ from controls in either representation. Compactness is inconsistent: Raw is lower at labeled sites (BH q = 0.0203), while spatial ICA is not distinguishable from control (BH q = 0.4863).

## Interpretation

This supports localized morphology as a measurable profile dimension. It does not establish neuron identity, precision, or biological negativity of translated controls. Spatial ICA's positive TIFF is globally linearly encoded and invertible within its recorded limits, but 58.85% of pixels are clipped at zero and 0.20% saturate. Cross-representation amplitude claims are therefore prohibited; the analysis emphasizes within-representation shape contrasts.

The human Raw/ICA/combined morphology packet remains deferred and is not required for this automated result.
