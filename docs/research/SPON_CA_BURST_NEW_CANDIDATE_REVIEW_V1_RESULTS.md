# Spontaneous calcium-burst new-candidate review v1

## Result

One expert reviewed all 18 blinded, previously unmatched detector sites using synchronized six-panel full-trace videos. The conservative normalization produced 9 definite-neuron calls, 4 probable-neuron calls, 4 uncertain calls, and 1 artifact-or-noise call. Combining definite and probable gives a provisional likely-neuron yield of 13/18 (72.2%). No existing site decision was reopened or modified.

## What the review adds

The result supports a broader neuronal appearance envelope than a circular, high-SNR template alone. Positive or probable calls included small and discreet candidates, a low-SNR candidate, and a crescent or possibly multi-source footprint. At the same time, artifact adjacency and stronger neighboring sources remained important identity ambiguities. Size, morphology, SNR, artifact proximity, and overlap were therefore stored as attributes rather than used as automatic exclusions.

## Priority-score audit

The feature-derived score was constructed for review priority from evidence, uncertainty, and crowding—not as calibrated neuron probability. Likely-neuron yield was 4/6 at the top six, 8/12 at the top twelve, and 13/18 overall. The score AUC for likely calls versus unresolved/unlikely calls was 0.308. This inversion is interpretable: uncertain artifact-adjacent or neighbor-confounded sites were deliberately promoted, while several definite neurons had low scores. Future ranking should distinguish two objectives:

1. a review-value score that emphasizes ambiguity and potential failure modes; and
2. a neuron-likelihood score trained and calibrated only after sufficient independent labels exist.

## Evidence boundary

This is a selected new-candidate batch with one reviewer. It does not estimate detector precision because the field was not exhaustively reviewed, and it does not establish one-to-one biological identity. The four uncertain sites remain unresolved rather than negative. The normalized calls are provisional until independent second review and adjudication.

## Artifacts

- `Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/user_review_v1.tsv`: verbatim feedback, site identities, normalized labels, and separate attributes.
- `Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/review_summary_v1.json`: counts, top-k review yields, prioritization audit, and limitations.
- `Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/manifest.json`: immutable blind-ID mapping and proof that existing decisions were not reopened.
