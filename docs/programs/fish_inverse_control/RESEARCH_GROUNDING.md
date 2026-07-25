# Research Grounding for Fish Intent and Inverse Control

Reviewed through 2026-07-21. Sources below are primary papers or official
proceedings. “Program implication” is an inference for this repository, not a
claim made by the paper.

## Detection is a structured source-separation problem

### Joint spatial, temporal, and background modeling

[Pnevmatikakis et al. (2016), simultaneous denoising, deconvolution, and
demixing](https://pmc.ncbi.nlm.nih.gov/articles/PMC4881387/) jointly estimates
spatial footprints, calcium dynamics, background, and noise with CNMF.

Program implication: learnable CFAR should be compared with a modular
footprint → trace → deconvolution pipeline. A learnable contrast statistic alone
does not address overlapping sources, kinetics, or structured background.

[Zhou et al. (2018), CNMF-E](https://elifesciences.org/articles/28728) models
local background using pixels beyond a soma radius because background may be
stronger than neural signal and vary on similar timescales.

Program implication: the best “soft CFAR” candidate is a soma-excluding annular
background model that retains absolute amplitude, not unconstrained smoothing.

[Giovannucci et al. (2019), CaImAn](https://elifesciences.org/articles/38173)
integrates motion correction, source extraction, deconvolution, registration,
and streaming analysis, benchmarked with multi-labeler annotations.

Program implication: use the existing external-tool attachment architecture to
add a frozen CaImAn comparator before building another custom detector family.

### Noise and event inference must match the acquisition

[Rupprecht et al. (2021), CASCADE](https://www.nature.com/articles/s41593-021-00895-5)
trained on simultaneous spike/calcium ground truth, including zebrafish, and
noise-matched models to target frame rate and SNR.

Program implication: stratify by SNR, baseline intensity, frame rate, fish, and
session; do not expect one normalization to generalize universally.

[Friedrich et al. (2017), OASIS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)
provides fast online calcium deconvolution suitable for closed-loop latency.

Program implication: treat event inference as a post-ROI temporal module and
benchmark its timing/precision separately from spatial neuron discovery.

[Lecoq et al. (2021), DeepInterpolation](https://www.nature.com/articles/s41592-021-01285-2)
and [Li et al. (2021), DeepCAD](https://www.nature.com/articles/s41592-021-01225-0)
show that self-supervised spatiotemporal denoising can improve calcium-imaging
SNR without clean targets.

Program implication: denoising is a bounded preprocessing comparator, not an
assumed improvement. It must pass direct-amplitude preservation and artifact
strata on the sealed benchmark.

### Precision and attribution require better truth

[Berens et al. (2018), community spike-inference benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC5997358/)
used hidden real-data tests, multiple indicators/datasets, and multiple metrics;
different algorithms often performed similarly.

Program implication: retain a sealed test panel and report event PR, temporal
tolerance, calibration, and group uncertainty—not one four-burst recall mean.

[Gauthier et al. (2022), false-transient correction](https://www.nature.com/articles/s41592-022-01422-5)
found material transient misattribution and used residual spatial diagnostics.

Program implication: add neighbor contamination, residual correlation,
duplicate, split, and merge errors. A detected event assigned to the wrong ROI
is a precision failure even if its time is correct.

[Kiryo et al. (2017), non-negative PU learning](https://papers.nips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)
addresses overfitting when training flexible classifiers from positives and
unlabeled examples.

Program implication: PU risk is worth one bounded confirmatory branch after
estimating class prior and label-selection assumptions. It does not eliminate
the need for an exhaustive precision test.

## Left/right activity is likely both spatial and temporal

[Dunn et al. (2016), brain-wide exploratory locomotion](https://elifesciences.org/articles/12741)
identified lateralized bilateral ARTR populations with slow dynamics that
predicted and causally biased turn direction.

Program implication: test signed hemispheric imbalance, recent state history,
and their interaction. The literature argues against assuming either spatial or
temporal information is dispensable.

[Koyama et al. (2016), bilateral turn-choice circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC4978520/)
identified bilateral competition and feed-forward inhibition in left/right
escape choice.

Program implication: include ipsilateral-minus-contralateral features and
left/right mirror controls. A genuine spatial representation should collapse
when coordinates are mirrored or shuffled.

[Naumann et al. (2016), zebrafish optomotor circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC5111816/)
showed sequential integration of direction- and eye-specific sensory streams,
interhemispheric inhibition, and separation of turning from forward-swimming
commands.

Program implication: condition on stimulus direction and separate turn
direction from forward locomotion to avoid decoding input or execution.

[Dragomir et al. (2020), evidence accumulation](https://www.nature.com/articles/s41593-019-0535-8)
found distributed decision signals spanning multiple timescales, influenced by
sensory and motor history.

Program implication: pre-movement neural windows need history-only behavioral
baselines, multiple lead windows, and time-shift controls.

[Visuomotor decision-making through multifeature convergence (2026)](https://www.nature.com/articles/s41467-026-69633-4)
used congruent, conflicting, and blank sensory conditions plus withheld stimulus
configurations.

Program implication: conflicting/blank trials and withheld configurations are a
strong template for separating intent from stimulus decoding.

## Control requires measured actions and uncertainty

[Bolus et al. (2021), state-space optogenetic control](https://pmc.ncbi.nlm.nih.gov/articles/PMC8356067/)
identified stimulation-to-spiking dynamics and used optimal feedback with
adaptive state estimation.

Program implication: begin with controlled linear state-space identification;
validate target and off-target populations before nonlinear models.

[Chai et al. (2023), all-optical zebrafish interrogation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10776927/)
showed unilateral stimulation could bias turning, while many decodable regions
were not necessarily causally effective.

Program implication: decoding is not controllability. Require sham,
opsin-negative when applicable, no-action, contralateral, and off-target
controls.

[Sani et al. (2021), preferential subspace identification](https://www.nature.com/articles/s41593-020-00733-0)
prioritizes behaviorally relevant neural dynamics, while
[input preferential subspace identification (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10873612/)
dissociates intrinsic and measured-input-driven dynamics.

Program implication: a controlled linear/IPSID baseline is more informative and
sample-efficient than immediately adding action channels to a large recurrent
network.

[Chua et al. (2018), PETS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)
uses probabilistic dynamics ensembles and trajectory sampling for
uncertainty-aware, sample-efficient model-based control.

Program implication: use an ensemble as a simulation/MPC comparator and reject
unsupported trajectories. Ensemble uncertainty is not itself a biological
safety guarantee.

[Sui et al. (2015), SafeOpt](https://proceedings.mlr.press/v37/sui15.html)
restricts exploration to points predicted to satisfy a safety threshold under
explicit assumptions.

Program implication: exploration must stay within a separately approved action
envelope. Do not import theoretical guarantees without validating their
assumptions for fish physiology.

## Resulting search-space constraints

The literature narrows the next work to:

- exhaustive truth and attribution-aware metrics before precision claims;
- structured, soma-excluding background plus explicit trace/event inference;
- frozen external comparators before new architecture proliferation;
- spatial laterality × temporal history ablations with stimulus/motion leakage
  controls;
- measured actions and simple controlled state-space baselines before deep
  inverse models;
- uncertainty-gated simulation and shadow mode before stimulation commands.

