# Report source and chart notes

## Activation chart contract

- Question: did learnable contrast/direct tuning exceed the frozen raw-direct
  detector on the existing held-out burst-recall benchmark?
- Takeaway: learned contrast improved from v1 to v2 but remained below
  raw-direct; v3 direct tuning moved parameters and reduced training loss but
  tied the frozen raw-direct recall.
- Chart: one-series bar chart with method on the x-axis and fractional mean
  held-out recall on a zero-based y-axis.
- Color: intentionally omitted because there is one measure and four method
  categories.
- Tooltip: method, recall, and detector family.
- Precision is omitted because the 79 labeled burst rows across 27 unique ROIs
  are non-exhaustive sparse positives; unmatched candidates are not reliable
  false positives.

## Intent table contract

- The intent result is a different task and denominator, so it is not plotted
  on the activation chart.
- It appears as a table containing the three reported smoke-test metrics and
  explicit reference baselines.
- The table is diagnostic only: 11 leave-one-video-out samples are inadequate
  for a robust biological conclusion.

## Omitted visuals

No numeric “program maturity” chart was created. Activation, intent,
action-identification, and controller readiness are categorical stage gates,
not measurements on a valid common scale.

