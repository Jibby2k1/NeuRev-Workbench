/* Hidden, source-backed technical appendix for the Spon Ca Burst deck. */

module.exports = function addTechnicalAppendix(ctx) {
  const { pptx, pptxgen, C, addTitle, panel, label } = ctx;

  function tech(kicker, title, subtitle) {
    const slide = pptx.addSlide("MASTER");
    slide.hidden = true;
    addTitle(slide, kicker, title, subtitle);
    slide.addText("HIDDEN TECHNICAL APPENDIX", {
      x: 10.4, y: 0.36, w: 2.3, h: 0.2, fontSize: 8, bold: true,
      color: C.amber, align: "right", charSpacing: 1.1, margin: 0,
    });
    return slide;
  }

  function row(slide, x, y, key, value, note, width = 5.15) {
    slide.addText(key, {
      x, y, w: 1.48, h: 0.23, fontSize: 9.2, bold: true,
      color: C.cyan, margin: 0, fit: "shrink",
    });
    slide.addText(value, {
      x: x + 1.53, y, w: 1.52, h: 0.23, fontSize: 9.8, bold: true,
      color: C.text, margin: 0, fit: "shrink",
    });
    slide.addText(note, {
      x: x + 3.1, y, w: width - 3.1, h: 0.25, fontSize: 8.5,
      color: C.muted, margin: 0, fit: "shrink",
    });
  }

  function source(slide, text) {
    slide.addText(`Source: ${text}`, {
      x: 0.76, y: 6.79, w: 11.7, h: 0.18, fontSize: 7.1,
      italic: true, color: "64748B", margin: 0,
    });
  }

  function callout(slide, x, y, w, h, text, color = C.amber) {
    slide.addShape(pptx.ShapeType.roundRect, {
      x, y, w, h,
      fill: { color: color === C.green ? "0B3B37" : "3A2A12" },
      line: { color, width: 1 },
    });
    slide.addText(text, {
      x: x + 0.24, y: y + 0.14, w: w - 0.48, h: h - 0.25,
      fontSize: 10.5, bold: true, color: C.text, align: "center",
      margin: 0, fit: "shrink",
    });
  }

  // A1 — dataset and evaluation contract
  {
    const slide = tech(
      "Technical A1",
      "Dataset and evaluation contract",
      "The conventions behind every result in the visible deck."
    );
    panel(slide, 0.7, 1.75, 5.9, 4.82);
    label(slide, "Acquisition and windows", 1.02, 2.05, 2.5);
    row(slide, 1.02, 2.43, "Frame period", "20 ms", "50 Hz acquisition");
    row(slide, 1.02, 2.80, "Review", "UI 1800–2359", "560 frames; inclusive");
    row(slide, 1.02, 3.17, "Quiet calibration", "UI 1800–1899", "100 frames = 2.0 s");
    row(slide, 1.02, 3.54, "NumPy interval", "1799:2359", "zero-based, half-open");
    row(slide, 1.02, 3.91, "Image shape", "340 × 573", "y=row; x=column");
    row(slide, 1.02, 4.28, "Known labels", "79 points", "27 ROI identities");
    row(slide, 1.02, 4.65, "Labels / burst", "15, 20, 21, 23", "four windows");
    row(slide, 1.02, 5.02, "Burst UI 1–2", "2003–2026", "2040–2063");
    row(slide, 1.02, 5.39, "Burst UI 3–4", "2122–2149", "2254–2300");

    panel(slide, 6.88, 1.75, 5.75, 4.82, C.panel2);
    label(slide, "Detection and statistics", 7.2, 2.05, 2.5);
    row(slide, 7.2, 2.43, "Match radius", "6 px", "one-to-one matching", 5.0);
    row(slide, 7.2, 2.80, "NMS distance", "6 px", "nonmaximum suppression", 5.0);
    row(slide, 7.2, 3.17, "Capacity", "58 / burst", "232 total; Raw burden", 5.0);
    row(slide, 7.2, 3.54, "Bootstrap", "10,000", "paired over 27 ROIs", 5.0);
    row(slide, 7.2, 3.91, "95% interval", "percentile", "+.0119 to +.2254", 5.0);
    row(slide, 7.2, 4.28, "Deadline", "20 ms / frame", "50 Hz software budget", 5.0);
    callout(
      slide, 7.2, 4.88, 4.75, 1.05,
      "Sparse positives support recall and candidate burden. Unmatched candidates are unknown—not false positives—so ordinary precision is not identified."
    );
    source(slide, "resolved configs; labels_normalized.tsv; causal proposal preflight.json");
  }

  // A2 — EMA, gate, and displayed pipeline
  {
    const slide = tech(
      "Technical A2",
      "Exact EMA and slide-4 gate parameters",
      "The EMA recursion is causal; the displayed visualization as a whole is not."
    );
    panel(slide, 0.7, 1.75, 5.65, 4.82);
    label(slide, "Exponential moving average", 1.03, 2.07, 2.8);
    slide.addText("Eₜ = αxₜ + (1−α)Eₜ₋₁", {
      x: 1.03, y: 2.48, w: 4.65, h: 0.43, fontSize: 22,
      bold: true, color: C.text, margin: 0, align: "center",
    });
    slide.addText("α = 2 / (span + 1)", {
      x: 1.03, y: 3.08, w: 4.65, h: 0.34, fontSize: 18,
      bold: true, color: C.cyan, margin: 0, align: "center",
    });
    slide.addText("xₜ = zₜ²; span = 4; α = 0.4; E₀ = z₀²", {
      x: 1.03, y: 3.62, w: 4.65, h: 0.3, fontSize: 13.5,
      bold: true, color: C.text, margin: 0, align: "center",
    });
    slide.addText("Gₜ = 1 − exp[−Eₜ/(2τ²)]", {
      x: 1.03, y: 4.25, w: 4.65, h: 0.36, fontSize: 18,
      bold: true, color: C.text, margin: 0, align: "center",
    });
    slide.addText("τ = 2.5 z;  Yₜ = Xₜ[f + (1−f)Gₜ];  f = 0.2", {
      x: 1.03, y: 4.83, w: 4.65, h: 0.3, fontSize: 13.2,
      bold: true, color: C.cyan, margin: 0, align: "center",
    });
    slide.addText("Larger span → smaller α → longer memory.", {
      x: 1.03, y: 5.48, w: 4.65, h: 0.28, fontSize: 12,
      color: C.muted, margin: 0, align: "center",
    });

    panel(slide, 6.63, 1.75, 6.0, 4.82, C.panel2);
    label(slide, "Full displayed pipeline", 6.96, 2.07, 2.4);
    row(slide, 6.96, 2.43, "Spatial", "Gaussian σ=1 px", "reflect boundary", 5.3);
    row(slide, 6.96, 2.78, "Temporal", "Savitzky–Golay", "window 7; polyorder 2", 5.3);
    row(slide, 6.96, 3.13, "Derivative", "lag 1 = 20 ms", "Iₜ−Iₜ₋₁", 5.3);
    row(slide, 6.96, 3.48, "Quiet scale", "1.4826 × MAD", "10th-pct floor", 5.3);
    row(slide, 6.96, 3.83, "Compression", "asinh gain 5", "fixed over time", 5.3);
    row(slide, 6.96, 4.18, "Artifact factor", "1−0.7A", "A: persistent-bright", 5.3);
    row(slide, 6.96, 4.53, "Motion correction", "none", "motion edges survive", 5.3);
    callout(
      slide, 6.96, 5.02, 4.95, 0.92,
      "Centered 7-frame smoothing uses 3 frames / 60 ms of look-ahead. Multiplicative change gating also suppresses slow calcium.",
      C.red
    );
    source(slide, "activity_gated_video.py; activity-gate config.resolved.json");
  }
  // A3 — causal winner
  {
    const s = tech("Technical A3", "Capacity-matched causal winner: exact settings",
      "Slide 5 uses fractional_ecbe304455—not the centered slide-4 TIFF.");
    panel(s, 0.7, 1.75, 5.75, 4.82);
    label(s, "Selected causal pipeline", 1.03, 2.07, 2.5);
    row(s, 1.03, 2.43, "Spatial", "Gaussian σ=1 px", "reflect boundary");
    row(s, 1.03, 2.80, "Temporal EMA", "span 2; α=2/3", "strictly causal");
    row(s, 1.03, 3.17, "Artifact attenuation", "0.7", "multiply by 1−0.7A");
    row(s, 1.03, 3.54, "Intensity", "asinh gain 10", "quiet pct 1, 99.8");
    row(s, 1.03, 3.91, "Baseline", "frozen median", "first 100 frames");
    row(s, 1.03, 4.28, "Residual", "positive only", "max[(X−B)/scale,0]");
    row(s, 1.03, 4.65, "Pooling", "LME τ=0.1", "τ[logsumexp(r/τ)−logT]");
    row(s, 1.03, 5.02, "Peak policy", "58 / burst", "232 candidates");
    row(s, 1.03, 5.39, "Match / NMS", "6 px / 6 px", "same as Raw");
    panel(s, 6.73, 1.75, 5.9, 4.82, C.panel2);
    label(s, "Search and evidence", 7.06, 2.07, 2.3);
    row(s, 7.06, 2.43, "Breadth", "72 methods", "648 evaluations", 5.15);
    row(s, 7.06, 2.80, "Fusion", "96 methods", "864 max evaluations", 5.15);
    row(s, 7.06, 3.17, "Robustness", "12 × 31", "372 evaluations", 5.15);
    row(s, 7.06, 3.54, "Logical total", "1,884", "all completed", 5.15);
    row(s, 7.06, 3.91, "Recall", "58 / 79", "0.7342; 4/4 wins", 5.15);
    row(s, 7.06, 4.28, "Runtime median", "5.96 ms", "pooling + NMS", 5.15);
    row(s, 7.06, 4.65, "Runtime p95", "6.41 ms", "13.59 ms headroom", 5.15);
    row(s, 7.06, 5.02, "Worst condition", "0.6464", "photobleach", 5.15);
    callout(s, 7.06, 5.49, 4.85, 0.55,
      "This result is genuinely causal and real-time feasible.", C.green);
    source(s, "causal proposal stage_a_results.jsonl row 43; resolved_config.json");
  }
  // A4 — ICA and fusion
  {
    const s = tech("Technical A4", "Pairwise ICA and fusion parameters",
      "Adjacent-frame ICA nearly reproduces the derivative; fusion was intentionally bounded.");
    panel(s, 0.7, 1.75, 4.0, 4.82);
    label(s, "Shared ICA setup", 1.02, 2.07, 2.0);
    row(s, 1.02, 2.43, "Input", "[Iₜ₋₁,Iₜ]", "lag 1 = 20 ms", 3.3);
    row(s, 1.02, 2.80, "Preprocess", "σ=1; EMA span 4", "α=.4; causal", 3.3);
    row(s, 1.02, 3.17, "Seed", "20260727", "uniform anatomy", 3.3);
    row(s, 1.02, 3.54, "Samples", "1024 / 4096", "screen / confirm", 3.3);
    row(s, 1.02, 3.91, "Primary z", "3.0", "grid 2–4", 3.3);
    row(s, 1.02, 4.28, "Min pixels", "1, 3, 5", "one-sided", 3.3);
    row(s, 1.02, 4.65, "Covariance κ", "≈22,990", "collinear", 3.3);
    row(s, 1.02, 5.02, "InfoMax", "lr .01; 500 iters", "tol 1e−7; 6 starts", 3.3);
    row(s, 1.02, 5.39, "CS-Parzen", "bw .35; block 256", "3° then .25°", 3.3);
    panel(s, 4.98, 1.75, 3.45, 4.82, C.panel2);
    label(s, "Recovered directions", 5.3, 2.07, 2.1);
    s.addText("InfoMax", {x:5.3,y:2.58,w:1,h:.22,fontSize:9,color:C.muted,margin:0});
    s.addText("[−.698722, .715393]", {x:5.3,y:2.91,w:2.55,h:.3,fontSize:14,bold:true,color:C.text,align:"center",margin:0});
    s.addText("cos = .9999305", {x:5.3,y:3.35,w:2.55,h:.25,fontSize:10.5,bold:true,color:C.cyan,align:"center",margin:0});
    s.addText("CS-Parzen", {x:5.3,y:4.05,w:1,h:.22,fontSize:9,color:C.muted,margin:0});
    s.addText("[−.707394, .706819]", {x:5.3,y:4.38,w:2.55,h:.3,fontSize:14,bold:true,color:C.text,align:"center",margin:0});
    s.addText("cos ≈ .9999999", {x:5.3,y:4.82,w:2.55,h:.25,fontSize:10.5,bold:true,color:C.cyan,align:"center",margin:0});
    s.addText("to normalized [−1,1]", {x:5.3,y:5.52,w:2.55,h:.3,fontSize:10.5,color:C.muted,align:"center",margin:0});
    panel(s, 8.7, 1.75, 3.93, 4.82);
    label(s, "Fusion tuning", 9.02, 2.07, 1.7);
    row(s, 9.02, 2.43, "Fixed λ", ".05,.1,.2,.4", "12 lanes", 3.25);
    row(s, 9.02, 2.80, "Feature clip", "5 z", "bounded", 3.25);
    row(s, 9.02, 3.17, "Gate floors", ".5,.7,.85", "nonzero", 3.25);
    row(s, 9.02, 3.54, "Learned lr", ".001", "300 epochs", 3.25);
    row(s, 9.02, 3.91, "L2 / max λ", ".1 / .4", "hard bound", 3.25);
    row(s, 9.02, 4.28, "Learned λ", ".0364–.0460", "feature-specific", 3.25);
    row(s, 9.02, 4.65, "Fixed result", "49 / 79", "tied Raw", 3.25);
    row(s, 9.02, 5.02, "Nested", ".5806", "Raw .6056", 3.25);
    row(s, 9.02, 5.39, "Best gate", "47 / 79", "floor .85", 3.25);
    source(s, "pairwise and fusion config.resolved.json; ICA fit.json files");
  }

  // A5 — latent AR(1)
  {
    const s = tech("Technical A5", "Latent amplitude model: equation and parameters",
      "The stable shared AR(1) model was selected by likelihood without labels.");
    panel(s, 0.7, 1.75, 5.55, 4.82);
    label(s, "State-space model", 1.03, 2.07, 1.9);
    s.addText("xₜ = γxₜ₋₁ + ηₜ,   ηₜ ~ N(0,q)", {x:1.03,y:2.48,w:4.55,h:.36,fontSize:17,bold:true,color:C.text,margin:0,align:"center"});
    s.addText("yₜ = xₜ + εₜ,         εₜ ~ N(0,r)", {x:1.03,y:3.01,w:4.55,h:.36,fontSize:17,bold:true,color:C.text,margin:0,align:"center"});
    s.addText("γ = exp(−Δt / τ)", {x:1.03,y:3.56,w:4.55,h:.36,fontSize:18,bold:true,color:C.cyan,margin:0,align:"center"});
    row(s, 1.03, 4.16, "Frame period Δt", "20 ms", "");
    row(s, 1.03, 4.53, "Selected τ", "1280 ms", "top grid boundary");
    row(s, 1.03, 4.90, "Selected γ", ".98449644", "margin .01550");
    row(s, 1.03, 5.27, "q / r", ".03", "process / observation");
    row(s, 1.03, 5.64, "q ; r", ".028888 ; .962932", "shared across pixels");
    panel(s, 6.53, 1.75, 6.1, 4.82, C.panel2);
    label(s, "Fit grid and interpretation", 6.86, 2.07, 2.6);
    row(s, 6.86, 2.43, "τ grid (ms)", "40…1280", "40,80,160,320,640,1280", 5.35);
    row(s, 6.86, 2.80, "q/r grid", ".01…1.0", ".01,.03,.1,.3,1", 5.35);
    row(s, 6.86, 3.17, "Models", "6 × 5 = 30", "bounded likelihood", 5.35);
    row(s, 6.86, 3.54, "Sample pixels", "4096", "seed 20260727", 5.35);
    row(s, 6.86, 3.91, "Validation", "5 blocks", "labels not used", 5.35);
    row(s, 6.86, 4.28, "Causal filter", "53 / 79", "312 candidates", 5.35);
    row(s, 6.86, 4.65, "RTS smoother", "55 / 79", "320 candidates", 5.35);
    callout(s, 6.86, 5.18, 5.05, .88,
      "The smoother uses future frames and is an offline upper bound. Only the forward filter is eligible for real-time control.");
    source(s, "latent dynamics config.resolved.json; fit/selected_model.json");
  }
  // A6 — compact parameter glossary
  {
    const s = tech("Technical A6", "Parameter glossary for rapid Q&A",
      "A compact answer key for the symbols used throughout the deck.");
    panel(s, 0.7, 1.75, 5.75, 4.82);
    label(s, "Signal processing", 1.03, 2.07, 2.1);
    row(s, 1.03, 2.43, "σ", "Gaussian spatial SD", "pixels");
    row(s, 1.03, 2.80, "span", "EMA memory control", "α=2/(span+1)");
    row(s, 1.03, 3.17, "α", "EMA new-sample weight", "larger = faster");
    row(s, 1.03, 3.54, "z", "quiet-standardized Δ", "median/MAD");
    row(s, 1.03, 3.91, "τ gate", "gate sensitivity", "2.5 z on slide 4");
    row(s, 1.03, 4.28, "f", "structural floor", "amplitude retained at G=0");
    row(s, 1.03, 4.65, "A", "persistent artifact score", "quiet-only; [0,1]");
    row(s, 1.03, 5.02, "LME τ", "pooling softness", "small = peak-sensitive");
    row(s, 1.03, 5.39, "λ", "auxiliary feature weight", "zero = Raw only");
    panel(s, 6.73, 1.75, 5.9, 4.82, C.panel2);
    label(s, "Model and evaluation", 7.06, 2.07, 2.3);
    row(s, 7.06, 2.43, "γ", "AR(1) persistence", "exp(−Δt/τ)", 5.15);
    row(s, 7.06, 2.80, "q / r", "process / observation", "variance ratio", 5.15);
    row(s, 7.06, 3.17, "NMS", "nonmaximum suppression", "one local peak", 5.15);
    row(s, 7.06, 3.54, "Recall", "matched / known labels", "positives only", 5.15);
    row(s, 7.06, 3.91, "Candidate burden", "number proposed", "review workload", 5.15);
    row(s, 7.06, 4.28, "Capacity match", "same candidate count", "fair recall test", 5.15);
    row(s, 7.06, 4.65, "Causal", "past/current only", "real-time eligible", 5.15);
    row(s, 7.06, 5.02, "Offline", "may use future frames", "upper bound", 5.15);
    source(s, "implementation definitions and resolved experiment contracts");
  }
};
