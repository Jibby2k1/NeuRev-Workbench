#!/usr/bin/env node
/* Build a self-contained PowerPoint with embedded Spon Ca Burst videos. */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const root = path.resolve(__dirname, "..");
const outDir = path.join(
  root,
  "Outputs/Presentations/spon_ca_burst_temporal_methods_meeting_2026-07-28"
);
const mediaDir = path.join(outDir, "media");
const outPath = path.join(outDir, "Spon_Ca_Burst_Temporal_Methods_Meeting.pptx");

if (fs.existsSync(outPath)) {
  throw new Error(`Refusing to overwrite ${outPath}`);
}

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "NeuRev Workbench";
pptx.company = "CNEL";
pptx.subject = "Spon Ca Burst temporal features, ICA, and amplitude preservation";
pptx.title = "What temporal change adds—and what it removes";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineSlideMaster({
  title: "MASTER",
  background: { color: "07101D" },
  objects: [
    {
      rect: {
        x: 0,
        y: 0,
        w: 13.333,
        h: 0.08,
        fill: { color: "22D3EE" },
        line: { color: "22D3EE" },
      },
    },
    {
      text: {
        text: "NEUREV WORKBENCH  /  SPON CA BURST",
        options: {
          x: 0.55,
          y: 7.12,
          w: 5.8,
          h: 0.16,
          fontFace: "Aptos",
          fontSize: 8,
          bold: true,
          color: "64748B",
          margin: 0,
          charSpacing: 1.1,
        },
      },
    },
  ],
  slideNumber: {
    x: 12.3,
    y: 7.08,
    w: 0.45,
    h: 0.2,
    color: "64748B",
    fontFace: "Aptos",
    fontSize: 8,
    align: "right",
  },
});

const C = {
  bg: "07101D",
  panel: "0E1B2C",
  panel2: "132337",
  text: "F4F8FC",
  muted: "9FB0C5",
  cyan: "22D3EE",
  cyan2: "0891B2",
  green: "34D399",
  amber: "FBBF24",
  red: "FB7185",
  slate: "334155",
  white: "FFFFFF",
};

function addTitle(slide, kicker, title, subtitle) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.62, y: 0.35, w: 5.7, h: 0.22,
    fontSize: 9, bold: true, color: C.cyan, charSpacing: 1.4, margin: 0,
  });
  slide.addText(title, {
    x: 0.62, y: 0.63, w: 12.0, h: 0.55,
    fontSize: 27, bold: true, color: C.text, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.64, y: 1.21, w: 11.9, h: 0.35,
      fontSize: 11.5, color: C.muted, margin: 0, breakLine: false,
    });
  }
}

function panel(slide, x, y, w, h, fill = C.panel) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
}

function label(slide, text, x, y, w, color = C.cyan) {
  slide.addText(text.toUpperCase(), {
    x, y, w, h: 0.22,
    fontSize: 8.5, bold: true, color, charSpacing: 1.1, margin: 0,
  });
}

function metric(slide, x, y, value, title, color = C.text, width = 1.8) {
  slide.addText(value, {
    x, y, w: width, h: 0.46,
    fontSize: 25, bold: true, color, margin: 0,
  });
  slide.addText(title, {
    x, y: y + 0.48, w: width, h: 0.38,
    fontSize: 9.5, color: C.muted, margin: 0,
  });
}

function addVideo(slide, key, x, y, w, h) {
  const video = path.join(mediaDir, `${key}.mp4`);
  const poster = path.join(mediaDir, `${key}_poster.png`);
  const cover = `data:image/png;base64,${fs.readFileSync(poster).toString("base64")}`;
  slide.addMedia({
    type: "video",
    path: video,
    cover,
    x, y, w, h,
    objectName: key,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + w - 0.68, y: y + h - 0.52, w: 0.48, h: 0.32,
    fill: { color: C.cyan2, transparency: 5 },
    line: { color: C.cyan2 },
  });
  slide.addText("▶", {
    x: x + w - 0.61, y: y + h - 0.47, w: 0.26, h: 0.15,
    fontSize: 10, bold: true, align: "center", color: C.white, margin: 0,
  });
}

function bullet(slide, text, x, y, w, color = C.text, size = 14) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y: y + 0.1, w: 0.08, h: 0.08,
    fill: { color: C.cyan }, line: { color: C.cyan },
  });
  slide.addText(text, {
    x: x + 0.18, y, w: w - 0.18, h: 0.42,
    fontSize: size, color, margin: 0, breakLine: false,
  });
}

function bar(slide, x, y, w, labelText, value, max, color, detail) {
  slide.addText(labelText, {
    x, y, w: 2.15, h: 0.25, fontSize: 10.5, color: C.text, margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 2.18, y: y + 0.02, w, h: 0.18,
    fill: { color: C.slate }, line: { color: C.slate },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 2.18, y: y + 0.02, w: w * value / max, h: 0.18,
    fill: { color }, line: { color },
  });
  slide.addText(detail, {
    x: x + 2.23 + w, y: y - 0.01, w: 0.64, h: 0.24,
    fontSize: 9.5, bold: true, color, margin: 0, fit: "shrink",
  });
}

// 1 — Cover
{
  const slide = pptx.addSlide("MASTER");
  slide.background = { color: C.bg };
  slide.addImage({
    path: path.join(mediaDir, "raw_plus_ica_poster.png"),
    x: 7.45, y: 0.55, w: 5.25, h: 3.49,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.65, y: 0, w: 6.68, h: 7.5,
    fill: { color: C.bg, transparency: 50 },
    line: { color: C.bg, transparency: 100 },
  });
  slide.addText("SPON CA BURST  /  MEETING READOUT", {
    x: 0.72, y: 0.72, w: 5.5, h: 0.25,
    fontSize: 10, bold: true, color: C.cyan, charSpacing: 1.3, margin: 0,
  });
  slide.addText("What temporal change adds—\nand what it removes", {
    x: 0.7, y: 1.15, w: 7.3, h: 1.55,
    fontSize: 32, bold: true, color: C.text, margin: 0, breakLine: false,
  });
  slide.addText(
    "Differencing, adaptive artifact suppression, ICA, feature fusion, and latent smoothing",
    {
      x: 0.74, y: 2.92, w: 5.9, h: 0.72,
      fontSize: 15, color: C.muted, margin: 0,
    }
  );
  panel(slide, 0.72, 4.55, 5.75, 1.12, C.panel2);
  slide.addText("BOTTOM LINE", {
    x: 1.0, y: 4.82, w: 1.3, h: 0.2,
    fontSize: 9, bold: true, color: C.cyan, charSpacing: 1.2, margin: 0,
  });
  slide.addText(
    "Preserve amplitude for detection. Use temporal change as auxiliary evidence—not as a hard gate.",
    {
      x: 1.0, y: 5.08, w: 5.02, h: 0.42,
      fontSize: 15.5, bold: true, color: C.text, margin: 0,
    }
  );
  slide.addText("28 JUL 2026", {
    x: 0.74, y: 6.42, w: 2.0, h: 0.2,
    fontSize: 9, bold: true, color: C.muted, charSpacing: 1.2, margin: 0,
  });
  slide.addNotes(
    "Open with the decision, not the algorithm. The experiments agree that temporal change is informative, but hard temporal gating destroys slow calcium activity."
  );
}

// 2 — Evaluation contract
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Evaluation anchor",
    "A small labeled set makes candidate burden inseparable from recall",
    "Four annotated bursts; unlabeled event pixels remain unknown—not negative."
  );
  panel(slide, 0.62, 1.75, 4.0, 4.78);
  label(slide, "Raw Direct anchor", 0.95, 2.05, 2.2);
  metric(slide, 0.95, 2.45, "0.6056", "macro held-out recall", C.text, 2.1);
  metric(slide, 0.95, 3.5, "49 / 79", "known labels matched", C.green, 2.1);
  metric(slide, 0.95, 4.55, "232", "detected candidates", C.amber, 2.1);
  slide.addText("Burst recalls", {
    x: 0.95, y: 5.57, w: 1.3, h: 0.2, fontSize: 9, bold: true, color: C.muted, margin: 0,
  });
  slide.addText("0.467   0.550   0.667   0.739", {
    x: 0.95, y: 5.86, w: 2.9, h: 0.26, fontSize: 13, bold: true, color: C.text, margin: 0,
  });

  panel(slide, 4.92, 1.75, 7.8, 2.14);
  label(slide, "What the numbers can—and cannot—say", 5.25, 2.05, 3.7);
  bullet(slide, "Recall is exact against the known positive annotations.", 5.25, 2.48, 6.9);
  bullet(slide, "Candidate count is the operational cost of visual review.", 5.25, 3.03, 6.9);
  panel(slide, 4.92, 4.15, 7.8, 2.38, C.panel2);
  label(slide, "Precision caveat", 5.25, 4.48, 2.3, C.amber);
  slide.addText(
    "We do not have exhaustive negatives, so “known-label candidate fraction” is only a lower bound—not precision.",
    {
      x: 5.25, y: 4.84, w: 6.8, h: 0.62,
      fontSize: 17, bold: true, color: C.text, margin: 0,
    }
  );
  slide.addText(
    "This is why the best next evidence is targeted manual review of novel candidates, followed by causal confirmation.",
    {
      x: 5.25, y: 5.7, w: 6.75, h: 0.42,
      fontSize: 11.5, color: C.muted, margin: 0,
    }
  );
  slide.addNotes(
    "Keep the precision caveat explicit. Unmatched candidates may be true biological events because labels are sparse."
  );
}

// 3 — Raw video
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Visual anchor",
    "The raw movie mixes anatomy, persistent brightness, and transient activity",
    "UI frames 1800–2359; annotated bursts are flagged in the video."
  );
  addVideo(slide, "raw", 0.72, 1.78, 8.2, 5.45);
  panel(slide, 9.22, 1.78, 3.45, 5.45);
  label(slide, "Look for", 9.53, 2.08, 1.3);
  bullet(slide, "Stable anatomy that should remain visible", 9.52, 2.54, 2.72, C.text, 12.5);
  bullet(slide, "Persistent bright artifacts that compete with cells", 9.52, 3.35, 2.72, C.text, 12.5);
  bullet(slide, "Slow and propagating calcium dynamics", 9.52, 4.28, 2.72, C.text, 12.5);
  slide.addShape(pptx.ShapeType.line, {
    x: 9.53, y: 5.3, w: 2.78, h: 0,
    line: { color: C.slate, width: 1 },
  });
  slide.addText("Question", {
    x: 9.53, y: 5.58, w: 1.0, h: 0.2,
    fontSize: 9, bold: true, color: C.cyan, margin: 0,
  });
  slide.addText(
    "Can we suppress nuisance persistence without erasing biologically slow activity?",
    {
      x: 9.53, y: 5.92, w: 2.65, h: 0.72,
      fontSize: 15, bold: true, color: C.text, margin: 0,
    }
  );
  slide.addNotes(
    "Play the clip and ask the room to identify stable anatomy, obvious artifacts, and transient propagation before showing processed outputs."
  );
}

// 4 — Derivative gate
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Differencing result",
    "Pure derivative-energy gating is visually selective—and scientifically too harsh",
    "The useful variant preserves amplitude while adapting to persistent background."
  );
  addVideo(slide, "artifact_gate", 0.7, 1.74, 7.35, 4.89);
  panel(slide, 8.35, 1.74, 4.28, 2.05);
  label(slide, "Proposed temporal support", 8.68, 2.04, 2.7);
  slide.addText("Gₜ = 1 − exp[−EMA(zₜ²)/(2·2.5²)]", {
    x: 8.67, y: 2.48, w: 3.55, h: 0.34,
    fontSize: 16, bold: true, color: C.text, margin: 0,
  });
  slide.addText("Yₜ = Xₜ · [floor + (1−floor)Gₜ]", {
    x: 8.67, y: 2.99, w: 3.55, h: 0.3,
    fontSize: 14, color: C.muted, margin: 0,
  });
  panel(slide, 8.35, 4.06, 4.28, 2.57, C.panel2);
  label(slide, "Detection consequence", 8.68, 4.36, 2.2);
  bar(slide, 8.68, 4.8, 1.55, "Raw Direct", 49, 79, C.muted, "49 / 79");
  bar(slide, 8.68, 5.25, 1.55, "Derivative gate, floor .2", 4, 79, C.red, "4 / 79");
  bar(slide, 8.68, 5.7, 1.55, "Artifact-only causal", 58, 79, C.green, "58 / 79");
  slide.addText(
    "Slow calcium violates a “must be changing now” gate.",
    {
      x: 0.76, y: 6.79, w: 7.0, h: 0.3,
      fontSize: 14, bold: true, color: C.amber, margin: 0,
    }
  );
  slide.addNotes(
    "The formula is still useful as an explanatory support image, but the hard gate misses slow calcium. The successful causal lane is adaptive background/artifact handling while preserving amplitude."
  );
}

// 5 — Capacity matched causal result
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "High-impact result",
    "At equal review capacity, causal artifact suppression recovered 9 more labels",
    "This is the strongest real-time result in the current study."
  );
  panel(slide, 0.7, 1.78, 7.1, 4.95);
  label(slide, "Capacity-matched comparison", 1.05, 2.1, 2.8);
  bar(slide, 1.05, 2.72, 3.75, "Raw Direct", 49, 79, C.muted, "49 / 79");
  bar(slide, 1.05, 3.42, 3.75, "Causal artifact-only", 58, 79, C.green, "58 / 79");
  slide.addText("Both: 232 candidates", {
    x: 3.23, y: 4.05, w: 2.8, h: 0.28,
    fontSize: 12, bold: true, color: C.amber, margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 1.05, y: 4.54, w: 6.05, h: 0,
    line: { color: C.slate, width: 1 },
  });
  metric(slide, 1.05, 4.88, "+0.1139", "pooled recall difference", C.green, 2.1);
  metric(slide, 3.52, 4.88, "98.03%", "bootstrap mass > 0", C.green, 2.1);
  metric(slide, 5.88, 4.88, "11 / 2", "gains / losses", C.text, 1.5);

  panel(slide, 8.1, 1.78, 4.55, 4.95, C.panel2);
  label(slide, "Real-time envelope", 8.48, 2.1, 2.2);
  metric(slide, 8.48, 2.58, "2.76 ms", "preprocessing median", C.cyan, 2.0);
  metric(slide, 10.65, 2.58, "3.17 ms", "preprocessing p95", C.cyan, 1.7);
  metric(slide, 8.48, 3.75, "5.96 ms", "+ pooling/NMS median", C.text, 2.0);
  metric(slide, 10.65, 3.75, "6.41 ms", "+ pooling/NMS p95", C.text, 1.7);
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 8.48, y: 5.22, w: 3.58, h: 0.78,
    fill: { color: "0B3B37" }, line: { color: C.green, width: 1.2 },
  });
  slide.addText("Comfortably below the 20 ms frame budget", {
    x: 8.74, y: 5.48, w: 3.02, h: 0.25,
    fontSize: 12.5, bold: true, color: C.green, align: "center", margin: 0,
  });
  slide.addText("Bootstrap 95% CI: +0.0119 to +0.2254", {
    x: 0.78, y: 6.84, w: 4.0, h: 0.24,
    fontSize: 10.5, color: C.muted, margin: 0,
  });
  slide.addNotes(
    "Emphasize that capacity matching avoids winning by simply proposing more candidates. Timing includes the full candidate-producing path."
  );
}

// 6 — ICA
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Professor-proposed ICA",
    "For adjacent frames, ICA independently rediscovers the temporal derivative",
    "The component is interpretable—but still inherits the derivative’s slow-signal limitation."
  );
  addVideo(slide, "ica_activity", 0.7, 1.76, 7.2, 4.79);
  panel(slide, 8.18, 1.76, 4.48, 2.32);
  label(slide, "Two-frame geometry", 8.5, 2.05, 2.1);
  slide.addText("persistent direction", {
    x: 8.5, y: 2.48, w: 1.72, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("[ 1,  1 ]", {
    x: 10.31, y: 2.43, w: 1.25, h: 0.3, fontSize: 17, bold: true, color: C.text, margin: 0,
  });
  slide.addText("background-null", {
    x: 8.5, y: 2.94, w: 1.72, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("[−1,  1 ]", {
    x: 10.31, y: 2.89, w: 1.25, h: 0.3, fontSize: 17, bold: true, color: C.cyan, margin: 0,
  });
  slide.addText("= temporal derivative", {
    x: 9.22, y: 3.45, w: 2.35, h: 0.25,
    fontSize: 12, bold: true, color: C.cyan, align: "center", margin: 0,
  });
  panel(slide, 8.18, 4.33, 4.48, 2.22, C.panel2);
  label(slide, "Recovered components", 8.5, 4.62, 2.3);
  slide.addText("InfoMax", {
    x: 8.5, y: 5.04, w: 0.9, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("[−0.699, 0.715]  ·  cosine 0.99993", {
    x: 9.4, y: 5.0, w: 2.8, h: 0.28, fontSize: 13, bold: true, color: C.text, margin: 0,
  });
  slide.addText("CS-Parzen", {
    x: 8.5, y: 5.5, w: 0.9, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("[−0.707, 0.707]  ·  cosine ≈ 1", {
    x: 9.4, y: 5.46, w: 2.8, h: 0.28, fontSize: 13, bold: true, color: C.text, margin: 0,
  });
  slide.addText("Standalone InfoMax: 30 / 79 labels, 161 candidates", {
    x: 0.76, y: 6.77, w: 5.2, h: 0.28,
    fontSize: 13, bold: true, color: C.amber, margin: 0,
  });
  slide.addNotes(
    "The key scientific point is that this is not a separate mystery feature: in the two-frame setup, the ICA activity component is almost exactly the derivative direction."
  );
}

// 7 — Fusion
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Feature fusion",
    "Adding ICA to raw amplitude preserved the baseline—but did not improve it",
    "Soft fusion was safe; aggressive multiplicative gating again removed useful slow activity."
  );
  addVideo(slide, "raw_plus_ica", 0.7, 1.74, 7.35, 4.89);
  panel(slide, 8.35, 1.74, 4.28, 4.89);
  label(slide, "Fusion families", 8.68, 2.05, 2.0);
  slide.addText("Additive support", {
    x: 8.68, y: 2.48, w: 1.65, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("S = R + λ · quiet_scale(R) · F", {
    x: 8.68, y: 2.79, w: 3.45, h: 0.3,
    fontSize: 14, bold: true, color: C.text, margin: 0,
  });
  slide.addText("Soft multiplicative gate", {
    x: 8.68, y: 3.32, w: 2.0, h: 0.23, fontSize: 10, color: C.muted, margin: 0,
  });
  slide.addText("Y = X · [floor + (1−floor)F]", {
    x: 8.68, y: 3.63, w: 3.45, h: 0.3,
    fontSize: 14, bold: true, color: C.text, margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 8.68, y: 4.19, w: 3.25, h: 0,
    line: { color: C.slate, width: 1 },
  });
  bar(slide, 8.68, 4.5, 1.45, "Raw Direct", 49, 79, C.muted, "49 / 79");
  bar(slide, 8.68, 4.95, 1.45, "12 additive fusions", 49, 79, C.cyan, "49 / 79");
  bar(slide, 8.68, 5.4, 1.45, "InfoMax gate, floor .85", 47, 79, C.amber, "47 / 79");
  slide.addText("Learned λ = 0.036–0.046; still 49 / 79", {
    x: 8.68, y: 5.95, w: 3.25, h: 0.27,
    fontSize: 11.5, bold: true, color: C.text, margin: 0,
  });
  slide.addText(
    "Interpretation: ICA is useful for timing, visualization, and review prioritization—not yet as the detector’s decision surface.",
    {
      x: 0.76, y: 6.78, w: 7.4, h: 0.38,
      fontSize: 13.5, bold: true, color: C.amber, margin: 0,
    }
  );
  slide.addNotes(
    "The fixed and learned additive lanes were stable but flat. Treat ICA as an auxiliary channel for ranking, explanation, and later intent timing."
  );
}

// 8 — Latent smoothing
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Amplitude-preserving denoising",
    "The best recall came from smoothing amplitude—not differencing it",
    "The offline smoother is an upper bound; the causal filter is the deployable comparator."
  );
  addVideo(slide, "latent_smoother", 0.7, 1.74, 7.35, 4.89);
  panel(slide, 8.35, 1.74, 4.28, 4.89);
  label(slide, "Held-out recall", 8.68, 2.05, 1.7);
  bar(slide, 8.68, 2.54, 1.48, "Raw Direct", 49, 79, C.muted, "49 / 79");
  bar(slide, 8.68, 3.06, 1.48, "Causal filter amplitude", 53, 79, C.cyan, "53 / 79");
  bar(slide, 8.68, 3.58, 1.48, "Offline smoother amplitude", 55, 79, C.green, "55 / 79");
  slide.addShape(pptx.ShapeType.line, {
    x: 8.68, y: 4.18, w: 3.25, h: 0,
    line: { color: C.slate, width: 1 },
  });
  metric(slide, 8.68, 4.47, "0.6867", "offline macro recall", C.green, 1.65);
  metric(slide, 10.58, 4.47, "+37.9%", "candidate burden", C.amber, 1.45);
  slide.addText("Selected τ = 1280 ms", {
    x: 8.68, y: 5.62, w: 1.85, h: 0.25,
    fontSize: 11.5, bold: true, color: C.text, margin: 0,
  });
  slide.addText("top grid boundary", {
    x: 10.55, y: 5.64, w: 1.35, h: 0.22,
    fontSize: 9.5, color: C.amber, margin: 0,
  });
  slide.addText("Post-denoising differencing performed poorly.", {
    x: 8.68, y: 6.07, w: 3.2, h: 0.25,
    fontSize: 11.5, bold: true, color: C.red, margin: 0,
  });
  slide.addText(
    "The winning smoother uses future frames and is not eligible for real-time control.",
    {
      x: 0.76, y: 6.79, w: 6.6, h: 0.32,
      fontSize: 13.5, bold: true, color: C.amber, margin: 0,
    }
  );
  slide.addNotes(
    "This result strengthens the amplitude-preservation hypothesis. Do not present the offline smoother as deployable; it is a useful upper bound."
  );
}

// 9 — Synthesis
{
  const slide = pptx.addSlide("MASTER");
  addTitle(
    slide,
    "Decision and next evidence",
    "Build the detector around preserved amplitude; route change features beside it",
    "The next bottleneck is label quality and hard-negative review—not another broad blind sweep."
  );
  panel(slide, 0.7, 1.8, 3.82, 4.92);
  label(slide, "Primary channel", 1.03, 2.12, 1.65, C.green);
  slide.addText("Amplitude", {
    x: 1.03, y: 2.55, w: 2.3, h: 0.44,
    fontSize: 25, bold: true, color: C.text, margin: 0,
  });
  bullet(slide, "Raw or causally denoised fluorescence", 1.03, 3.26, 2.9, C.text, 12.5);
  bullet(slide, "Adaptive background and artifact suppression", 1.03, 4.1, 2.9, C.text, 12.5);
  bullet(slide, "Candidate production and recall optimization", 1.03, 5.05, 2.9, C.text, 12.5);

  panel(slide, 4.75, 1.8, 3.82, 4.92, C.panel2);
  label(slide, "Auxiliary channel", 5.08, 2.12, 1.75, C.cyan);
  slide.addText("Temporal change", {
    x: 5.08, y: 2.55, w: 2.8, h: 0.44,
    fontSize: 25, bold: true, color: C.text, margin: 0,
  });
  bullet(slide, "Derivative / pairwise ICA activity", 5.08, 3.26, 2.9, C.text, 12.5);
  bullet(slide, "Timing, propagation, and review ranking", 5.08, 4.1, 2.9, C.text, 12.5);
  bullet(slide, "Later: left/right intent dynamics", 5.08, 5.05, 2.9, C.text, 12.5);

  panel(slide, 8.8, 1.8, 3.82, 4.92);
  label(slide, "Next experiment", 9.13, 2.12, 1.6, C.amber);
  slide.addText("Audit → label → confirm", {
    x: 9.13, y: 2.55, w: 2.9, h: 0.44,
    fontSize: 22, bold: true, color: C.text, margin: 0,
  });
  slide.addText("1", {
    x: 9.13, y: 3.3, w: 0.28, h: 0.32, fontSize: 18, bold: true, color: C.cyan, margin: 0,
  });
  slide.addText("Review the 206 queued candidates; stratify gains, losses, and obvious artifacts.", {
    x: 9.52, y: 3.25, w: 2.55, h: 0.7, fontSize: 11.5, color: C.text, margin: 0,
  });
  slide.addText("2", {
    x: 9.13, y: 4.35, w: 0.28, h: 0.32, fontSize: 18, bold: true, color: C.cyan, margin: 0,
  });
  slide.addText("Add explicit hard negatives and distinguish known matches from accepted new positives.", {
    x: 9.52, y: 4.3, w: 2.55, h: 0.7, fontSize: 11.5, color: C.text, margin: 0,
  });
  slide.addText("3", {
    x: 9.13, y: 5.42, w: 0.28, h: 0.32, fontSize: 18, bold: true, color: C.cyan, margin: 0,
  });
  slide.addText("Confirm the causal amplitude-preserving lane across bursts and then across fish.", {
    x: 9.52, y: 5.37, w: 2.55, h: 0.7, fontSize: 11.5, color: C.text, margin: 0,
  });
  slide.addText(
    "Meeting question: which novel candidates are biologically credible enough to expand the ground truth?",
    {
      x: 0.78, y: 6.87, w: 10.8, h: 0.3,
      fontSize: 14, bold: true, color: C.amber, margin: 0,
    }
  );
  slide.addNotes(
    "Close by asking for concrete annotation decisions. The 206-row queue turns the discussion into the next dataset improvement."
  );
}

require("./spon_ca_burst_meeting_appendix")({
  pptx, pptxgen, C, addTitle, panel, label,
});

pptx.writeFile({ fileName: outPath });
console.log(`Wrote ${outPath}`);
