# Event labeler

The labeler gets people to label calcium events quickly. A detector ranks candidate events. People review only
those candidates, as short clips, answering **Accept** or **Decline**. That replaces scanning every pixel of every
frame. Every answer is saved as it is given, together with a record of how the item was reviewed.

The tool is self-contained. It runs from this folder with numpy, scipy and tifffile, and does not need the rest of
`neurobench` or its heavy dependencies.

## Two roles

| Role | Does | Needs |
|---|---|---|
| **Pack builder** | Runs the detector on a recording and writes a *pack* (clips) plus a *key* (which item is which). Gives out only the pack. | This repository, Python 3.9+, `requirements.txt` |
| **Labeller** | Opens the pack in a browser and decides each item. Sends back one log file. | Python 3.8+ only |

**The builder must not be a labeller for the same pack.** The key reveals which items are control patches and
which are hidden reference annotations, and the test only works if labellers do not know.

## Install (pack builder)

```
git clone https://github.com/Jibby2k1/NeuRev-Workbench.git
cd NeuRev-Workbench
python3 -m venv .venv-labeler
source .venv-labeler/bin/activate            # Windows: .venv-labeler\Scripts\activate
pip install -r neurobench/tools/event_labeler/requirements.txt
```

On Ubuntu/Debian, if `python3 -m venv` fails with "ensurepip is not available", run
`sudo apt install python3-venv` first, or create the environment with [uv](https://docs.astral.sh/uv/):
`uv venv .venv-labeler && uv pip install -r neurobench/tools/event_labeler/requirements.txt`.

Check the install (optional, needs `pip install pytest`): `python -m pytest tests/test_event_labeler.py`.

## Walkthrough: Spon Ca Burst

The recording and its reference annotations are in the lab's shared `NeuRev-Workbench` folder:

- `Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif`: 2359 frames, 340 × 573 px, 50 Hz
- `Inputs/Spon Ca Burst/labels/labels_normalized.tsv`: the reference annotations, 79 cells in 4 bursts

1. **Build the pack** (about 2 minutes):
   ```
   python neurobench/tools/event_labeler/build_pack.py --preset spon \
       --tif "PATH/TO/Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif" \
       --annotations "PATH/TO/Inputs/Spon Ca Burst/labels/labels_normalized.tsv" \
       --out packs/spon
   ```
   This writes `packs/spon/` (the pack, about 75 MB) and `packs/spon_KEY_DO_NOT_SHARE/` (the key). With these
   settings the pack holds 699 items:
   - **600 detector proposals**, most relevant first.
   - **67 random control patches** at random queue positions.
   - **32 hidden reference annotations**: the annotations the detector did not already propose, placed at random
     positions among the first 300 items.

   The other 47 annotations are already covered by a proposal, within 6 px inside their burst window. The key
   records which proposal covers each one.
2. **Give out the pack.** Zip `packs/spon/` and send it, or put it in the shared folder. Do not include the key.
3. **Labellers** follow `LABELLING.md`, which is inside the pack: `python3 label_server.py . --annotator NAME`.
   Each one sends back `labels_<pack id>_<NAME>.jsonl`.
4. **Summarize** with the key:
   ```
   python neurobench/tools/event_labeler/summarize.py --key packs/spon_KEY_DO_NOT_SHARE/key.json labels_*.jsonl
   ```
   The summary reports, per labeller:
   - accept rates for proposals, control patches and hidden annotations;
   - how many reference annotations were accepted again;
   - seconds per decision;
   - how the proposal accept rate falls with queue position.

## Any other recording

```
python neurobench/tools/event_labeler/build_pack.py --tif MOVIE.tif --out packs/movie [--annotations my.tsv]
```

Defaults for a recording without a preset:

- **Reviewed area:** the whole frame (`--region Y X H W` to restrict it).
- **Detector warm-up:** the first 200 frames (`--warmup`).
- **Noise calibration:** the photon-noise line and the Gamma-LS floor are fitted on the next 200 frames
  (`--setup FIRST LAST`). Pick frames without strong activity if you can, or give a known noise line with
  `--photon A B`.
- **Frame rate:** 50 Hz (`--fps`).

Other useful options:

- `--max-proposals` sets the number of proposals.
- `--random-frac` sets the share of control patches.
- `--half`, `--pre` and `--post` set the clip size and length.

**Annotation file:** tab-separated, with a header row and columns `x_px y_px start_frame_ui end_frame_ui`, plus
optionally `roi_identity` and `burst_id`.

- Coordinates are full-frame pixels.
- Frames are 1-based and give the window in which the cell is active.
- Each hidden annotation is shown at the frame in its window where the centre disk is brightest.

The input must be a single T × H × W TIFF stack. Movies in other formats must be converted first.

## What a pack contains

| File | Content |
|---|---|
| `manifest.json` | Item positions (frame, y, x) in queue order, clip geometry and display ranges. It holds no item kinds and no scores. |
| `raw.u16`, `ls.i16` | Clips: raw counts, and Gamma-LS z × 100. |
| `context.u8` | The mean image used for the location map. |
| `label_server.py`, `labeler.html`, `LABELLING.md` | The review app and the labeller guide. |

The key (`key.json`) holds each item's kind and detector score, every annotation's outcome (covered, hidden, or
skipped), and the build settings.

## What the log records

Each line of `labels_<pack id>_<name>.jsonl` is one JSON record. Decision lines hold the answer plus a `process`
record:

- **Timing:** active, paused and hidden time on the item, and time to the first interaction.
- **Viewing:** time spent on each clip frame, loops played, frame steps and play/stop presses.
- **Display changes:** every ring, contrast and speed change, with its time, and the display at the start and at
  the decision.
- **Navigation:** keyboard or mouse, and whether the item was revisited and what the earlier answer was.

Session start, pause, resume, navigation and stop lines carry the timeline. Session start also records the
browser, the screen and the tool's file hashes. Logs are append-only and tied to one pack: a log cannot be resumed
against a different pack.

## The detector

`operators.py` holds standalone ports of the NeuRev three-part prescreener operators:

- **P1-phys (arm A5):** Gaussian blur σ 3.24 px, fast EMA 0.4 minus lagged EMA 0.141, divided by the photon-noise
  SD.
- **Gamma-LS (arm A1):** a sigma-1 blur, the one-frame change of EMA 0.4 levels, standardized by a Gamma ring
  (guard 7, half-width 15).
- **Candidate extractor:** 13 × 13 local maxima with greater than 6 px separation.

Parity against the original implementation, on all 2359 Spon frames:

- **P1-phys scores:** bit-identical.
- **Candidate rows:** all 397,775 identical.
- **Gamma-LS z:** at most 1.2 × 10⁻¹³ apart.
- **Gamma-LS floor:** identical.

The Spon preset uses the sealed noise line `3.578 · mean − 303.4`. The automatic fit gives about 3.73 · mean − 408
when it is run on that recording's whole frame.

Recordings and packs are lab data. Keep them out of this repository.
