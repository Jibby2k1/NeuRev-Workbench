# Labelling guide

You have received a **pack folder**. It contains everything you need. Install only Python 3.8 or newer
(python.org); no other packages are needed.

## Start

Open a terminal in the pack folder and run:

```
python3 label_server.py . --annotator YOURNAME
```

On Windows, use `py label_server.py . --annotator YOURNAME`.

Your browser opens and asks **how much you want to review**:

- **This sitting:** a number of items (25, 50, 100, 200), a time (10, 20 or 30 min), or no limit. When you
  reach it, choose **Keep going** or **Finish for now**.
- **Out of the queue:** the items are ordered most relevant first. Choose to review only the top 100, 200 or
  300, or all of them. You can widen this later.

Then you start on the first item you have not decided yet. Use the same name every time.

## Decide each item

Each item is a short clip, 25 × 25 pixels and 0.8 s long, centred on one spot. Most items are candidates the
software found; a few are controls. Judge only what you see.

- **Video:** the raw recording, looping. The box on the small brain image shows where the clip is.
- **Local standardized:** the same clip, showing how much each pixel changes compared with the tissue around it.
  Orange means brighter than its surroundings, blue means darker.
- **Centre vs annulus:** the brightness (ΔF/F₀) inside the orange circle and in the ring between the dashed
  circles, over the clip. The dashed vertical line is the frame the item is centred on.

**Accept** when a cell at the centre lights up: the centre brightens, more than the ring around it, and then fades.
**Decline** anything else: noise, flicker, movement, or a brightening of the whole area.

| Key | Action |
|---|---|
| `A` | Accept |
| `D` | Decline |
| `B` | Back, to change an earlier answer (the latest answer counts) |
| `Space` | Play / stop the clip |
| `←` `→` | Step one frame |
| `C` | Switch contrast: the same scale for every clip, or one scale for this clip |
| `P` | Pause, or change this sitting's goal |

## Stop and continue

Every answer is saved the moment you give it. Close the window and press Ctrl+C in the terminal whenever you like.
Run the same command again to continue where you stopped.

## Send back

When you are done, or at the end of each session, send the file `labels_<pack id>_YOURNAME.jsonl` from the pack
folder to whoever gave you the pack. It holds your answers and a record of how you reviewed each item (timings,
frames viewed). It contains nothing else from your computer apart from the browser name and screen size.

Two people on one computer: give each a different port, e.g. `--port 8766`.
