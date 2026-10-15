# Labelling guide

You have received a **pack**. The easiest way to review it needs nothing installed:

1. Open <https://www.raulv.dev/neurev/review/> (or open `labeler.html` from the pack folder).
2. Enter your name, then choose the pack `.zip`.
3. When you are done, press **Download results** and send the zip back.

Nothing is uploaded: the pack is read on your computer and your answers stay in your browser.

## Start with the local server instead (Python 3.8+)

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
**Unsure** when you cannot tell; add a short note (`N`) whenever something is worth saying.

If you know why you are declining, you can say so in the same keystroke: press `1`–`6` instead of `D`
(or click the matching button under the answers). This is optional; `D` alone is fine.

| Key | Decline because… |
|---|---|
| `1` | Noise or flicker |
| `2` | Movement |
| `3` | The whole area brightens, not one cell |
| `4` | Saturated (the brightest pixels are clipped) |
| `5` | A cell is nearby, but not at the centre |
| `6` | Other: type a short note, then `Enter` |

| Key | Action |
|---|---|
| `A` | Accept |
| `D` | Decline |
| `U` | Unsure |
| `N` | Type a note for this item (`Esc` leaves the box) |
| `B` | Back, to change an earlier answer (the latest answer counts) |
| `Space` | Play / stop the clip |
| `←` `→` | Step one frame |
| `C` | Switch contrast: the same scale for every clip, or one scale for this clip |
| `P` | Pause, or change this sitting's goal |
| `L` | Link repeated cells (see below) |

## Link repeated cells (optional, after reviewing)

The software often finds the same cell more than once. Press `L` (or **Link cells**) at any time, usually when
you have finished: the tool shows pairs of clips **you accepted** whose centres are close together, side by side,
and asks whether they show the **same cell** (`S`) or **different cells** (`X`). `K` skips a pair you cannot judge,
`Esc` goes back to reviewing. Pairs already joined through your answers are not asked again.

## Stop and continue

Every answer is saved the moment you give it. Close the window and press Ctrl+C in the terminal whenever you like.
Run the same command again to continue where you stopped.

## Send back

In the browser, press **Download results** and send that zip. With the local server, send the file
`labels_<pack id>_YOURNAME.jsonl` from the pack folder to whoever gave you the pack. It holds your answers and a record of how you reviewed each item (timings,
frames viewed). It contains nothing else from your computer apart from the browser name and screen size.

Two people on one computer: give each a different port, e.g. `--port 8766`.
