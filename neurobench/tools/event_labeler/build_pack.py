#!/usr/bin/env python3
"""Build a review pack for the event labeler from any calcium-imaging TIFF stack (numpy, scipy, tifffile).

    python build_pack.py --tif MOVIE.tif --out packs/my_pack [--preset spon] [--annotations ANNOTATIONS.tsv]

The pack is a folder of short clips (raw + Gamma-LS "local standardized") around
  * proposals    events of the P1-phys detector (operators.P1Phys), one item per event at its peak, ranked by
                 peak score (most relevant first);
  * random       control patches at random places in the same periods as the proposals, away from every detector
                 event and every annotation (default 10% of items), placed at random positions in the queue;
  * annotation   (only with --annotations) reference annotations that are NOT already among the proposals, hidden
                 at random positions among the top-ranked items (--annotation-window).
Nothing in the pack says which item is which. That key (and every score) goes to the key folder, which must never
be given to the people labelling: whoever labels must not be the person who builds the pack.

Defaults for a new dataset: the whole frame is reviewed, the first 200 frames warm the detector up, and the
photon-noise line and the Gamma-LS floor are fitted on the next 200 frames. --preset spon sets the values used for
the Spon Ca Burst recording ("3 hindbrain to tail 488 20ms.tif").

Annotations TSV (tab-separated, header row): x_px, y_px (full-frame pixels), start_frame_ui, end_frame_ui
(1-based frames of the window in which the cell is active); optional roi_identity, burst_id. This is the format of
the Spon labels_normalized.tsv. Each hidden annotation's clip is centred on the frame of its window where the mean
of the centre disk (radius 3 px) is brightest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operators as O  # noqa: E402

PACK_FORMAT = 2
PRESETS = {
    "spon": dict(region=(49, 49, 242, 475), photon=(3.578, -303.4), warmup=200, setup=(1600, 1799),
                 note="Spon Ca Burst, 3 hindbrain to tail 488 20ms.tif (340 x 573, 2359 frames, 50 Hz)"),
}


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def open_tif(path):
    import tifffile
    try:
        return tifffile.memmap(str(path), mode="r")
    except ValueError:                      # compressed / non-contiguous TIFF: read into memory
        return tifffile.imread(str(path))


def read_annotations(path, H, W):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for n, r in enumerate(csv.DictReader(f, delimiter="\t"), 1):
            try:
                a = dict(x=float(r["x_px"]), y=float(r["y_px"]), start=int(float(r["start_frame_ui"])),
                         end=int(float(r["end_frame_ui"])))
            except (KeyError, ValueError, TypeError) as e:
                raise SystemExit(f"{path} row {n}: needs x_px, y_px, start_frame_ui, end_frame_ui ({e})")
            a["id"] = str(r.get("roi_identity") or f"row{n}")
            if r.get("burst_id"):
                a["id"] = f"b{int(float(r['burst_id'])):02d}__{a['id']}"
            if not (0 <= a["x"] < W and 0 <= a["y"] < H) or a["end"] < a["start"]:
                raise SystemExit(f"{path} row {n}: outside the frame or end < start")
            rows.append(a)
    return rows


def chain_events(rows, z_min, link_px=6.0, link_frames=2):
    """Single linkage over candidate rows (t, y, x, score, sat) with score >= z_min (<= 6 px, <= 2 frames); one
    event per component at its highest-score row (ties: earliest)."""
    on = sorted((r for r in rows if r[3] >= z_min), key=lambda r: r[0])
    parent = list(range(len(on)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_t = {}
    for k, r in enumerate(on):
        by_t.setdefault(r[0], []).append(k)
    for k, (t, y, x, _s, _sat) in enumerate(on):
        for dt in range(0, link_frames + 1):
            for j in by_t.get(t - dt, ()):
                if dt == 0 and j >= k:
                    continue
                if (on[j][1] - y) ** 2 + (on[j][2] - x) ** 2 <= link_px ** 2:
                    a, b = find(j), find(k)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
    comps = {}
    for k in range(len(on)):
        comps.setdefault(find(k), []).append(k)
    events = []
    for m in comps.values():
        best = min(m, key=lambda k: (-on[k][3], on[k][0]))
        t, y, x, s, sat = on[best]
        events.append(dict(t=t, y=y, x=x, score=s, sat=sat, first=min(on[k][0] for k in m),
                           last=max(on[k][0] for k in m), rows=len(m)))
    events.sort(key=lambda e: -e["score"])
    return events


def disk_mean(fr, y, x, r=3):
    H, W = fr.shape
    ys, ye, xs, xe = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
    yy, xx = np.mgrid[ys:ye, xs:xe]
    m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return float(np.asarray(fr[ys:ye, xs:xe], dtype=np.float64)[m].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", required=True, help="the recording (T x H x W TIFF stack)")
    ap.add_argument("--out", required=True, help="pack folder to create (this is what labellers receive)")
    ap.add_argument("--key-out", default=None, help="key folder (default: <out>_KEY_DO_NOT_SHARE next to the pack)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--first-ui", type=int, default=1, help="frame number of the TIFF's first frame")
    ap.add_argument("--region", type=int, nargs=4, metavar=("Y", "X", "H", "W"), default=None,
                    help="reviewed region (default: the whole frame)")
    ap.add_argument("--photon", type=float, nargs=2, metavar=("A", "B"), default=None,
                    help="noise line var = A * mean + B (default: fitted on the setup frames)")
    ap.add_argument("--warmup", type=int, default=None, help="detector warm-up frames (default 200)")
    ap.add_argument("--setup", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
                    help="frames used for the noise line and the Gamma-LS floor (default: the 200 after warm-up)")
    ap.add_argument("--annotations", default=None, help="reference annotations TSV to hide among the proposals")
    ap.add_argument("--annotation-window", type=int, default=300,
                    help="hidden annotations go at random positions within the first N queue positions")
    ap.add_argument("--z-min", type=float, default=4.0, help="detector event threshold (P1-phys z)")
    ap.add_argument("--max-proposals", type=int, default=600)
    ap.add_argument("--random-frac", type=float, default=0.10)
    ap.add_argument("--half", type=int, default=12, help="clip half-width in px (clip = 2*half+1 square)")
    ap.add_argument("--pre", type=int, default=15, help="clip frames before the item frame")
    ap.add_argument("--post", type=int, default=25, help="clip frames after the item frame")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--dedupe-px", type=float, default=10.0)
    ap.add_argument("--dedupe-frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip, ready to send to reviewers")
    a = ap.parse_args(argv)

    pre = PRESETS.get(a.preset, {})
    out = Path(a.out).resolve()
    key_dir = Path(a.key_out).resolve() if a.key_out else out.parent / (out.name + "_KEY_DO_NOT_SHARE")
    if key_dir == out or out in key_dir.parents:
        raise SystemExit("the key folder must not be inside the pack folder")
    out.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    logf = open(key_dir / "build.log", "a", encoding="utf-8")

    def log(s):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    stack = open_tif(a.tif)
    if stack.ndim != 3:
        raise SystemExit(f"expected a T x H x W stack, got shape {stack.shape}")
    T, H, W = stack.shape
    ui0 = a.first_ui
    log(f"{os.path.basename(a.tif)}: {T} frames {H}x{W} {stack.dtype}")

    y0, x0, h, w = a.region or pre.get("region") or (0, 0, H, W)
    if y0 < 0 or x0 < 0 or y0 + h > H or x0 + w > W:
        raise SystemExit("region outside the frame")
    b = O.NMS_DISTANCE_PX
    warm = a.warmup if a.warmup is not None else pre.get("warmup", 200)
    s_first, s_last = a.setup or pre.get("setup") or (ui0 + warm, ui0 + warm + 199)
    if not (ui0 <= s_first < s_last <= ui0 + T - 1):
        raise SystemExit(f"setup frames {s_first}-{s_last} are outside the recording; set --setup / --warmup")
    region_mask = np.zeros((H, W), bool)
    region_mask[y0:y0 + h, x0:x0 + w] = True

    # ---- pass 0: noise line and Gamma-LS floor on the setup frames (label-free)
    if a.photon or pre.get("photon"):
        pa, pb = a.photon or pre["photon"]
        photon_rec = dict(a=pa, b=pb, source="command line" if a.photon else f"preset {a.preset}")
    else:
        pa, pb, photon_rec = O.photon_line((stack[i - ui0] for i in range(s_first, s_last + 1)), region_mask)
        photon_rec["source"] = f"fitted on frames {s_first}-{s_last}"
    log(f"noise line var = {pa:.4g} * mean + {pb:.4g} ({photon_rec['source']})")
    gl = O.GammaLS()
    spread = []
    for i in range(0, s_last - ui0 + 1):
        _D, _M, V = gl.step(stack[i])
        if i + ui0 >= s_first:
            spread.append(np.sqrt(V[y0:y0 + h, x0:x0 + w]).ravel())
    sp = np.concatenate(spread)
    floor = float(np.percentile(sp[sp > 0], 10))
    log(f"Gamma-LS floor {floor:.6g} (p10 of the positive setup spread)")

    ann = read_annotations(a.annotations, H, W) if a.annotations else []
    if ann:
        log(f"{len(ann)} reference annotations read")

    # ---- pass 1: P1-phys candidates over the whole recording; annotation centre-disk traces
    p1 = O.P1Phys(pa, pb)
    first_ui = ui0 + warm
    rows = []
    ann_trace = [dict() for _ in ann]
    t0 = time.time()
    for i in range(T):
        ui = i + ui0
        fr = np.asarray(stack[i])
        z = p1.step(fr)
        for k, an in enumerate(ann):
            if an["start"] <= ui <= an["end"]:
                ann_trace[k][ui] = disk_mean(fr, int(round(an["y"])), int(round(an["x"])))
        if ui < first_ui:
            continue
        sat = O.sat_mask(fr)
        for s, yy, xx in O.frame_candidates(z[y0:y0 + h, x0:x0 + w]):
            rows.append((ui, yy + y0, xx + x0, s, bool(sat[yy + y0, xx + x0]) if sat is not None else False))
        if i % 500 == 0:
            log(f"detector frame {i}/{T} ({time.time() - t0:.0f}s)")
    events = chain_events(rows, a.z_min)
    log(f"{len(rows)} candidate rows, {len(events)} events at z >= {a.z_min}")

    lo_ui, hi_ui = max(first_ui, ui0 + a.pre), ui0 + T - 1 - a.post
    if hi_ui < lo_ui:
        raise SystemExit("recording too short for the clip length after warm-up")

    def fits(y, x, t):
        return a.half <= y < H - a.half and a.half <= x < W - a.half and lo_ui <= t <= hi_ui

    chosen = []
    for e in events:
        if not fits(e["y"], e["x"], e["t"]):
            continue
        if any(abs(e["t"] - c["t"]) <= a.dedupe_frames and
               (e["y"] - c["y"]) ** 2 + (e["x"] - c["x"]) ** 2 <= a.dedupe_px ** 2 for c in chosen):
            continue
        chosen.append(dict(t=e["t"], y=e["y"], x=e["x"], kind="proposal", score=e["score"], sat=e["sat"],
                           first=e["first"], last=e["last"]))
        if len(chosen) >= a.max_proposals:
            break
    log(f"{len(chosen)} proposals")

    # ---- reference annotations: covered by a proposal (<= 6 px, inside its window) or hidden as their own item
    ann_items, ann_rec = [], []
    for k, an in enumerate(ann):
        cover = [c for c in chosen if an["start"] <= c["t"] <= an["end"]
                 and (c["y"] - an["y"]) ** 2 + (c["x"] - an["x"]) ** 2 <= 36]
        rec = dict(annotation_id=an["id"], x=an["x"], y=an["y"], window=[an["start"], an["end"]])
        if cover:
            best = max(cover, key=lambda c: c["score"])
            best.setdefault("covers_annotations", []).append(an["id"])
            rec.update(status="covered_by_proposal", item=f"t{best['t']}:y{best['y']}:x{best['x']}")
        elif not ann_trace[k]:
            rec.update(status="skipped_window_outside_recording")
        else:
            tr = ann_trace[k]
            t = min(max(max(tr, key=lambda u: (tr[u], -u)), lo_ui), hi_ui)
            yy, xx = int(round(an["y"])), int(round(an["x"]))
            if not fits(yy, xx, t):
                rec.update(status="skipped_too_close_to_edge")
            elif any(it["t"] == t and it["y"] == yy and it["x"] == xx for it in ann_items):
                rec.update(status="duplicate_of_another_annotation")
            else:
                ann_items.append(dict(t=t, y=yy, x=xx, kind="annotation", score=None, annotation_id=an["id"]))
                rec.update(status="hidden_item", item=f"t{t}:y{yy}:x{xx}",
                           frame_rule="brightest centre-disk (r 3 px) mean in the window")
        ann_rec.append(rec)
    if ann:
        n_cov = sum(r["status"] == "covered_by_proposal" for r in ann_rec)
        log(f"annotations: {n_cov} already among the proposals, {len(ann_items)} hidden as items, "
            f"{len(ann) - n_cov - len(ann_items)} skipped")

    # ---- random control patches: same periods as the proposals, away from events and annotations
    rng = np.random.default_rng(a.seed)
    n_rand = int(round(a.random_frac * len(chosen) / max(1e-9, 1 - a.random_frac)))
    prop_t = np.array([c["t"] for c in chosen]) if chosen else np.arange(lo_ui, hi_ui + 1)
    avoid = np.array([(e["t"], e["y"], e["x"]) for e in events], dtype=np.float64).reshape(-1, 3)
    ann_sites = [(an["start"], an["end"], an["y"], an["x"]) for an in ann]
    rand, tries = [], 0
    while len(rand) < n_rand and tries < 200000:
        tries += 1
        y = int(rng.integers(y0 + b, y0 + h - b))
        x = int(rng.integers(x0 + b, x0 + w - b))
        t = int(np.clip(prop_t[rng.integers(len(prop_t))] + rng.integers(-a.dedupe_frames, a.dedupe_frames + 1),
                        lo_ui, hi_ui))
        if not fits(y, x, t):
            continue
        if len(avoid) and ((np.abs(avoid[:, 0] - t) <= a.dedupe_frames) &
                           ((avoid[:, 1] - y) ** 2 + (avoid[:, 2] - x) ** 2 <= a.dedupe_px ** 2)).any():
            continue
        if any(s - a.dedupe_frames <= t <= e + a.dedupe_frames and (ay - y) ** 2 + (ax - x) ** 2 <= a.dedupe_px ** 2
               for s, e, ay, ax in ann_sites):
            continue
        rand.append(dict(t=t, y=y, x=x, kind="random", score=None))
    log(f"{len(rand)} random control patches")

    # ---- queue order: proposals by score; annotations at random within the top window; random patches anywhere
    n_tot = len(chosen) + len(rand) + len(ann_items)
    slots = [None] * n_tot
    win = min(n_tot, max(a.annotation_window, len(ann_items)))
    for pos, it in zip(rng.choice(win, size=len(ann_items), replace=False), ann_items):
        slots[int(pos)] = it
    free = [k for k in range(n_tot) if slots[k] is None]
    for pos, it in zip(rng.choice(len(free), size=len(rand), replace=False), rand):
        slots[free[int(pos)]] = it
    props = iter(chosen)
    items = [s if s is not None else next(props) for s in slots]
    for it in items:
        it["id"] = f"t{it['t']}:y{it['y']}:x{it['x']}"
    if len({it["id"] for it in items}) != len(items):
        raise SystemExit("two items share a position; rerun with another --seed")

    # ---- pass 2: clips (raw + Gamma-LS z)
    S, F, N = 2 * a.half + 1, a.pre + a.post + 1, len(items)
    raw = np.zeros((N, F, S, S), dtype="<u2")
    lsz = np.zeros((N, F, S, S), dtype="<i2")
    starts = {}
    for n, it in enumerate(items):
        starts.setdefault(it["t"] - a.pre, []).append(n)
    gl = O.GammaLS(floor)
    active = []
    t0 = time.time()
    for i in range(T):
        ui = i + ui0
        fr = np.asarray(stack[i])
        z = gl.z(*gl.step(fr))
        active += [(n, ui) for n in starts.get(ui, ())]
        keep = []
        for n, s_ui in active:
            k = ui - s_ui
            it = items[n]
            ys, xs = slice(it["y"] - a.half, it["y"] + a.half + 1), slice(it["x"] - a.half, it["x"] + a.half + 1)
            raw[n, k] = np.clip(fr[ys, xs], 0, 65535)
            lsz[n, k] = np.clip(np.rint(z[ys, xs] * 100.0), -32000, 32000)
            if k < F - 1:
                keep.append((n, s_ui))
        active = keep
        if i % 500 == 0:
            log(f"clips frame {i}/{T} ({time.time() - t0:.0f}s)")

    samp = np.stack([np.asarray(stack[i])[y0:y0 + h, x0:x0 + w] for i in range(0, T, max(1, T // 200))])
    raw_lo, raw_hi = (float(v) for v in np.percentile(samp, [0.5, 99.8]))
    ctx = np.mean(np.stack([np.asarray(stack[i], np.float64) for i in range(0, T, max(1, T // 100))]), axis=0)
    c_lo, c_hi = np.percentile(ctx, [1, 99.7])
    ctx_u8 = np.clip((ctx - c_lo) / max(c_hi - c_lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    raw.tofile(out / "raw.u16")
    lsz.tofile(out / "ls.i16")
    ctx_u8.tofile(out / "context.u8")

    manifest = dict(
        pack_format=PACK_FORMAT,
        source=dict(file=os.path.basename(a.tif), sha256=sha256_file(a.tif), first_ui=ui0, frames=T, height=H,
                    width=W, fps=a.fps),
        clip=dict(size=S, half=a.half, pre=a.pre, post=a.post, frames=F),
        display=dict(raw_lo=raw_lo, raw_hi=raw_hi, ls_scale=100.0, ls_range=6.0,
                     center_r=max(1, min(3, a.half - 2)), annulus_r0=max(2, min(6, a.half - 1)),
                     annulus_r1=max(3, min(10, a.half))),
        files=dict(raw="raw.u16", ls="ls.i16", context="context.u8"),
        order="most relevant first (detector ranking); control items are mixed in",
        items=[dict(id=it["id"], t=it["t"], y=it["y"], x=it["x"]) for it in items],
    )
    manifest["file_sha256"] = {k: sha256_file(out / v) for k, v in manifest["files"].items()}
    manifest["pack_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ("label_server.py", "labeler.html", "LABELLING.md"):
        src = Path(__file__).resolve().parent / f
        (out / f).write_bytes(src.read_bytes())

    key = dict(pack_id=manifest["pack_id"], built=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               settings=dict(preset=a.preset, region=[y0, x0, h, w], warmup=warm, setup=[s_first, s_last],
                             z_min=a.z_min, max_proposals=a.max_proposals, random_frac=a.random_frac,
                             annotation_window=a.annotation_window, half=a.half, pre=a.pre, post=a.post,
                             dedupe=[a.dedupe_px, a.dedupe_frames], seed=a.seed),
               detector=dict(name="P1-phys (port of three-part arm A5)", photon=photon_rec),
               local_standardized=dict(name="Gamma-LS (port of arm A1)", floor=floor),
               annotations=dict(file=os.path.basename(a.annotations) if a.annotations else None,
                                sha256=sha256_file(a.annotations) if a.annotations else None, rows=ann_rec),
               counts=dict(proposals=len(chosen), random=len(rand), annotation=len(ann_items), events=len(events),
                           candidate_rows=len(rows)),
               items=[dict({k: it.get(k) for k in ("id", "kind", "score", "sat", "first", "last", "annotation_id",
                                                   "covers_annotations")}, position=n)
                      for n, it in enumerate(items)])
    (key_dir / "key.json").write_text(json.dumps(key, indent=1))
    log(f"pack {manifest['pack_id']}: {N} items -> {out}")
    if a.zip:
        import zipfile
        zpath = out.parent / (out.name + ".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(out.iterdir()):
                if f.is_file() and not f.name.startswith("labels_"):
                    z.write(f, f.name)
        log(f"zip -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB): send this to reviewers")
    log(f"key -> {key_dir}  (keep it away from labellers)")


if __name__ == "__main__":
    main()
