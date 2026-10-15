#!/usr/bin/env python3
"""Event labeler: local review server (Python 3.8+, standard library only).

    python3 label_server.py PACK_DIR --annotator NAME [--port 8765] [--log-dir DIR]

Opens http://127.0.0.1:PORT in the browser. Every decision is appended to
    <log-dir>/labels_<pack_id>_<annotator>.jsonl      (one JSON object per line, flushed + fsynced)
so closing the browser, the terminal or the machine never loses a decision. Starting the server again with the
same pack and annotator resumes at the first item without a decision.

When the page opens the labeller picks a goal for the sitting (a number of items, a number of minutes, or no
limit) and how much of the queue to review (the top N items, or all). --budget, --minutes and --scope preselect
those choices. Reaching the goal offers to keep going or to finish; the scope can be widened later. Items are in the pack's order (most
relevant first). The latest decision for an item wins
(Back lets you change one). Session start / pause / resume / navigate / stop lines are logged too.

Each decision line carries a "process" record of how it was made: time on the item (active, paused, hidden), time
to first interaction, time spent on each clip frame (dwell_ms), loops played, frame steps, play toggles, ring /
contrast / speed changes with their times, start and end display state, keyboard or mouse, and whether the item was
revisited (and its earlier answer). session_start lines carry the browser/screen and the tool's file hashes.

A decline may carry a reason (keys 1-6 in the page): noise, motion, whole_area, saturated, off_centre, other.
"link" lines record the optional linking pass over pairs of accepted items close to each other: whether the two
are the same cell (same=true) or different cells (same=false). The latest answer for a pair wins.

Export the decisions as a table:
    python3 label_server.py PACK_DIR --annotator NAME --export labels.tsv
"""
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECISIONS = ("accept", "decline", "unsure")
DECLINE_REASONS = ("noise", "motion", "whole_area", "saturated", "off_centre", "other")
EVENTS = ("session_start", "pause", "resume", "session_stop", "navigate", "settings", "goal_reached", "reviewer",
          "link_session")
MAX_RECORD_BYTES = 64 * 1024


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


TOOL_VERSION = {"label_server.py": _sha(__file__), "labeler.html": _sha(HERE / "labeler.html")}


class Pack:
    def __init__(self, root):
        self.root = Path(root)
        self.m = json.loads((self.root / "manifest.json").read_text())
        if self.m.get("pack_format") not in (1, 2):
            raise SystemExit("unsupported pack format")
        c = self.m["clip"]
        self.S, self.F = c["size"], c["frames"]
        self.n = len(self.m["items"])
        self.item_bytes = self.F * self.S * self.S * 2
        for k in ("raw", "ls"):
            size = (self.root / self.m["files"][k]).stat().st_size
            if size != self.n * self.item_bytes:
                raise SystemExit(f"{k} file has {size} bytes, expected {self.n * self.item_bytes}")
        self.ids = [it["id"] for it in self.m["items"]]
        self.index = {i: k for k, i in enumerate(self.ids)}
        self._lock = threading.Lock()
        self._fh = {k: open(self.root / self.m["files"][k], "rb") for k in ("raw", "ls")}

    def clip(self, kind, k):
        with self._lock:
            f = self._fh[kind]
            f.seek(k * self.item_bytes)
            return f.read(self.item_bytes)

    def context(self):
        return (self.root / self.m["files"]["context"]).read_bytes()

    def public_manifest(self):
        m = dict(self.m)
        m.pop("file_sha256", None)
        return m


class Log:
    """Append-only JSONL. State = latest decision per item id; pack id checked on every line."""

    def __init__(self, path, pack_id, annotator):
        self.path, self.pack_id, self.annotator = Path(path), pack_id, annotator
        self.decisions = {}
        self.reasons = {}            # item id -> decline reason of its latest decision
        self.links = {}              # "id_a|id_b" (sorted) -> latest same-cell answer for that pair
        self.scope = None            # latest "review the top N items" choice (None = whole queue)
        self.notes = {}
        self.lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"warning: {self.path}:{n} is not valid JSON (an interrupted write?); skipped",
                              file=sys.stderr)
                        continue
                    if r.get("pack_id") != pack_id:
                        raise SystemExit(f"{self.path} line {n} belongs to pack {r.get('pack_id')}, "
                                         f"not {pack_id}: refusing to mix packs")
                    self._apply(r)
        self.f = open(self.path, "a", encoding="utf-8")

    def _apply(self, r):
        if r.get("type") == "decision":
            self.decisions[r["item_id"]] = r["decision"]
            if r.get("reason"):
                self.reasons[r["item_id"]] = r["reason"]
            else:
                self.reasons.pop(r["item_id"], None)
            self._note(r)
        elif r.get("type") == "link":
            self.links[pair_key(r["item_id"], r["other_id"])] = bool(r["same"])
        elif r.get("type") == "settings" and "scope" in r:
            self.scope = r["scope"]

    def _note(self, r):
        if r.get("note"):
            self.notes[r["item_id"]] = r["note"]
        else:
            self.notes.pop(r["item_id"], None)

    def write(self, rec):
        rec = dict(rec, pack_id=self.pack_id, annotator=self.annotator,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), unix=round(time.time(), 3))
        with self.lock:
            self.f.write(json.dumps(rec, sort_keys=True) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())
            self._apply(rec)
        return rec


def pair_key(a, b):
    return "|".join(sorted((a, b)))


def scope_limit(pack, log):
    s = log.scope
    return pack.n if not isinstance(s, int) or s <= 0 else min(s, pack.n)


def first_open(pack, log):
    """First undecided item within the chosen scope; the scope size when every item in it is decided."""
    lim = scope_limit(pack, log)
    for k, i in enumerate(pack.ids[:lim]):
        if i not in log.decisions:
            return k
    return lim


def make_handler(pack, log, defaults=None):
    defaults = dict(defaults or {})
    page = (HERE / "labeler.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if p == "/api/state":
                return self._json(dict(manifest=pack.public_manifest(), annotator=log.annotator,
                                       decisions=log.decisions, next=first_open(pack, log),
                                       notes=log.notes, reasons=log.reasons, links=log.links, scope=log.scope, defaults=defaults,
                                       log_file=str(log.path)))
            if p == "/api/context":
                return self._send(200, pack.context(), "application/octet-stream")
            parts = p.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "api" and parts[1] in ("raw", "ls") and parts[2].isdigit():
                k = int(parts[2])
                if 0 <= k < pack.n:
                    return self._send(200, pack.clip(parts[1], k), "application/octet-stream")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/log":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_RECORD_BYTES:
                    return self._json(dict(error="record too large"), 413)
                r = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(dict(error="bad request"), 400)
            t = r.get("type")
            if t == "decision":
                if r.get("item_id") not in pack.index or r.get("decision") not in DECISIONS:
                    return self._json(dict(error="unknown item or decision"), 400)
                if r.get("reason") is not None and (r["decision"] != "decline" or r["reason"] not in DECLINE_REASONS):
                    return self._json(dict(error="a reason goes only with a decline, and must be one of "
                                                 + ", ".join(DECLINE_REASONS)), 400)
                rec = dict(type="decision", item_id=r["item_id"], item_index=pack.index[r["item_id"]],
                           decision=r["decision"], active_ms=r.get("active_ms"),
                           display=r.get("display"), process=r.get("process"))
                if r.get("reason"):
                    rec["reason"] = r["reason"]
                if isinstance(r.get("note"), str) and r["note"].strip():
                    rec["note"] = r["note"].strip()[:500]
            elif t == "link":
                a_id, b_id = r.get("item_id"), r.get("other_id")
                if a_id not in pack.index or b_id not in pack.index or a_id == b_id or not isinstance(r.get("same"), bool):
                    return self._json(dict(error="a link needs two different pack items and same = true/false"), 400)
                rec = dict(type="link", item_id=a_id, other_id=b_id, same=r["same"], process=r.get("process"))
            elif t in EVENTS:
                rec = dict(type=t, item_index=r.get("item_index"))
                for f in ("to_index", "reason", "input", "client", "process", "goal", "scope", "session", "name",
                          "experience", "open_pairs", "closed", "answered"):
                    if f in r:
                        rec[f] = r[f]
                if t == "session_start":
                    rec.update(tool_version=TOOL_VERSION, pack_dir=str(pack.root), n_items=pack.n,
                               decided_before=len(log.decisions))
            else:
                return self._json(dict(error="unknown record type"), 400)
            log.write(rec)
            return self._json(dict(ok=True, decided=len(log.decisions), next=first_open(pack, log), scope=log.scope))

    return H


def export(pack, log, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write("item_id\tframe_ui\ty_px\tx_px\tdecision\tnote\tannotator\tpack_id\tdecline_reason\n")
        for it in pack.m["items"]:
            d = log.decisions.get(it["id"])
            if d:
                note = log.notes.get(it["id"], "").replace("\t", " ").replace("\n", " ")
                f.write(f"{it['id']}\t{it['t']}\t{it['y']}\t{it['x']}\t{d}\t{note}\t{log.annotator}\t{pack.m['pack_id']}"
                        f"\t{log.reasons.get(it['id'], '')}\n")
    print(f"wrote {out}: {len(log.decisions)} of {pack.n} items decided")
    if log.links:
        lpath = Path(out).with_name(Path(out).stem + "_links.tsv")
        with open(lpath, "w", encoding="utf-8") as f:
            f.write("item_id\tother_id\tsame_cell\tannotator\tpack_id\n")
            for k, same in sorted(log.links.items()):
                a_id, b_id = k.split("|")
                f.write(f"{a_id}\t{b_id}\t{'yes' if same else 'no'}\t{log.annotator}\t{pack.m['pack_id']}\n")
        print(f"wrote {lpath}: {len(log.links)} linked pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-dir", default=None, help="default: the pack directory")
    ap.add_argument("--export", default=None, help="write decisions to this TSV and exit")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--budget", type=int, default=None, help="suggested session goal: this many decisions")
    ap.add_argument("--minutes", type=float, default=None, help="suggested session goal: this many active minutes")
    ap.add_argument("--scope", type=int, default=None, help="suggested scope: review only the top N items")
    a = ap.parse_args()
    name = "".join(ch for ch in a.annotator if ch.isalnum() or ch in "-_")
    if not name:
        raise SystemExit("annotator name must contain letters or digits")
    pack = Pack(a.pack)
    log_dir = Path(a.log_dir or a.pack)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = Log(log_dir / f"labels_{pack.m['pack_id']}_{name}.jsonl", pack.m["pack_id"], name)
    if a.export:
        return export(pack, log, a.export)
    defaults = {k: v for k, v in (("budget", a.budget), ("minutes", a.minutes), ("scope", a.scope)) if v}
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(pack, log, defaults))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"pack {pack.m['pack_id']} ({pack.n} items), annotator {name}: {len(log.decisions)} decided, "
          f"resuming at item {first_open(pack, log) + 1}\nlog: {log.path}\nopen {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.write(dict(type="session_stop", item_index=None))
        print("\nstopped; every decision is saved")


if __name__ == "__main__":
    main()
