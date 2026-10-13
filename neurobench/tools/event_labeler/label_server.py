#!/usr/bin/env python3
"""Event labeler: local review server (Python 3.8+, standard library only).

    python3 label_server.py PACK_DIR --annotator NAME [--port 8765] [--log-dir DIR]

Opens http://127.0.0.1:PORT in the browser. Every decision is appended to
    <log-dir>/labels_<pack_id>_<annotator>.jsonl      (one JSON object per line, flushed + fsynced)
so closing the browser, the terminal or the machine never loses a decision. Starting the server again with the
same pack and annotator resumes at the first item without a decision. Items are in the pack's order (most
relevant first). The latest decision for an item wins
(Back lets you change one). Session start / pause / resume / navigate / stop lines are logged too.

Each decision line carries a "process" record of how it was made: time on the item (active, paused, hidden), time
to first interaction, time spent on each clip frame (dwell_ms), loops played, frame steps, play toggles, ring /
contrast / speed changes with their times, start and end display state, keyboard or mouse, and whether the item was
revisited (and its earlier answer). session_start lines carry the browser/screen and the tool's file hashes.

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
DECISIONS = ("accept", "decline")
EVENTS = ("session_start", "pause", "resume", "session_stop", "navigate")
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
                    if r.get("type") == "decision":
                        self.decisions[r["item_id"]] = r["decision"]
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, rec):
        rec = dict(rec, pack_id=self.pack_id, annotator=self.annotator,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), unix=round(time.time(), 3))
        with self.lock:
            self.f.write(json.dumps(rec, sort_keys=True) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())
            if rec.get("type") == "decision":
                self.decisions[rec["item_id"]] = rec["decision"]
        return rec


def first_open(pack, log):
    for k, i in enumerate(pack.ids):
        if i not in log.decisions:
            return k
    return pack.n


def make_handler(pack, log):
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
                rec = dict(type="decision", item_id=r["item_id"], item_index=pack.index[r["item_id"]],
                           decision=r["decision"], active_ms=r.get("active_ms"),
                           display=r.get("display"), process=r.get("process"))
            elif t in EVENTS:
                rec = dict(type=t, item_index=r.get("item_index"))
                for f in ("to_index", "reason", "input", "client", "process"):
                    if f in r:
                        rec[f] = r[f]
                if t == "session_start":
                    rec.update(tool_version=TOOL_VERSION, pack_dir=str(pack.root), n_items=pack.n,
                               decided_before=len(log.decisions))
            else:
                return self._json(dict(error="unknown record type"), 400)
            log.write(rec)
            return self._json(dict(ok=True, decided=len(log.decisions), next=first_open(pack, log)))

    return H


def export(pack, log, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write("item_id\tframe_ui\ty_px\tx_px\tdecision\tannotator\tpack_id\n")
        for it in pack.m["items"]:
            d = log.decisions.get(it["id"])
            if d:
                f.write(f"{it['id']}\t{it['t']}\t{it['y']}\t{it['x']}\t{d}\t{log.annotator}\t{pack.m['pack_id']}\n")
    print(f"wrote {out}: {len(log.decisions)} of {pack.n} items decided")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-dir", default=None, help="default: the pack directory")
    ap.add_argument("--export", default=None, help="write decisions to this TSV and exit")
    ap.add_argument("--no-browser", action="store_true")
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
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(pack, log))
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
