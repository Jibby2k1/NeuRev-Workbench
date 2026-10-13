"""Event labeler: build a pack from a small synthetic movie, then drive the review server's API (no lab data)."""
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "neurobench" / "tools" / "event_labeler"
tifffile = pytest.importorskip("tifffile")


def _movie(path, T=420, H=96, W=128, seed=0):
    rng = np.random.default_rng(seed)
    base = 400.0 + 200.0 * np.exp(-((np.arange(H)[:, None] - 48) ** 2 + (np.arange(W)[None] - 64) ** 2) / 4000.0)
    yy, xx = np.mgrid[:H, :W]
    events = [(300, 40, 50), (330, 60, 90), (360, 30, 100)]
    mov = np.empty((T, H, W), np.uint16)
    for t in range(T):
        f = base.copy()
        for te, ye, xe in events:
            if t >= te:
                f += 300.0 * np.exp(-(t - te) / 7.0) * np.exp(-((yy - ye) ** 2 + (xx - xe) ** 2) / 18.0)
        mov[t] = np.clip(rng.poisson(f * 3.5) / 3.5, 0, 4095).astype(np.uint16)
    tifffile.imwrite(path, mov)
    return events


def test_build_and_review(tmp_path):
    events = _movie(tmp_path / "m.tif")
    ann = tmp_path / "ann.tsv"
    ann.write_text("x_px\ty_px\tstart_frame_ui\tend_frame_ui\troi_identity\n50\t40\t295\t320\tcellA\n"
                   "20\t20\t340\t360\tquiet_spot\n")
    pack, key = tmp_path / "pack", tmp_path / "key"
    subprocess.run([sys.executable, str(TOOL / "build_pack.py"), "--tif", str(tmp_path / "m.tif"), "--out", str(pack),
                    "--key-out", str(key), "--warmup", "100", "--max-proposals", "40", "--z-min", "4",
                    "--annotations", str(ann)], check=True, capture_output=True)
    m = json.loads((pack / "manifest.json").read_text())
    k = json.loads((key / "key.json").read_text())
    assert "kind" not in json.dumps(m["items"]) and (pack / "label_server.py").exists()
    kinds = {it["kind"] for it in k["items"]}
    assert "proposal" in kinds and "annotation" in kinds
    top = [it for it in k["items"] if it["kind"] == "proposal"][:3]
    found = {(int(i.split(":")[0][1:]), int(i.split(":")[1][1:]), int(i.split(":")[2][1:])) for i in
             (it["id"] for it in top)}
    for te, ye, xe in events:          # every planted event is among the top proposals, at its onset +- 3 frames
        assert any(abs(t - te) <= 3 and abs(y - ye) <= 2 and abs(x - xe) <= 2 for t, y, x in found)

    sys.path.insert(0, str(TOOL))
    import label_server as LS
    p = LS.Pack(pack)
    log = LS.Log(tmp_path / "l.jsonl", m["pack_id"], "tester")
    srv = LS.ThreadingHTTPServer(("127.0.0.1", 0), LS.make_handler(p, log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(rec):
        req = urllib.request.Request(url + "/api/log", data=json.dumps(rec).encode(), method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    st = json.loads(urllib.request.urlopen(url + "/api/state").read())
    assert st["next"] == 0
    first = m["items"][0]["id"]
    assert post(dict(type="decision", item_id=first, decision="accept", process=dict(active_ms=900)))["next"] == 1
    srv.shutdown()
    again = LS.Log(tmp_path / "l.jsonl", m["pack_id"], "tester")      # resume
    assert again.decisions == {first: "accept"} and LS.first_open(p, again) == 1
    with pytest.raises(SystemExit):
        LS.Log(tmp_path / "l.jsonl", "otherpack", "tester")
