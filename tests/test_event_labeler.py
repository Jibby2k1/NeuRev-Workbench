"""Event labeler: build a pack from a small synthetic movie, then drive the review server's API (no lab data)."""
import json
import subprocess
import sys
import threading
import urllib.error
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
    first, second, third = (it["id"] for it in m["items"][:3])
    assert post(dict(type="decision", item_id=first, decision="accept", process=dict(active_ms=900)))["next"] == 1

    def rejected(rec):
        try:
            post(rec)
        except urllib.error.HTTPError as e:
            return e.code == 400
        return False

    assert rejected(dict(type="decision", item_id=second, decision="decline", reason="boredom"))
    assert rejected(dict(type="decision", item_id=second, decision="accept", reason="motion"))
    assert rejected(dict(type="link", item_id=first, other_id=first, same=True))
    assert rejected(dict(type="link", item_id=first, other_id="nope", same=True))
    assert rejected(dict(type="link", item_id=first, other_id=second, same="yes"))
    assert post(dict(type="decision", item_id=second, decision="decline", reason="motion"))["ok"]
    assert post(dict(type="decision", item_id=third, decision="accept"))["ok"]
    assert post(dict(type="link", item_id=third, other_id=first, same=True))["ok"]
    st = json.loads(urllib.request.urlopen(url + "/api/state").read())
    assert st["reasons"] == {second: "motion"} and st["links"] == {"|".join(sorted((first, third))): True}
    srv.shutdown()
    again = LS.Log(tmp_path / "l.jsonl", m["pack_id"], "tester")      # resume
    assert again.decisions == {first: "accept", second: "decline", third: "accept"} and LS.first_open(p, again) == 3
    assert again.reasons == {second: "motion"} and len(again.links) == 1
    LS.export(p, again, tmp_path / "out.tsv")
    rows = (tmp_path / "out.tsv").read_text().splitlines()
    assert rows[0].endswith("\tdecline_reason") and any(r.startswith(second) and r.endswith("\tmotion") for r in rows)
    assert len((tmp_path / "out_links.tsv").read_text().splitlines()) == 2
    summ = subprocess.run([sys.executable, str(TOOL / "summarize.py"), "--key", str(key / "key.json"),
                           str(tmp_path / "l.jsonl"), "--json", str(tmp_path / "s.json")],
                          check=True, capture_output=True, text=True)
    rep = json.loads((tmp_path / "s.json").read_text())["annotators"]["tester"]
    assert rep["linking"] == dict(pairs_answered=1, same_cell=1, accepted_items=2, distinct_accepted_cells=1,
                                  contradictions=0, pairs_dropped_after_redecision=0)
    assert sum(sum(v.values()) for v in rep["decline_reasons_by_kind"].values()) == 1, summ.stdout
    again.write(dict(type="settings", scope=1))                     # review only the top item: it is done
    assert LS.first_open(p, again) == 1 and LS.Log(tmp_path / "l.jsonl", m["pack_id"], "tester").scope == 1
    with pytest.raises(SystemExit):
        LS.Log(tmp_path / "l.jsonl", "otherpack", "tester")
