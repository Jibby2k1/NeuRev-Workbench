#!/usr/bin/env python3
"""Summarize labelling sessions against a pack's key (Python 3 standard library only).

    python summarize.py --key PACK_KEY_DO_NOT_SHARE/key.json RESULTS [more ...] [--json out.json]

RESULTS are label_server.py logs (labels_<pack>_<name>.jsonl) or the results zips reviewers download from the
browser version (neurev_review_<pack>_<name>_<date>.zip); both can be mixed.

Per annotator (latest decision per item counts):
  * decisions, accept rate per item kind: proposal, random (control patches), annotation (hidden reference
    annotations that were not among the proposals), and proposals that cover a reference annotation;
  * reference annotations re-accepted: hidden + covered items accepted / decided;
  * review speed: median / mean active seconds per decision, decisions per hour of active time;
  * accept rate of proposals by queue position (blocks of 100), to see how relevance falls with rank.
Run it only where the key lives (not on a labeller's machine).
"""
import argparse
import io
import json
import zipfile
import statistics
from collections import defaultdict


def load_logs(paths, pack_id):
    by_ann = defaultdict(dict)
    proc = defaultdict(list)
    sittings = defaultdict(lambda: dict(sessions=0, goals_reached=0, last_scope=None))
    def lines(p):
        if str(p).endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith(".jsonl"):
                        yield from io.TextIOWrapper(z.open(n), encoding="utf-8")
        else:
            with open(p, encoding="utf-8") as f:
                yield from f

    for p in paths:
        if True:
            for line in lines(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("pack_id") != pack_id:
                    raise SystemExit(f"{p} belongs to pack {r.get('pack_id')}, the key is for {pack_id}")
                st = sittings[r.get("annotator")]
                if r.get("type") == "reviewer":
                    st["experience"] = r.get("experience")
                if r.get("type") == "decision" and r.get("note"):
                    st.setdefault("notes", {})[r["item_id"]] = r["note"]
                elif r.get("type") == "decision":
                    st.setdefault("notes", {}).pop(r["item_id"], None)
                if r.get("type") == "session_start":
                    st["sessions"] += 1
                elif r.get("type") == "goal_reached":
                    st["goals_reached"] += 1
                elif r.get("type") == "settings":
                    st["last_scope"] = r.get("scope")
                if r.get("type") == "decision":
                    by_ann[r["annotator"]][r["item_id"]] = r["decision"]
                    ms = (r.get("process") or {}).get("active_ms", r.get("active_ms"))
                    if isinstance(ms, (int, float)):
                        proc[r["annotator"]].append(ms / 1000.0)
    return by_ann, proc, sittings


def rate(decs):
    n = len(decs)
    acc = sum(d == "accept" for d in decs)
    uns = sum(d == "unsure" for d in decs)
    return dict(decided=n, accepted=acc, unsure=uns, accept_rate=round(acc / n, 3) if n else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    key = json.load(open(a.key, encoding="utf-8"))
    items = {it["id"]: it for it in key["items"]}
    by_ann, secs, sittings = load_logs(a.logs, key["pack_id"])
    report = dict(pack_id=key["pack_id"], counts=key["counts"], annotators={})
    for name, decs in sorted(by_ann.items()):
        kinds = defaultdict(list)
        blocks = defaultdict(list)
        for iid, d in decs.items():
            it = items.get(iid)
            if it is None:
                continue
            kinds[it["kind"]].append(d)
            if it.get("covers_annotations"):
                kinds["proposal_covering_annotation"].append(d)
            if it["kind"] == "proposal":
                blocks[it["position"] // 100].append(d)
        ref = kinds.get("annotation", []) + kinds.get("proposal_covering_annotation", [])
        s = secs.get(name, [])
        rep = dict(decisions=len(decs), sittings=sittings[name],
                   by_kind={k: rate(v) for k, v in sorted(kinds.items())},
                   reference_annotations_reaccepted=rate(ref),
                   seconds_per_decision=dict(median=round(statistics.median(s), 2) if s else None,
                                             mean=round(statistics.mean(s), 2) if s else None,
                                             decisions_per_active_hour=round(3600 * len(s) / sum(s), 1)
                                             if s and sum(s) > 0 else None),
                   proposal_accept_rate_by_queue_block={f"{100 * k + 1}-{100 * k + 100}": rate(v)["accept_rate"]
                                                        for k, v in sorted(blocks.items())})
        report["annotators"][name] = rep
        sit = sittings[name]
        print(f"\n== {name}: {len(decs)} decisions in {sit['sessions']} sittings "
              f"({sit['goals_reached']} session goals reached; scope: "
              f"{'top ' + str(sit['last_scope']) if sit['last_scope'] else 'whole queue'}; "
              f"experience: {sit.get('experience') or 'not given'}; notes: {len(sit.get('notes', {}))})")
        for k, v in rep["by_kind"].items():
            print(f"  {k:30s} {v['accepted']:4d} / {v['decided']:4d} accepted ({v['accept_rate']}), {v['unsure']} unsure")
        r = rep["reference_annotations_reaccepted"]
        print(f"  reference annotations re-accepted: {r['accepted']} / {r['decided']}")
        sp = rep["seconds_per_decision"]
        print(f"  seconds per decision: median {sp['median']}, mean {sp['mean']}; "
              f"{sp['decisions_per_active_hour']} decisions per active hour")
        print("  proposal accept rate by queue block:", rep["proposal_accept_rate_by_queue_block"])
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
