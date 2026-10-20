"""What is still wrong after removing unanswerable rows and ignoring free text?

  python analysis/remaining_errors.py [run_name]
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
g = {"__file__": str(here / "rescore.py")}
exec((here / "rescore.py").read_text().split('for split in ["test_seen"')[0], g)   # reuse helpers
answerable, core_args, RES, RUN = g["answerable"], g["core_args"], g["RES"], g["RUN"]

for split in ["test_seen", "test_unseen"]:
    path = RES / f"{RUN}_{split}_predictions.json"
    if not path.exists():
        print("missing:", path); continue
    rows = [p for p in json.load(open(path)) if answerable(json.loads(p["gold_args"]), p["query"], p["tool"])]
    wrong_tool, bad = [], defaultdict(list)
    fields = defaultdict(Counter)
    for p in rows:
        gold = json.loads(p["gold_args"])
        if not p["pred"] or p["pred"]["name"] != p["tool"]:
            wrong_tool.append(p); continue
        pred = p["pred"]["arguments"]
        cg, cp = core_args(p["tool"], gold), core_args(p["tool"], pred)
        if cg == cp:
            continue
        p["diff"] = [k for k in sorted(set(cg) | set(cp)) if cg.get(k) != cp.get(k)]
        for k in p["diff"]:
            fields[p["tool"]][k] += 1
        bad[p["tool"]].append(p)
    n_bad = sum(len(v) for v in bad.values())
    print(f"\n########## {split}: {len(rows)} answerable rows | wrong tool/invalid {len(wrong_tool)} | wrong core args {n_bad} ##########")
    for t in sorted(bad, key=lambda t: -len(bad[t]))[:6]:
        print(f"\n--- {t}: {len(bad[t])} wrong. fields: {dict(fields[t].most_common(3))}")
        for p in bad[t][:2]:
            gold, pred = json.loads(p["gold_args"]), p["pred"]["arguments"]
            k = p["diff"][0]
            print("   Q:", p["query"][:150])
            print("     gold", k, "=", json.dumps(gold.get(k), ensure_ascii=False)[:150])
            print("     pred", k, "=", json.dumps(pred.get(k), ensure_ascii=False)[:150])