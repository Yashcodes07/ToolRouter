import json, sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "run1-r32-lr2e-4"
RES = Path(__file__).resolve().parent.parent / "results"


def norm(x):
    if isinstance(x, str):
        return x.strip().lower()
    if isinstance(x, dict):
        return {k: norm(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return sorted((norm(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    return x


def category(p):
    if not p["pred"]:
        return "invalid_json"
    if p["pred"]["name"] != p["tool"]:
        return "wrong_tool"
    return "exact" if p["args_exact"] else "wrong_args"


for split in ["test_seen", "test_unseen"]:
    path = RES / f"{RUN}_{split}_predictions.json"
    if not path.exists():
        print("missing:", path); continue
    preds = json.load(open(path))
    n = len(preds)
    cats = Counter(category(p) for p in preds)
    lenient = cats["exact"]
    fields, bad_by_tool = defaultdict(Counter), defaultdict(list)
    for p in preds:
        if category(p) != "wrong_args":
            continue
        gold, pred = json.loads(p["gold_args"]), p["pred"]["arguments"]
        if norm(gold) == norm(pred):
            lenient += 1; fields[p["tool"]]["(case/spacing/order only)"] += 1
        else:
            for k in set(gold) | set(pred):
                if k not in pred:
                    fields[p["tool"]][f"{k}: missing"] += 1
                elif k not in gold:
                    fields[p["tool"]][f"{k}: extra"] += 1
                elif norm(gold[k]) != norm(pred[k]):
                    fields[p["tool"]][f"{k}: wrong value"] += 1
        bad_by_tool[p["tool"]].append(p)

    print(f"\n########## {split} (n={n}) ##########")
    print({k: f"{v} ({100 * v / n:.0f}%)" for k, v in cats.items()})
    print(f"exact {100 * cats['exact'] / n:.1f}%  |  lenient (ignores case, spacing, list order) {100 * lenient / n:.1f}%")
    conf = Counter((p["tool"], p["pred"]["name"]) for p in preds if category(p) == "wrong_tool")
    print("tool confusions (gold -> predicted):", conf.most_common(6))
    for t in sorted(bad_by_tool, key=lambda t: -len(bad_by_tool[t]))[:6]:
        print(f"\n--- {t}: {len(bad_by_tool[t])} wrong-argument cases. Most common issues:")
        for f, c in fields[t].most_common(4):
            print(f"    {c:3d}  {f}")
        p = bad_by_tool[t][0]
        print("    e.g. Q:", p["query"][:160])
        print("         gold:", p["gold_args"][:200])
        print("         pred:", json.dumps(p["pred"]["arguments"])[:200])