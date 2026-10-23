import json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
g = {"__file__": str(here / "rescore.py")}
exec((here / "rescore.py").read_text().split('for split in ["test_seen"')[0], g)   # reuse helpers
answerable, core_args, RES = g["answerable"], g["core_args"], g["RES"]

runs = sys.argv[1:] or ["baseline", "run1-r32-lr2e-4"]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


print("| Split | Model | JSON valid | Tool correct | Args exact | Args core (answerable rows) |")
print("|---|---|---|---|---|---|")
for split in ["test_seen", "test_unseen"]:
    for run in runs:
        path = RES / f"{run}_{split}_predictions.json"
        if not path.exists():
            print(f"| {split} | {run} | missing file | | | |"); continue
        preds = json.load(open(path))
        n = len(preds)
        rows = [p for p in preds if answerable(json.loads(p["gold_args"]), p["query"], p["tool"])]
        core_ok = sum(
            1 for p in rows
            if p["pred"] and p["pred"]["name"] == p["tool"]
            and core_args(p["tool"], p["pred"]["arguments"]) == core_args(p["tool"], json.loads(p["gold_args"])))
        print(f"| {split} (n={n}) | {run} | {pct(sum(p['json_valid'] for p in preds), n)} "
              f"| {pct(sum(p['tool_correct'] for p in preds), n)} "
              f"| {pct(sum(p['args_exact'] for p in preds), n)} "
              f"| {pct(core_ok, len(rows))} (n={len(rows)}) |")