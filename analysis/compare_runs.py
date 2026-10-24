"""Paired comparison of two runs on the same rows (is the difference real or noise?).

  python analysis/compare_runs.py run1-r32-lr2e-4 run2-r64-lr2e-4
"""
import json, sys
from math import comb
from pathlib import Path

here = Path(__file__).resolve().parent
g = {"__file__": str(here / "rescore.py")}
exec((here / "rescore.py").read_text().split('for split in ["test_seen"')[0], g)   # reuse helpers
answerable, core_args, RES = g["answerable"], g["core_args"], g["RES"]
A, B = sys.argv[1], sys.argv[2]


def load(run, split):
    return {(p["tool"], p["query"]): p for p in json.load(open(RES / f"{run}_{split}_predictions.json"))}


def core_ok(p):
    return int(bool(p["pred"]) and p["pred"]["name"] == p["tool"]
               and core_args(p["tool"], p["pred"]["arguments"]) == core_args(p["tool"], json.loads(p["gold_args"])))


def sign_test(wins_a, wins_b):
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


for split in ["test_seen", "test_unseen"]:
    a, b = load(A, split), load(B, split)
    keys = [k for k in a if k in b and answerable(json.loads(a[k]["gold_args"]), k[1], k[0])]
    if not keys:
        print(f"\n=== {split}: no common answerable rows ==="); continue
    print(f"\n=== {split}: {len(keys)} answerable rows in common ===")
    for name, fn in (("tool_correct", lambda p: p["tool_correct"]), ("args_core", core_ok)):
        wa = sum(1 for k in keys if fn(a[k]) and not fn(b[k]))
        wb = sum(1 for k in keys if fn(b[k]) and not fn(a[k]))
        acc_a = 100 * sum(fn(a[k]) for k in keys) / len(keys)
        acc_b = 100 * sum(fn(b[k]) for k in keys) / len(keys)
        print(f"{name:<13} {A}: {acc_a:5.1f}%  {B}: {acc_b:5.1f}%  | only {A} right: {wa:3d}, "
              f"only {B} right: {wb:3d} | sign-test p = {sign_test(wa, wb):.3f}")