"""How much of the training data would survive the stricter 'answerable' check?"""
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "analysis" / "rescore.py"
g = {"__file__": str(src)}
exec(src.read_text().split('for split in ["test_seen"')[0], g)
answerable = g["answerable"]

rows = json.load(open(root / "data" / "clean.json"))
tot, ok = Counter(), Counter()
for r in rows:
    tot[r["tool"]] += 1
    ok[r["tool"]] += answerable(r["arguments"], r["query"], r["tool"])
print(f"{'tool':<22}{'kept':>6}{'of':>6}{'share':>8}")
for t in sorted(tot, key=lambda t: ok[t] / tot[t]):
    print(f"{t:<22}{ok[t]:>6}{tot[t]:>6}{100 * ok[t] / tot[t]:>7.0f}%")
print(f"\nTOTAL kept {sum(ok.values())} of {sum(tot.values())} ({100 * sum(ok.values()) / sum(tot.values()):.0f}%)")