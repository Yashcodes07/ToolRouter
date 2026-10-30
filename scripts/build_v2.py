"""Build the v2 dataset: stricter training rows, test rows copied unchanged from v1.

  python scripts/build_v2.py --repo Yashcodes07/toolrouter-mcp-calls-v2          # dry run
  python scripts/build_v2.py --repo Yashcodes07/toolrouter-mcp-calls-v2 --push   # upload (private)
"""
import argparse, json, os
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="Yashcodes07/toolrouter-mcp-calls")   # v1: test sets are copied from here
ap.add_argument("--repo", required=True)                                # new v2 dataset repo
ap.add_argument("--extra", default="data/clean_v2_new.json")            # newly generated + filtered rows
ap.add_argument("--push", action="store_true")
ap.add_argument("--cap", type=int, default=200)
ap.add_argument("--public", action="store_true")
args = ap.parse_args()

g = {"__file__": str(ROOT / "analysis" / "rescore.py")}
exec((ROOT / "analysis" / "rescore.py").read_text().split('for split in ["test_seen"')[0], g)
answerable = g["answerable"]

tools = {t["name"]: t for t in json.load(open(ROOT / "data/tools_kept.json"))}
hold = set(json.load(open(ROOT / "data/holdout.json")))
token = os.getenv("HF_TOKEN") or None

v1 = {s: list(load_dataset(args.src, split=s, token=token)) for s in ["val", "test_seen", "test_unseen"]}


def sig(tool, a):
    return (tool, json.dumps(a, sort_keys=True))


blocked = {sig(r["tool"], json.loads(r["arguments"])) for rows in v1.values() for r in rows}
test_q = {(r["tool"], r["query"].lower().strip()) for s in ("test_seen", "test_unseen") for r in v1[s]}

pool = json.load(open(ROOT / "data/clean.json"))
extra = ROOT / args.extra
if extra.exists():
    pool += json.load(open(extra))
    print("added extra rows from", extra)
else:
    print("no extra file found, using only filtered v1 rows")

stats, seen_q, train = Counter(), set(), []
for r in pool:
    t = r["tool"]
    if t in hold or t not in tools:
        continue
    if sig(t, r["arguments"]) in blocked:
        stats["same call as a val/test row"] += 1; continue
    key = (t, r["query"].lower().strip())
    if key in test_q or key in seen_q:
        stats["duplicate or test query"] += 1; continue
    if not answerable(r["arguments"], r["query"], t):
        stats["not answerable"] += 1; continue
    seen_q.add(key)
    train.append({"tool": t, "server": tools[t]["server"], "query": r["query"],
                  "arguments": json.dumps(r["arguments"], sort_keys=True)})
import random
rng = random.Random(42)
by_tool = {}
for r in train:
    by_tool.setdefault(r["tool"], []).append(r)
train = []
for rows_t in by_tool.values():
       rng.shuffle(rows_t)
       train += rows_t[:args.cap]
rng.shuffle(train)

val = [r for r in v1["val"] if answerable(json.loads(r["arguments"]), r["query"], r["tool"])]
print("dropped from pool:", dict(stats))
print(f"train {len(train)} | val {len(val)} (of {len(v1['val'])}) | "
      f"test_seen {len(v1['test_seen'])} | test_unseen {len(v1['test_unseen'])} (copied unchanged)")
per = Counter(r["tool"] for r in train)
print("train rows per tool:", dict(sorted(per.items(), key=lambda kv: kv[1])))
assert not {r["tool"] for r in train} & hold, "held-out tool in train"
assert not {sig(r["tool"], json.loads(r["arguments"])) for r in train} & blocked, "train overlaps val/test"

if not args.push:
    raise SystemExit("Dry run done. Add --push to upload.")

ds = DatasetDict({"train": Dataset.from_list(train), "val": Dataset.from_list(val),
                  "test_seen": Dataset.from_list(v1["test_seen"]),
                  "test_unseen": Dataset.from_list(v1["test_unseen"])})
ds.push_to_hub(args.repo, private=not args.public)
api = HfApi(token=token)
for src, dst in (("data/tools_kept.json", "tools.json"), ("data/holdout.json", "holdout_tools.json")):
    api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=dst, repo_id=args.repo, repo_type="dataset")
print("Pushed:", f"https://huggingface.co/datasets/{args.repo}")