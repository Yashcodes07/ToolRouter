import argparse, json, os, random
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)          # e.g. yourname/toolrouter-mcp-calls
ap.add_argument("--push", action="store_true")
ap.add_argument("--public", action="store_true")
ap.add_argument("--unseen-cap", type=int, default=60)
args = ap.parse_args()

tools = {t["name"]: t for t in json.load(open(ROOT / "data/tools_kept.json"))}
hold = set(json.load(open(ROOT / "data/holdout.json")))
clean = json.load(open(ROOT / "data/clean.json"))
rng = random.Random(42)


def row(r):
    # arguments stored as a JSON string: their shapes differ per tool, which breaks Arrow schemas
    return {"tool": r["tool"], "server": tools[r["tool"]]["server"], "query": r["query"],
            "arguments": json.dumps(r["arguments"], sort_keys=True)}


# test_unseen: held-out tools only, capped per tool
by_tool = defaultdict(list)
for r in clean:
    if r["tool"] in hold:
        by_tool[r["tool"]].append(row(r))
unseen = []
for t, rs in by_tool.items():
    rng.shuffle(rs); unseen += rs[:args.unseen_cap]

# seen tools: split by distinct call (tool + arguments), so no call crosses splits
groups = defaultdict(list)
for r in clean:
    if r["tool"] not in hold:
        groups[(r["tool"], json.dumps(r["arguments"], sort_keys=True))].append(row(r))
keys = list(groups); rng.shuffle(keys)
n = len(keys); k = int(n * 0.1)
parts = {"val": keys[:k], "test_seen": keys[k:2 * k], "train": keys[2 * k:]}
data = {name: [x for g in ks for x in groups[g]] for name, ks in parts.items()}
data["test_unseen"] = unseen

assert not {r["tool"] for r in data["train"]} & hold, "held-out tool leaked into train"
for name, rs in data.items():
    print(f"{name:<12}{len(rs):>5} examples, {len({r['tool'] for r in rs})} tools")

if not args.push:
    raise SystemExit("Dry run done. Add --push to upload.")

ds = DatasetDict({name: Dataset.from_list(rs) for name, rs in data.items()})
ds.push_to_hub(args.repo, private=not args.public)
api = HfApi(token=os.getenv("HF_TOKEN") or None)
for src, dst in (("data/tools_kept.json", "tools.json"), ("data/holdout.json", "holdout_tools.json")):
    api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=dst,
                    repo_id=args.repo, repo_type="dataset")
print("Pushed:", f"https://huggingface.co/datasets/{args.repo}")