import json, sys
from pathlib import Path
from jsonschema import Draft202012Validator

root = Path(__file__).resolve().parent.parent
tools = json.load(open(root / "data/tools_kept.json"))
hold = set(json.load(open(root / "data/holdout.json")))
names = [t["name"] for t in tools]
errors = []
if len(names) != len(set(names)):
    errors.append("duplicate tool names")
if not hold <= set(names):
    errors.append(f"held-out tools missing from tool list: {hold - set(names)}")
for t in tools:
    try:
        Draft202012Validator.check_schema(t["inputSchema"])
    except Exception as e:
        errors.append(f"{t['name']}: invalid schema ({e})")
    if not t.get("description"):
        errors.append(f"{t['name']}: missing description")
if errors:
    print("\n".join(errors)); sys.exit(1)
print(f"OK: {len(names)} tools, {len(hold)} held out")