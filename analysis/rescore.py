"""Re-score saved predictions with fairer metrics. No GPU needed.

  python analysis/rescore.py [run_name]

answerable = the gold values can actually be read from the request text:
             full paths (segments in order), names, titles, relation types, task ids,
             head/tail numbers, and any non-default priority/status/dryRun.
args_exact = strict exact match (same as the notebooks).
args_core  = exact match after dropping free text (file content, observation text, exclude
             globs), dropping arguments that equal the schema default, and ignoring case,
             trailing slashes and trailing punctuation.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "run1-r32-lr2e-4"
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TOOLS = {t["name"]: t["inputSchema"] for t in json.load(open(ROOT / "data" / "tools_kept.json"))}
DEFAULTS = {n: {k: v["default"] for k, v in s.get("properties", {}).items() if "default" in v}
            for n, s in TOOLS.items()}

STRICT_STR = {"path", "source", "destination", "paths", "name", "names", "entityName",
              "entityNames", "from", "to", "title", "keyword", "pattern", "query", "relationType"}
PATHY = {"path", "source", "destination", "paths"}
NUM_KEYS = {"task_id", "head", "tail", "limit"}
WORD_KEYS = {"priority", "status"}            # a non-default value must be named in the request
FREE_KEYS = {"content", "contents", "excludePatterns"}


def nq(s):
    return re.sub(r"[\s_\-*./\\]+", "", s.lower())


def in_order(parts, qn):
    pos = 0
    for p in parts:
        i = qn.find(nq(p), pos)
        if i < 0:
            return False
        pos = i + len(nq(p))
    return True


def answerable(args, query, tool=None):
    qn, ql, ok = nq(query), query.lower(), [True]
    dflt = DEFAULTS.get(tool, {})

    def walk(x, key=None, top=False):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k, top)
        elif isinstance(x, list):
            for i in x:
                walk(i, key)
        elif isinstance(x, bool):
            if key == "dryRun" and x and not re.search(r"dry|preview|diff", ql):
                ok[0] = False
        elif isinstance(x, str):
            if key in STRICT_STR:
                parts = [p for p in re.split(r"[\\/]", x) if p] if key in PATHY else [x]
                if not (in_order(parts, qn) if key in PATHY else nq(x) in qn):
                    ok[0] = False
            elif key in WORD_KEYS and x != dflt.get(key) and x.lower() not in ql:
                ok[0] = False
        elif isinstance(x, int) and key in NUM_KEYS:
            if not re.search(rf"(?<!\d){x}(?!\d)", query):
                ok[0] = False

    walk(args, top=True)
    return ok[0]


def core(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in FREE_KEYS:
                continue
            if k == "observations" and isinstance(v, list) and all(isinstance(i, str) for i in v):
                continue
            out[k] = core(v)
        return {k: out[k] for k in sorted(out)}
    if isinstance(x, list):
        return sorted((core(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, str):
        return x.strip().lower().rstrip("/.! ")
    return x


def core_args(tool, args):
    d = DEFAULTS.get(tool, {})
    a = {k: v for k, v in args.items() if v is not None and not (k in d and v == d[k])}
    return core(a)


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


for split in ["test_seen", "test_unseen"]:
    path = RES / f"{RUN}_{split}_predictions.json"
    if not path.exists():
        print("missing:", path); continue
    preds = json.load(open(path))
    S = {"all": defaultdict(int), "ans": defaultdict(int)}
    per_tool = defaultdict(lambda: [0, 0])          # tool -> [answerable, total]
    for p in preds:
        gold = json.loads(p["gold_args"])
        a = answerable(gold, p["query"], p["tool"])
        per_tool[p["tool"]][1] += 1
        per_tool[p["tool"]][0] += a
        pred = p["pred"]
        core_ok = int(bool(pred) and pred["name"] == p["tool"]
                      and core_args(p["tool"], pred["arguments"]) == core_args(p["tool"], gold))
        for grp in (["all", "ans"] if a else ["all"]):
            S[grp]["n"] += 1
            S[grp]["tool_correct"] += p["tool_correct"]
            S[grp]["args_exact"] += p["args_exact"]
            S[grp]["args_core"] += core_ok
    n, m = S["all"]["n"], S["ans"]["n"]
    print(f"\n=== {RUN} | {split} ===  rows {n} | answerable {m} ({100 * m / n:.0f}%)")
    print(f"{'':<22}{'tool_correct':>14}{'args_exact':>12}{'args_core':>11}")
    for grp, label in (("all", "all rows"), ("ans", "answerable rows")):
        d = S[grp]
        print(f"{label:<22}{pct(d['tool_correct'], d['n']):>14}{pct(d['args_exact'], d['n']):>12}{pct(d['args_core'], d['n']):>11}")
    low = sorted(per_tool.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]
    print("least answerable tools:", ", ".join(f"{t} {100 * a / b:.0f}%" for t, (a, b) in low))