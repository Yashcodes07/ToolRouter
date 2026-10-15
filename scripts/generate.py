import argparse, json, os, random, re, time
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError, BadRequestError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY is empty: add it to .env and save the file")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/gpt-oss-120b")
ap.add_argument("--per-tool", type=int, default=250)   # raw examples per tool
ap.add_argument("--batch", type=int, default=12)       # examples per API call
ap.add_argument("--sleep", type=float, default=10)   
ap.add_argument("--only", default="")
ap.add_argument("--out", default="data/raw.jsonl")  # stays under the per-minute token limit
args = ap.parse_args()

TOOLS_FILE, OUT = ROOT / "data/tools_kept.json", ROOT / args.out
if not TOOLS_FILE.exists():
    raise SystemExit("data/tools_kept.json missing: run the keep/holdout step first")
tools = json.load(open(TOOLS_FILE))
if args.only:
      tools = [t for t in tools if t["name"] in set(args.only.split(","))]
client = Groq()

STYLES = ["terse", "polite", "casual", "detailed", "vague but sufficient",
          "with a typo or two", "imperative command", "question form"]
THEMES = ["software project", "personal notes", "study and exams", "travel planning",
          "work meetings", "home and groceries", "fitness", "finance and budgeting",
          "reading list", "team and people", "event planning", "research"]

ENT = ("Every entity name used in arguments must appear verbatim in the request, "
       "for example: Add to the entity AuthModule the facts: ...")
HINTS = {
    "edit_file": 'Every request must name the file path (folder and filename with extension) and quote the exact old text and the exact new text, e.g. In config/app.yaml replace "debug: true" with "debug: false".',
    "search_files": "Every request must give the start folder and the literal glob pattern exactly as in arguments, e.g. Find files matching *.md under docs/.",
    "add_task": "Every request must contain the exact task title used in arguments, with any due date written as YYYY-MM-DD, e.g. Add a task 'Pay electricity bill' due 2026-11-03, high priority.",
    "read_multiple_files": "Every request must list each full file path exactly as in arguments.",
    "read_text_file": "Every request must contain the full file path with folder and filename.",
    "get_file_info": "Every request must contain the full file path with folder and filename.",
    "create_directory": "Every request must contain the full directory path.",
    "directory_tree": "Every request must contain the full directory path.",
    "add_observations": ENT, "delete_observations": ENT, "create_entities": ENT,
    "create_relations": ENT, "delete_relations": ENT,
    "delete_entities": ENT, "open_nodes": ENT, "search_nodes": ENT,
}


def spec(t):
    return json.dumps({"name": t["name"], "description": (t["description"] or "")[:400],
                       "input_schema": t["inputSchema"]}, separators=(",", ":"))


def make_prompt(t):
    hint = HINTS.get(t["name"], "")
    hint_line = f"- {hint}\n" if hint else ""
    return f"""You create training data for a tool-calling model.
Tool:
{spec(t)}

Write {args.batch} different realistic user requests that require exactly this tool.
Theme: {random.choice(THEMES)}. Style: {random.choice(STYLES)}.
Rules:
- "arguments" must follow input_schema exactly: correct names and types, every required field present, no invented fields.
- Include optional arguments in about half of the examples.
- File paths must be relative with no leading slash (for example notes/todo.txt, src/app/main.py).
- The request text must contain every string value that appears in arguments (paths, names, titles, patterns). Never rely on the reader guessing a value.
{hint_line}- Vary wording, length and detail. Never repeat a request.
Return only JSON: {{"examples":[{{"query":"...","arguments":{{...}}}}]}}"""


def call(t, json_mode=True):
    extra = {"reasoning_effort": "low"} if "gpt-oss" in args.model else {}
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    return client.chat.completions.create(
        model=args.model, temperature=0.9,
        messages=[{"role": "user", "content": make_prompt(t)}],
        extra_body=extra, **kw)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))["examples"]


counts, fails = Counter(), Counter()
if OUT.exists():
    for line in open(OUT):
        counts[json.loads(line)["tool"]] += 1

used = 0
while True:
    todo = [t for t in tools if counts[t["name"]] < args.per_tool and fails[t["name"]] < 5]
    if not todo:
        print("Done: every tool reached its target or was skipped after repeated failures.")
        break
    t = min(todo, key=lambda x: counts[x["name"]])      # least-filled tool first
    try:
        try:
            r = call(t)
        except BadRequestError:                          # JSON mode failed: retry as plain text
            r = call(t, json_mode=False)
    except RateLimitError as e:
        msg = str(e).lower()
        if "per day" in msg or "tpd" in msg or "rpd" in msg:
            print("Daily limit reached. Switch model or run again tomorrow."); break
        print("Per-minute limit, waiting 30s"); time.sleep(30); continue
    except Exception as e:
        fails[t["name"]] += 1
        print(f"error on {t['name']} ({fails[t['name']]}/5):", str(e)[:120])
        time.sleep(5); continue

    used += r.usage.total_tokens
    n = 0
    try:
        exs = parse(r.choices[0].message.content)
        with open(OUT, "a") as f:
            for ex in exs:
                if isinstance(ex, dict) and isinstance(ex.get("query"), str) \
                        and isinstance(ex.get("arguments"), dict):
                    f.write(json.dumps({"tool": t["name"], "query": ex["query"],
                                        "arguments": ex["arguments"], "model": args.model}) + "\n")
                    n += 1
    except Exception:
        pass
    counts[t["name"]] += n
    fails[t["name"]] = 0 if n else fails[t["name"]] + 1
    print(f"{t['name']:<26} +{n:>2} | total {sum(counts.values())} | tokens {used}")
    time.sleep(args.sleep)