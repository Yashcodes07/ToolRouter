import argparse, asyncio, json, random, re, shutil, sqlite3, sys
from collections import Counter, defaultdict
from contextlib import AsyncExitStack
from pathlib import Path
from jsonschema import validate, ValidationError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
SANDBOX = ROOT / "sandbox"
WORK = SANDBOX / "work"          # filesystem examples run in here, wiped each time
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()

tools = {t["name"]: t for t in json.load(open(ROOT / "data/tools_kept.json"))}
rows = [json.loads(l) for l in open(ROOT / "data/raw.jsonl")]
if args.limit:
    rows = random.Random(0).sample(rows, min(args.limit, len(rows)))

PATH_KEYS = {"path", "source", "destination", "paths"}
CHECK_KEYS = {"path", "source", "destination", "paths", "pattern", "keyword",
              "title", "due", "name", "names", "entityName", "from", "to"}
NAME_KEYS = {"name", "entityName", "from", "to"}


def iter_paths(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in PATH_KEYS:
                if isinstance(v, str):
                    yield v
                elif isinstance(v, list):
                    yield from (i for i in v if isinstance(i, str))
            else:
                yield from iter_paths(v)
    elif isinstance(x, list):
        for i in x:
            yield from iter_paths(i)


def paths_ok(a):
    return all(not p.startswith("/") and ".." not in p for p in iter_paths(a))


def norm(s):
    return re.sub(r"[\s_\-*]+", "", s.lower())


def ungrounded(x, q, out=None):
    """Return the keys whose string values do not appear in the query."""
    out = [] if out is None else out
    qn = norm(q)

    def ok(k, v):
        if isinstance(v, str):
            if k in PATH_KEYS:
                v = v.rstrip("/").split("/")[-1]   # filename or folder name is enough
            v = norm(v)
            return not v or v in qn
        if isinstance(v, list):
            return all(ok(k, i) for i in v)
        return True

    if isinstance(x, dict):
        for k, v in x.items():
            if k in CHECK_KEYS and not isinstance(v, dict):
                if not ok(k, v):
                    out.append(k)
            else:
                ungrounded(v, q, out)
    elif isinstance(x, list):
        for i in x:
            ungrounded(i, q, out)
    return out


def rewrite(x):
    """Prefix relative paths with the sandbox work dir (execution only)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in PATH_KEYS and isinstance(v, str):
                out[k] = str(WORK / v)
            elif k in PATH_KEYS and isinstance(v, list):
                out[k] = [str(WORK / i) for i in v]
            else:
                out[k] = rewrite(v)
        return out
    if isinstance(x, list):
        return [rewrite(i) for i in x]
    return x


def names_in(x):
    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            if k in NAME_KEYS and isinstance(v, str):
                out.add(v)
            elif k == "names" and isinstance(v, list):
                out |= {s for s in v if isinstance(s, str)}
            else:
                out |= names_in(v)
    elif isinstance(x, list):
        for i in x:
            out |= names_in(i)
    return out


def touch(p, text="hello\nworld\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def seed_fs(tool, a):
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    P = lambda rel: WORK / rel
    if tool in {"read_text_file", "get_file_info"}:
        touch(P(a["path"]))
    elif tool == "read_multiple_files":
        for p in a["paths"]:
            touch(P(p))
    elif tool == "edit_file":
        touch(P(a["path"]), "\n".join(e.get("oldText", "") for e in a.get("edits", [])))
    elif tool in {"list_directory", "directory_tree", "search_files"}:
        P(a["path"]).mkdir(parents=True, exist_ok=True)
    elif tool == "write_file":
        P(a["path"]).parent.mkdir(parents=True, exist_ok=True)
    elif tool == "move_file":
        touch(P(a["source"]))
        P(a["destination"]).parent.mkdir(parents=True, exist_ok=True)


def seed_tasks(a):
    c = sqlite3.connect(SANDBOX / "tasks.db")
    c.execute("CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, title TEXT, "
              "priority TEXT DEFAULT 'medium', due TEXT, done INTEGER DEFAULT 0)")
    c.execute("DELETE FROM tasks")
    c.executemany("INSERT INTO tasks(id,title,priority,due,done) VALUES (?,?,?,?,0)",
                  [(i, f"Sample task {i}", "medium", None) for i in range(1, 61)])
    tid = a.get("task_id")
    if isinstance(tid, int):
        c.execute("INSERT OR IGNORE INTO tasks(id,title,priority,done) VALUES (?,?,?,0)",
                  (tid, f"Sample task {tid}", "medium"))
    c.commit(); c.close()


# ---------- stage 1: offline checks ----------
gkeys, dropped, total_by = Counter(), defaultdict(list), Counter()
stats, seen_q, arg_count, stage = Counter(), set(), Counter(), []
for r in rows:
    t = tools.get(r["tool"])
    if not t:
        stats["unknown_tool"] += 1; continue
    a, q = r["arguments"], r["query"].lower().strip()
    try:
        validate(a, t["inputSchema"])
    except ValidationError:
        stats["bad_schema"] += 1; continue
    if not paths_ok(a):
        stats["bad_path"] += 1; continue
    bad = ungrounded(a, q)
    if bad:
        stats["not_grounded"] += 1; gkeys[(r["tool"], bad[0])] += 1; continue
    kq, ka = (r["tool"], q), (r["tool"], json.dumps(a, sort_keys=True))
    if kq in seen_q or arg_count[ka] >= 5:
        stats["duplicate"] += 1; continue
    seen_q.add(kq); arg_count[ka] += 1; stage.append(r)
print(f"raw {len(rows)} -> after offline checks {len(stage)}  dropped: {dict(stats)}")
print("top ungrounded:", gkeys.most_common(8))
if args.dry:
    kept = Counter(r["tool"] for r in stage)
    for tl in sorted(total_by):
        print(f"{tl:<26} kept {kept[tl]:>3}/{total_by[tl]:>3}")
    for tl in ("edit_file", "read_multiple_files", "add_observations"):
        for qq, aa in dropped[tl][:3]:
            print(f"\n[{tl}] {qq}\n   {json.dumps(aa)[:200]}")
    sys.exit()

# ---------- stage 2: execute against the real servers ----------
SANDBOX.mkdir(exist_ok=True)
(SANDBOX / "memory.json").unlink(missing_ok=True)
SERVERS = {
    "filesystem": StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", str(SANDBOX)]),
    "tasks": StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "servers/tasks_server.py")]),
    "memory": StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-memory"],
        env={"MEMORY_FILE_PATH": str(SANDBOX / "memory.json")}),
}


async def main():
    clean, passed, failed, errs = [], Counter(), Counter(), defaultdict(list)
    async with AsyncExitStack() as stack:
        sess = {}
        for name, params in SERVERS.items():
            r, w = await stack.enter_async_context(stdio_client(params))
            s = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            sess[name] = s
        for i, r in enumerate(stage, 1):
            tool, a = r["tool"], r["arguments"]
            server = tools[tool]["server"]
            try:
                if server == "filesystem":
                    seed_fs(tool, a)
                elif server == "tasks":
                    seed_tasks(a)
                elif server == "memory" and tool != "create_entities":
                    ents = [{"name": n, "entityType": "item", "observations": ["seed"]}
                            for n in names_in(a)]
                    if ents:
                        await sess["memory"].call_tool("create_entities", {"entities": ents})
                res = await sess[server].call_tool(tool, rewrite(a))
                ok = not res.isError
                msg = res.content[0].text[:140] if res.content else ""
            except Exception as e:
                ok, msg = False, str(e)[:140]
            if ok:
                clean.append({"tool": tool, "query": r["query"], "arguments": a})
                passed[tool] += 1
            else:
                failed[tool] += 1
                if len(errs[tool]) < 2:
                    errs[tool].append(msg)
            if i % 200 == 0:
                print(f"  executed {i}/{len(stage)}")
    json.dump(clean, open(ROOT / "data/clean.json", "w"), indent=1)
    print(f"\nexecuted OK: {len(clean)}/{len(stage)}")
    for t in sorted(tools):
        if passed[t] or failed[t]:
            print(f"{t:<26} pass {passed[t]:>3}  fail {failed[t]:>3}")
    for t, m in errs.items():
        print(f"\nERRORS {t}:", *m, sep="\n  ")

asyncio.run(main())