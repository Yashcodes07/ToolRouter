import sqlite3
from pathlib import Path
from typing import Literal
from mcp.server.fastmcp import FastMCP

DB = Path(__file__).resolve().parent.parent / "sandbox" / "tasks.db"
mcp = FastMCP("tasks")
Priority = Literal["low", "medium", "high"]


def _db():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY, title TEXT, priority TEXT DEFAULT 'medium',
        due TEXT, done INTEGER DEFAULT 0)""")
    return c


@mcp.tool()
def add_task(title: str, priority: Priority = "medium", due: str | None = None) -> dict:
    """Create a task. due is an ISO date such as 2026-11-01."""
    with _db() as c:
        cur = c.execute("INSERT INTO tasks(title, priority, due) VALUES (?,?,?)",
                        (title, priority, due))
        return {"id": cur.lastrowid, "title": title, "priority": priority, "due": due}


@mcp.tool()
def list_tasks(status: Literal["open", "done", "all"] = "open",
               priority: Priority | None = None) -> dict:
    """List tasks, optionally filtered by status and priority."""
    q, args = "SELECT * FROM tasks WHERE 1=1", []
    if status != "all":
        q += " AND done=?"; args.append(1 if status == "done" else 0)
    if priority:
        q += " AND priority=?"; args.append(priority)
    with _db() as c:
        return {"tasks": [dict(r) for r in c.execute(q, args)]}


@mcp.tool()
def complete_task(task_id: int) -> dict:
    """Mark a task as done."""
    with _db() as c:
        n = c.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,)).rowcount
    if not n:
        raise ValueError(f"No task with id {task_id}")
    return {"completed": task_id}


@mcp.tool()
def delete_task(task_id: int) -> dict:
    """Permanently delete a task."""
    with _db() as c:
        n = c.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount
    if not n:
        raise ValueError(f"No task with id {task_id}")
    return {"deleted": task_id}


@mcp.tool()
def search_tasks(keyword: str) -> dict:
    """Find tasks whose title contains a keyword."""
    with _db() as c:
        rows = c.execute("SELECT * FROM tasks WHERE title LIKE ?", (f"%{keyword}%",))
        return {"tasks": [dict(r) for r in rows]}


if __name__ == "__main__":
    mcp.run()