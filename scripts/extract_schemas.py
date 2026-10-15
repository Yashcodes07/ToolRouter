import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {
    "filesystem": StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "./sandbox"]),
    "tasks": StdioServerParameters(
        command=sys.executable, args=["servers/tasks_server.py"]),
    "memory": StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-memory"]),
}


async def main():
    tools = []
    for name, params in SERVERS.items():
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    found = (await s.list_tools()).tools
                    for t in found:
                        tools.append({"server": name, "name": t.name,
                                      "description": t.description,
                                      "inputSchema": t.inputSchema})
                    print(f"{name}: {len(found)} tools")
        except Exception as e:
            print(f"{name}: FAILED -> {e}")
    json.dump(tools, open("data/tools.json", "w"), indent=2)
    print("total:", len(tools), "tools saved to data/tools.json")


asyncio.run(main())