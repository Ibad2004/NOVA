import asyncio

from app.core.orchestrator import Orchestrator


async def main():
    nova = Orchestrator()

    tools = await nova.mcp.list_tools()

    print("\nMCP TOOLS:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    response = await nova.handle(
        "Tell me the names of the files and folders in my current folder."
    )

    print("\nNOVA:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())