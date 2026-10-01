import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["app/mcp/computer/server.py"],
        )

    async def list_tools(self):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.list_tools()

                return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                return result


if __name__ == "__main__":
    client = MCPClient()

    tools = asyncio.run(client.list_tools())

    print("Available MCP tools:")

    for tool in tools:
        print(f"- {tool.name}")