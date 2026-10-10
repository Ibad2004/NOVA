
import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["app/mcp/computer/server.py"],
        )
        self.session = None
        self._stack = None

    async def __aenter__(self):
        if self.session is not None:
            raise RuntimeError("MCP client is already connected.")

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        try:
            read, write = await self._stack.enter_async_context(
                stdio_client(self.server_params)
            )

            self.session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )

            await self.session.initialize()
            return self

        except BaseException:
            await self._stack.aclose()
            self._stack = None
            self.session = None
            raise

    async def __aexit__(self, exc_type, exc, traceback):
        stack = self._stack
        self._stack = None
        self.session = None

        if stack is not None:
            return await stack.__aexit__(
                exc_type, exc, traceback
            )

        return False

    async def list_tools(self):
        if self.session is None:
            async with MCPClient() as client:
                return await client.list_tools()

        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        if self.session is None:
            async with MCPClient() as client:
                return await client.call_tool(
                    tool_name, arguments
                )

        return await self.session.call_tool(
            tool_name,
            arguments=arguments,
        )


if __name__ == "__main__":
    async def main():
        async with MCPClient() as client:
            tools = await client.list_tools()

            print("Available MCP tools:")
            for tool in tools:
                print(f"- {tool.name}")

    asyncio.run(main())
