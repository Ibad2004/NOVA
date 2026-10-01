import asyncio
import ollama

from app.mcp.client import MCPClient


async def main():
    mcp = MCPClient()

    # 1. Discover MCP tools
    tools = await mcp.list_tools()

    ollama_tools = []

    for tool in tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })

    # 2. Ask the model
    messages = [
        {
            "role": "user",
            "content": "What is my current directory?",
        }
    ]

    response = ollama.chat(
        model="qwen3:4b-instruct",
        messages=messages,
        tools=ollama_tools,
    )

    assistant_message = response["message"]

    print("Model tool call:")
    print(assistant_message.tool_calls)

    # 3. Execute the requested MCP tool
    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print(f"\nCalling MCP tool: {tool_name}")

        result = await mcp.call_tool(
            tool_name,
            arguments,
        )

        tool_result = result.content[0].text

        print(f"Tool result: {tool_result}")

        # 4. Give the tool result back to the model
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "content": tool_result,
        })

    # 5. Ask the model for the final answer
    final_response = ollama.chat(
        model="qwen3:4b-instruct",
        messages=messages,
    )

    print("\nFinal NOVA response:")
    print(final_response["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())