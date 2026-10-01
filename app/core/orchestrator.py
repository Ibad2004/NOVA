import os
import ollama

from app.llm.ollama_client import OllamaClient
from app.core.router import ModelRouter
from app.mcp.client import MCPClient


class Orchestrator:
    def __init__(self):
        self.llm = OllamaClient()
        self.router = ModelRouter()
        self.mcp = MCPClient()

        self.max_tool_steps = int(
            os.getenv("MAX_TOOL_STEPS", "5")
        )

        self.max_tool_retries = int(
            os.getenv("MAX_TOOL_RETRIES", "2")
        )

    async def handle(self, user_message: str) -> str:
        think = self.router.should_think(user_message)

        # 1. Discover available MCP tools
        tools = await self.mcp.list_tools()

        # 2. Convert MCP tools to Ollama format
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

        # 3. Start conversation
        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]

        # 4. Bounded tool-calling loop
        for step in range(self.max_tool_steps):

            print(
                f"\n--- Tool Step "
                f"{step + 1}/{self.max_tool_steps} ---"
            )

            response = ollama.chat(
                model=self.llm.model,
                messages=messages,
                tools=ollama_tools,
                think=think,
                options={
                    "temperature": self.llm.temperature,
                    "num_predict": self.llm.max_tokens,
                },
            )

            assistant_message = response["message"]

            
            print(assistant_message.tool_calls)

            # 5. Model has finished using tools
            if not assistant_message.tool_calls:
                return assistant_message["content"]

            # 6. Preserve assistant tool-call message
            messages.append(assistant_message)

            # 7. Execute requested tools
            for tool_call in assistant_message.tool_calls:

                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                print(f"Calling MCP tool: {tool_name}")
                print(f"Arguments: {arguments}")

                tool_result = None

                # Retry failed tool calls
                for attempt in range(
                    self.max_tool_retries + 1
                ):

                    try:
                        result = await self.mcp.call_tool(
                            tool_name,
                            arguments,
                        )

                        tool_result = result.content[0].text

                        print(
                            f"Tool succeeded on attempt "
                            f"{attempt + 1}"
                        )

                        break

                    except Exception as error:

                        print(
                            f"Tool attempt {attempt + 1} failed: "
                            f"{error}"
                        )

                        if attempt == self.max_tool_retries:
                            tool_result = (
                                f"Tool execution failed after "
                                f"{self.max_tool_retries + 1} "
                                f"attempts: {error}"
                            )

                print(f"Tool result: {tool_result}")

                # 8. Give result back to LLM
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                })

        # 9. Safety stop
        return (
            "I stopped because the maximum number of tool steps "
            f"({self.max_tool_steps}) was reached."
        )