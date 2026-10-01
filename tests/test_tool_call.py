import ollama


def get_current_directory() -> str:
    return "D:\\6thsemesterstudy\\NOVA"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_directory",
            "description": "Get the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]


response = ollama.chat(
    model="qwen3:4b-instruct",
    messages=[
        {
            "role": "user",
            "content": "What is my current directory?",
        }
    ],
    tools=tools,
)

print("Full response:")
print(response)

print("\nMessage:")
print(response["message"])

print("\nTool calls:")
print(response["message"].get("tool_calls"))