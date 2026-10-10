
import asyncio

from app.core.orchestrator import Orchestrator


async def main():
    nova = Orchestrator()

    response = await nova.handle(
        "Open data/workspace/agent_test.txt and append "
        "this exact text at the end: "
        "Write tool test passed."
    )

    print("\nNOVA:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
