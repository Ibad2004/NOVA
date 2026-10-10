
import asyncio

from app.core.orchestrator import Orchestrator


async def main():
    nova = Orchestrator()

    response = await nova.handle(
        "Create a new file named agent_test.txt inside "
        "data/workspace. Write exactly this text inside it: "
        "NOVA agent file creation test passed."
    )

    print("\nNOVA:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
