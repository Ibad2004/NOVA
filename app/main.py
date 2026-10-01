import asyncio

from app.core.orchestrator import Orchestrator


async def main():
    nova = Orchestrator()

    print("NOVA is ready.")
    user_message = input("\nYou: ")

    response = await nova.handle(user_message)

    print("\nNOVA:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())