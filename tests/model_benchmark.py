import time

from app.llm.ollama_client import OllamaClient


client = OllamaClient()


def run_test(name: str, prompt: str, think: bool):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Thinking: {think}")
    print(f"Prompt: {prompt}")

    start_time = time.perf_counter()

    response = client.chat(
        prompt,
        think=think,
    )

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print(f"\nTime: {elapsed:.2f} seconds")
    print("\nResponse:")
    print(response)

    return {
        "name": name,
        "think": think,
        "time": elapsed,
        "response": response,
    }


def main():
    print("=" * 60)
    print("NOVA MODEL BENCHMARK")
    print("=" * 60)

    model_name = client.model
    print(f"Model: {model_name}")

    results = []

    results.append(
        run_test(
            "TEST 1 — Basic Response",
            "What is the capital of Pakistan? Answer in one sentence.",
            False,
        )
    )

    results.append(
        run_test(
            "TEST 2 — Reasoning WITHOUT Thinking",
            "Explain why a machine learning model can overfit. Give three simple reasons.",
            False,
        )
    )

    results.append(
        run_test(
            "TEST 3 — Reasoning WITH Thinking",
            "Explain why a machine learning model can overfit and give three ways to prevent it.",
            True,
        )
    )

    results.append(
        run_test(
            "TEST 4 — Instruction Following",
            "Give exactly five Python programming tips. Use exactly five numbered points.",
            False,
        )
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print(f"Model: {model_name}")

    for result in results:
        print(
            f"{result['name']}: "
            f"{result['time']:.2f}s | "
            f"thinking={result['think']}"
        )


if __name__ == "__main__":
    main()