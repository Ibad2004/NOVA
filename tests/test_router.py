from app.core.router import ModelRouter


router = ModelRouter()


test_messages = [
    "What is the capital of Pakistan?",
    "Explain why a machine learning model can overfit.",
    "Open my project folder.",
    "What is Python?",
    "Debug this Python code.",
    "What is 25 multiplied by 8?",
]


for message in test_messages:
    think = router.should_think(message)

    print(f"\nMessage: {message}")
    print(f"Router decision: think={think}")