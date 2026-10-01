class ModelRouter:
    def should_think(self, user_message: str) -> bool:
        message = user_message.lower()

        complex_keywords = [
            "analyze",
            "analyse",
            "explain",
            "design",
            "compare",
            "debug",
            "architecture",
            "why",
            "reason",
            "evaluate",
        ]

        return any(keyword in message for keyword in complex_keywords)