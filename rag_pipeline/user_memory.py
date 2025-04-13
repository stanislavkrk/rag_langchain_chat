import re
from typing import List


class UserMemory:
    """
    Stores user's preferences, such as favorite ingredients.
    """

    def __init__(self):
        self.favorite_ingredients: List[str] = []

    def add_ingredients_from_text(self, text: str) -> None:
        """
        Extracts and stores favorite ingredients from a user message.

        Example trigger phrases:
        - "I like rum and lemon"
        - "My favorite ingredients are mint, sugar, and gin"

        :param text: User input.
        """
        # Simple rule-based detection
        match = re.search(r"(?:like|favorite.*are)\s(.+)", text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            # Split by comma or "and"
            parts = re.split(r",|\band\b", raw)
            for item in parts:
                ingredient = item.strip().lower()
                if ingredient and ingredient not in self.favorite_ingredients:
                    self.favorite_ingredients.append(ingredient)

    def get_favorites(self) -> List[str]:
        """
        Return list of known favorite ingredients.
        """
        return self.favorite_ingredients

    def clear(self) -> None:
        """
        Clear all stored ingredients.
        """
        self.favorite_ingredients = []
