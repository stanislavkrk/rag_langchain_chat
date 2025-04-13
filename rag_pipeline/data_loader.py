import pandas as pd
import ast
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Cocktail:
    name: str
    alcoholic: str
    category: str
    glass: str
    instructions: str
    ingredients: List[str]
    measures: List[str]


class CocktailLoader:
    """
    Loads cocktail data from CSV and parses it into structured Cocktail objects.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> List[Cocktail]:
        """
        Load and parse the cocktail dataset from CSV.

        :return: List of Cocktail objects.
        """
        df = pd.read_csv(self.csv_path)

        cocktails = []

        for _, row in df.iterrows():
            try:
                ingredients = ast.literal_eval(row["ingredients"]) if pd.notna(row["ingredients"]) else []
                measures = ast.literal_eval(row["ingredientMeasures"]) if pd.notna(row["ingredientMeasures"]) else []
                cocktail = Cocktail(
                    name=row["name"],
                    alcoholic=row["alcoholic"],
                    category=row["category"],
                    glass=row["glassType"],
                    instructions=row["instructions"],
                    ingredients=ingredients,
                    measures=measures,
                )
                cocktails.append(cocktail)
            except Exception as e:
                # Skip invalid rows
                continue

        return cocktails
