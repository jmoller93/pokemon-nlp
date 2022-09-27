import pandas as pd

"""
This script processes the pokemon dataset to get only 500 samples of primary-typing - description pairings 
"""
# Map type to numeric label
dict = {
    "water": 0,
    "normal": 1,
    "grass": 2,
    "bug": 3,
    "psychic": 4,
    "electric": 5,
    "fire": 6,
    "dark": 7,
    "rock": 8,
    "steel": 9,
    "dragon": 10,
    "fairy": 11,
    "ice": 12,
    "ground": 13,
    "ghost": 14,
    "fighting": 15,
    "poison": 16,
    "flying": 17,
}

# Load in the file
file = "../data/pokemon.csv"

# Create dataframe of interest
df = (
    pd.read_csv(file, index_col=0)
    .filter(regex="|".join(["primary", "description", "english"]))
    .sample(n=750, random_state=42)
    .replace({"primary_type": dict})
)

# Save into train-test splits
df_train = df.iloc[:500][["description", "primary_type"]]
print(df_train["primary_type"].value_counts())
df_train.to_csv("pokemon_train.csv", index=False)
df_test = df.iloc[500:][["description", "primary_type"]]
print(df_test["primary_type"].value_counts())
df_test.to_csv("pokemon_test.csv", index=False)

# One can see by printing out the train test splits, only flying type gets thrown out. Everything else is kept.
