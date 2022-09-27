import cohere
import pandas as pd
from sklearn.decomposition import PCA


def add_embed_to_dataframe(df, X, name="Embed"):
    for idx, data in enumerate(X.T):
        df[f"{name}_{idx}"] = data
    return df


# Call the cohere client
co = cohere.Client("2A0mtCaVSr0HFCBShT3coreZ4kO0tUpGyqq11r7b")

# Load in the pokemon dataset
df = pd.read_csv("../data/pokemon.csv")

# Embed in the un-finetuned model
response_normal = co.embed(texts=df["description"].to_list())

# Embed all pokemon representations in the finetuned model
response_finetune = co.embed(
    texts=df["description"].to_list(), model="78d3e399-16a7-4264-b3fc-62072ce2c882-ft"
)


# PCA on finetuned
pca = PCA(n_components=2)  # For visualization purposes
X_finetuned = pca.fit_transform(response_finetune.embeddings)
X_normal = pca.fit_transform(response_normal.embeddings)

# Save to dataframe
df = add_embed_to_dataframe(df, X_finetuned, name="Embed_Finetune")
df = add_embed_to_dataframe(df, X_normal, name="Embed")
df.to_csv("pokemon_embed.csv")
