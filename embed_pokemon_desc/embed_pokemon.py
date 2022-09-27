import cohere
from sklearn.decomposition import PCA

# Call the cohere client
co = cohere.Client('2A0mtCaVSr0HFCBShT3coreZ4kO0tUpGyqq11r7b')

# Load in the pokemon dataset
df = pd.read_csv("../data/pokemon.csv")

# Embed all pokemon representations in the finetuned model
response_finetune = co.embed(texts=df["description"].to_list(),
                    model='78d3e399-16a7-4264-b3fc-62072ce2c882-ft')

# Embed in the un-finetuned model
response_normal = co.embed(texts=df["description"].to_list())

# PCA on finetuned
pca = PCA(n_components=2) # For visualization purposes
X_finetuned = pca.fit_transform(response_finetune.embeddings)
X_normal = pca.fit_transform(response_normal.embeddings)

# Save to dataframe

