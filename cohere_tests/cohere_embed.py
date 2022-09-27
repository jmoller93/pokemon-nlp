import cohere
from sklearn.decomposition import PCA

# Call the cohere client
co = cohere.Client("2A0mtCaVSr0HFCBShT3coreZ4kO0tUpGyqq11r7b")

# Embed texts to a response
response = co.embed(
    texts=["dog", "cat", "horse", "elephant", "alligator", "toucan", "tiger", "lion"]
)

pca = PCA(n_components=2)
X_embed_pca = pca.fit_transform(response.embeddings)

# Print out embeddings
print(f"Embeddings: {X_embed_pca}")
