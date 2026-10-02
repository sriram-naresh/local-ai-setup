from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Kubernetes is a container orchestration platform."

# Convert text into an embedding
embedding = model.encode(text)

print("Text:")
print(text)

print("\nEmbedding type:")
print(type(embedding))

print("\nEmbedding dimensions:")
print(len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])