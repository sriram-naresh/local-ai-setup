import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to our local ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get our collection
collection = client.get_collection("my_documents")

# Ask a question
question = "What is EBS?"

# Convert the question into an embedding
question_embedding = model.encode(question).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

# Display results
print("\nQuestion:")
print(question)

print("\nRetrieved chunks:")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(document)