import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load reranker model
# --------------------------------------------------

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "my_documents"
)


# --------------------------------------------------
# 4. Ask question
# --------------------------------------------------

question = input(
    "\nEnter a question: "
)


# --------------------------------------------------
# 5. Convert question into embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    question
).tolist()


# --------------------------------------------------
# 6. Retrieve 15 candidates from ChromaDB
# --------------------------------------------------

results = collection.query(

    query_embeddings=[
        question_embedding
    ],

    n_results=15,

    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# --------------------------------------------------
# 7. Extract results
# --------------------------------------------------

documents = results["documents"][0]

metadatas = results["metadatas"][0]

distances = results["distances"][0]


# --------------------------------------------------
# 8. Show ChromaDB results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("CHROMADB RESULTS")
print("=" * 70)


for number, (
    document,
    metadata,
    distance
) in enumerate(
    zip(
        documents,
        metadatas,
        distances
    ),
    start=1
):

    print(f"\n--- Chroma Rank {number} ---")

    print("Distance:")
    print(distance)

    print("\nSource:")
    print(metadata.get("source"))

    print("\nDocument:")
    print(document)


# --------------------------------------------------
# 9. Create question-document pairs
# --------------------------------------------------

pairs = [
    [question, document]
    for document in documents
]


# --------------------------------------------------
# 10. Run reranker
# --------------------------------------------------

print("\n")
print("Calculating reranker scores...")


scores = reranker.predict(
    pairs
)


# --------------------------------------------------
# 11. Combine all information
# --------------------------------------------------

combined = list(
    zip(
        documents,
        metadatas,
        distances,
        scores
    )
)


# --------------------------------------------------
# 12. Sort by reranker score
# --------------------------------------------------

combined.sort(
    key=lambda x: x[3],
    reverse=True
)


# --------------------------------------------------
# 13. Show reranked results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("RERANKED RESULTS")
print("=" * 70)


for number, (
    document,
    metadata,
    distance,
    score
) in enumerate(
    combined,
    start=1
):

    print(f"\n--- Reranker Rank {number} ---")

    print("Reranker score:")
    print(score)

    print("Original Chroma distance:")
    print(distance)

    print("\nSource:")
    print(metadata.get("source"))

    print("\nDocument:")
    print(document)


# --------------------------------------------------
# 14. Show the final top 3
# --------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL TOP 3 FOR LLAMA")
print("=" * 70)


for number, (
    document,
    metadata,
    distance,
    score
) in enumerate(
    combined[:3],
    start=1
):

    print(f"\n--- Final Result {number} ---")

    print("Reranker score:")
    print(score)

    print("Source:")
    print(metadata.get("source"))

    print("\nDocument:")
    print(document)


print("\n")
print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)