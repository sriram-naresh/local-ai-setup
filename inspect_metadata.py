import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "my_documents"
)


results = collection.get(
    limit=5,
    include=[
        "documents",
        "metadatas"
    ]
)


for i in range(len(results["documents"])):

    print("\n--- Chunk", i + 1, "---")

    print("Document:")
    print(results["documents"][i])

    print("\nMetadata:")
    print(results["metadatas"][i])