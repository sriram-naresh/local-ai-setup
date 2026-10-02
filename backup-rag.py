import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get collection
collection = client.get_collection("my_documents")


while True:

    # Get question from user
    question = input("\nAsk a question: ")

    # Exit option
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert question into embedding
    question_embedding = model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2
    )

    # Get retrieved documents
    documents = results["documents"][0]

    # Combine documents into context
    context = "\n\n".join(documents)

    # Create prompt
    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Send to Llama
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Display answer
    print("\nAnswer:")
    print(response["message"]["content"])