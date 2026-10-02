import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "my_documents"
)


# --------------------------------------------------
# 3. Interactive RAG
# --------------------------------------------------

while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break


    # --------------------------------------------------
    # 4. Convert question into embedding
    # --------------------------------------------------

    question_embedding = model.encode(
        question
    ).tolist()


    # --------------------------------------------------
    # 5. Search ChromaDB
    # --------------------------------------------------
    /*
    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )
     

    # --------------------------------------------------
    # 6. Get retrieved documents
    # --------------------------------------------------

    documents = results["documents"][0]


    # --------------------------------------------------
    # 7. Build context
    # --------------------------------------------------

    context = "\n\n".join(
        documents
    )


    # --------------------------------------------------
    # 8. Build prompt for Llama
    # --------------------------------------------------

    prompt = f"""
You are a helpful RAG assistant.

Use the provided context as the primary source
when answering the user's question.

If the context contains the answer, use it.

You may also use your general knowledge when the
context does not contain enough information.

Give a clear and concise answer.

Do not explain the RAG process.
Do not mention the retrieved documents.
Do not mention the context.
Do not mention embeddings or vector databases.

Context:
{context}

Question:
{question}

Answer:
"""


    # --------------------------------------------------
    # 9. Ask Llama
    # --------------------------------------------------

    response = ollama.chat(

        model="llama3.2",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # --------------------------------------------------
    # 10. Display ONLY the answer
    # --------------------------------------------------

    print("\nAnswer:")
    print(
        response["message"]["content"]
    )