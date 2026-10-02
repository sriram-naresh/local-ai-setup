import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Connect to local ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "my_documents"
)


# --------------------------------------------------
# 3. Interactive RAG loop
# --------------------------------------------------

while True:

    question = input("\nAsk a question: ")

    # Exit
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

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=8,

        include=[
            "documents",
            "metadatas"
        ]
    )


    # --------------------------------------------------
    # 6. Get retrieved documents and metadata
    # --------------------------------------------------

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]


    # --------------------------------------------------
    # 7. Build context for Llama
    # --------------------------------------------------

    context = "\n\n".join(
        documents
    )


    # --------------------------------------------------
    # 8. Build prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful question-answering assistant.

Use the provided context as the primary source
when answering the user's question.

You may also use your general knowledge when the
context does not contain enough information.

IMPORTANT:
- Answer the user's question directly.
- Do NOT ask the user a question.
- Do NOT repeat the user's question.
- Do NOT describe your reasoning.
- Do NOT mention RAG.
- Do NOT mention embeddings.
- Do NOT mention ChromaDB.
- Give only the final answer.
- Keep the answer concise.

Context:
{context}

User question:
{question}

Final answer:
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
    # 10. Get final answer
    # --------------------------------------------------

    answer = response[
        "message"
    ][
        "content"
    ]


    # --------------------------------------------------
    # 11. Display answer
    # --------------------------------------------------

    print("\nAnswer:")
    print(answer)


    # --------------------------------------------------
    # 12. Display sources
    # --------------------------------------------------

    sources = []

    for metadata in metadatas:

        source = metadata.get(
            "source"
        )

        if source and source not in sources:
            sources.append(source)


    if sources:

        print("\nSources:")

        for source in sources:
            print(f"- {source}")