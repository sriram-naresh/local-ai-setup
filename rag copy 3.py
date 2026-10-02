import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama


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
# 3. Connect to local ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "my_documents"
)


# --------------------------------------------------
# 4. Interactive RAG loop
# --------------------------------------------------

while True:

    question = input("\nAsk a question: ")


    # --------------------------------------------------
    # Exit
    # --------------------------------------------------

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # --------------------------------------------------
    # 5. Convert question into embedding
    # --------------------------------------------------

    question_embedding = embedding_model.encode(
        question
    ).tolist()


    # --------------------------------------------------
    # 6. Retrieve candidate documents from ChromaDB
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        # Retrieve more candidates first
        n_results=10,

        include=[
            "documents",
            "metadatas"
        ]
    )


    # --------------------------------------------------
    # 7. Get candidates
    # --------------------------------------------------

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]


    # --------------------------------------------------
    # 8. Create question-document pairs
    # --------------------------------------------------

    pairs = [
        [question, document]
        for document in documents
    ]


    # --------------------------------------------------
    # 9. Rerank candidates
    # --------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------
    # 10. Combine documents, metadata and scores
    # --------------------------------------------------

    reranked_results = list(
        zip(
            documents,
            metadatas,
            scores
        )
    )


    # --------------------------------------------------
    # 11. Sort by reranker score
    # --------------------------------------------------

    reranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )


    # --------------------------------------------------
    # 12. Keep only the best 3 chunks
    # --------------------------------------------------

    top_results = reranked_results[:3]


    # --------------------------------------------------
    # 13. Build context for Llama
    # --------------------------------------------------

    context = "\n\n".join(

        document
        for document, metadata, score
        in top_results

    )


    # --------------------------------------------------
    # 14. Build prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful question-answering assistant.

Use the provided context as the primary source
for answering the user's question.

IMPORTANT RULES:

- Answer the user's question directly.
- Prefer facts explicitly present in the context.
- Do not contradict facts stated in the context.
- Do not invent specific facts about the user.
- If the context does not contain enough information,
  you may use general knowledge.
- Do not ask the user a question.
- Do not repeat the user's question.
- Do not describe your reasoning.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention ChromaDB.
- Do not mention the reranker.
- Give only the final answer.
- Keep the answer concise.

Context:
{context}

User question:
{question}

Final answer:
"""


    # --------------------------------------------------
    # 15. Ask Llama
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
    # 16. Get final answer
    # --------------------------------------------------

    answer = response[
        "message"
    ][
        "content"
    ]


    # --------------------------------------------------
    # 17. Display answer
    # --------------------------------------------------

    print("\nAnswer:")

    print(answer)


    # --------------------------------------------------
    # 18. Display sources
    # --------------------------------------------------

    sources = []


    for document, metadata, score in top_results:

        source = metadata.get(
            "source"
        )


        if source and source not in sources:

            sources.append(
                source
            )


    if sources:

        print("\nSources:")


        for source in sources:

            print(
                f"- {source}"
            )