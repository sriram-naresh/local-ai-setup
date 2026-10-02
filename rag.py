import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama


# ==================================================
# 1. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 2. LOAD RERANKER MODEL
# ==================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# 3. CONNECT TO LOCAL CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.get_collection(
    "my_documents"
)


# ==================================================
# 4. INTERACTIVE RAG LOOP
# ==================================================

while True:

    question = input(
        "\nAsk a question: "
    )


    # ==================================================
    # 5. EXIT
    # ==================================================

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # ==================================================
    # 6. CONVERT QUESTION INTO EMBEDDING
    # ==================================================

    question_embedding = embedding_model.encode(
        question
    ).tolist()


    # ==================================================
    # 7. RETRIEVE CANDIDATES FROM CHROMADB
    # ==================================================

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


    # ==================================================
    # 8. GET DOCUMENTS AND METADATA
    # ==================================================

    documents = results[
        "documents"
    ][0]


    metadatas = results[
        "metadatas"
    ][0]


    # ==================================================
    # 9. CREATE QUESTION-DOCUMENT PAIRS
    # ==================================================

    pairs = [

        [
            question,
            document
        ]

        for document in documents

    ]


    # ==================================================
    # 10. RERANK DOCUMENTS
    # ==================================================

    scores = reranker.predict(
        pairs
    )


    # ==================================================
    # 11. COMBINE DOCUMENTS + METADATA + SCORE
    # ==================================================

    reranked_results = list(

        zip(
            documents,
            metadatas,
            scores
        )

    )


    # ==================================================
    # 12. SORT BY RERANKER SCORE
    # ==================================================

    reranked_results.sort(

        key=lambda x: x[2],

        reverse=True

    )


    # ==================================================
    # 13. SELECT BEST 3 RESULTS
    # ==================================================

    top_results = (
        reranked_results[:3]
    )


    # ==================================================
    # 14. BUILD SOURCE-AWARE CONTEXT
    # ==================================================

    context_parts = []


    for document, metadata, score in top_results:

        source = metadata.get(
            "source",
            "Unknown source"
        )


        context_parts.append(

            f"SOURCE: {source}\n"
            f"{document}"

        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # ==================================================
    # 15. BUILD PROMPT
    # ==================================================

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using the provided source documents.

IMPORTANT RULES:

- Use the source documents as the primary source of truth.
- Prefer information explicitly stated in the sources.
- Do not invent specific facts.
- If multiple sources contain the same information, combine them.
- Do not contradict information explicitly stated in the sources.
- If the source does not contain enough information, say so clearly.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention ChromaDB.
- Do not mention the reranker.
- Do not describe your reasoning.
- Do not ask the user a question.
- Answer directly and concisely.

SOURCE DOCUMENTS:

{context}

USER QUESTION:

{question}

FINAL ANSWER:
"""


    # ==================================================
    # 16. SEND REQUEST TO LLAMA
    # ==================================================

    response = ollama.chat(

        model="llama3.2",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    # ==================================================
    # 17. GET FINAL ANSWER
    # ==================================================

    answer = response[
        "message"
    ][
        "content"
    ]


    # ==================================================
    # 18. DISPLAY ANSWER
    # ==================================================

    print(
        "\nAnswer:"
    )

    print(
        answer
    )


    # ==================================================
    # 19. DISPLAY SOURCES
    # ==================================================

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

        print(
            "\nSources:"
        )


        for source in sources:

            print(
                f"- {source}"
            )