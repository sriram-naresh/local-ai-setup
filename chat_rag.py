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
# 3. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.get_collection(
    "my_documents"
)


# ==================================================
# 4. CONVERSATION HISTORY
# ==================================================

conversation_history = []


# ==================================================
# 5. INTERACTIVE CHAT LOOP
# ==================================================

while True:

    question = input(
        "\nAsk a question: "
    )


    # ==================================================
    # 6. EXIT
    # ==================================================

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # ==================================================
    # 7. BUILD QUERY FOR RETRIEVAL
    # ==================================================
    #
    # We use the current question for vector search.
    #
    # Later, we can add a query-rewriting step so
    # follow-up questions such as:
    #
    # "What about storage?"
    #
    # can become:
    #
    # "What storage options does Amazon EKS support?"
    #
    # ==================================================

    retrieval_query = question


    # ==================================================
    # 8. CREATE QUESTION EMBEDDING
    # ==================================================

    question_embedding = embedding_model.encode(
        retrieval_query
    ).tolist()


    # ==================================================
    # 9. RETRIEVE CANDIDATES FROM CHROMADB
    # ==================================================

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=10,

        include=[
            "documents",
            "metadatas"
        ]

    )


    # ==================================================
    # 10. GET DOCUMENTS
    # ==================================================

    documents = results[
        "documents"
    ][0]


    metadatas = results[
        "metadatas"
    ][0]


    # ==================================================
    # 11. CREATE QUESTION-DOCUMENT PAIRS
    # ==================================================

    pairs = [

        [
            retrieval_query,
            document
        ]

        for document in documents

    ]


    # ==================================================
    # 12. RERANK
    # ==================================================

    scores = reranker.predict(
        pairs
    )


    # ==================================================
    # 13. COMBINE RESULTS
    # ==================================================

    reranked_results = list(

        zip(
            documents,
            metadatas,
            scores
        )

    )


    # ==================================================
    # 14. SORT BY RERANKER SCORE
    # ==================================================

    reranked_results.sort(

        key=lambda x: x[2],

        reverse=True

    )


    # ==================================================
    # 15. SELECT TOP 3
    # ==================================================

    top_results = (
        reranked_results[:3]
    )


    # ==================================================
    # 16. BUILD SOURCE-AWARE CONTEXT
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
    # 17. BUILD CONVERSATION HISTORY TEXT
    # ==================================================

    history_text = ""


    if conversation_history:

        history_parts = []


        for item in conversation_history:

            history_parts.append(

                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}"

            )


        history_text = "\n\n".join(
            history_parts
        )


    else:

        history_text = (
            "No previous conversation."
        )


    # ==================================================
    # 18. BUILD PROMPT
    # ==================================================

    prompt = f"""
You are a helpful conversational question-answering assistant.

Answer the user's current question using the provided
source documents and conversation history.

IMPORTANT RULES:

- Use the source documents as the primary source of truth.
- Use conversation history to understand follow-up questions.
- Prefer information explicitly stated in the source documents.
- Do not invent specific facts.
- Do not contradict information explicitly stated in the sources.
- If the source documents do not contain enough information,
  you may use general knowledge when appropriate.
- If the question refers to something from the previous
  conversation, resolve that reference using the history.
- Answer the current question directly.
- Do not repeat the user's question.
- Do not describe your reasoning.
- Do not ask the user a question.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention ChromaDB.
- Do not mention the reranker.
- Keep the answer concise.

CONVERSATION HISTORY:

{history_text}

SOURCE DOCUMENTS:

{context}

CURRENT USER QUESTION:

{question}

FINAL ANSWER:
"""


    # ==================================================
    # 19. SEND TO LLAMA
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
    # 20. GET ANSWER
    # ==================================================

    answer = response[
        "message"
    ][
        "content"
    ]


    # ==================================================
    # 21. DISPLAY ANSWER
    # ==================================================

    print(
        "\nAnswer:"
    )

    print(
        answer
    )


    # ==================================================
    # 22. DISPLAY SOURCES
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


    # ==================================================
    # 23. SAVE THIS TURN TO CONVERSATION HISTORY
    # ==================================================

    conversation_history.append(

        {
            "question": question,
            "answer": answer
        }

    )


    # ==================================================
    # 24. LIMIT HISTORY
    # ==================================================
    #
    # Keep the last 5 turns.
    #
    # This prevents the prompt from becoming
    # unnecessarily large during a long conversation.
    #
    # ==================================================

    if len(
        conversation_history
    ) > 5:

        conversation_history = (
            conversation_history[-5:]
        )