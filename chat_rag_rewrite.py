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
# 5. FUNCTION: REWRITE USER QUESTION
# ==================================================

def rewrite_question(question, history):

    # ----------------------------------------------
    # No history
    # ----------------------------------------------

    if not history:

        return question


    # ----------------------------------------------
    # Build history
    # ----------------------------------------------

    history_parts = []


    for item in history:

        history_parts.append(

            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"

        )


    history_text = "\n\n".join(
        history_parts
    )


    # ----------------------------------------------
    # Rewrite prompt
    # ----------------------------------------------

    prompt = f"""
Rewrite the user's latest question into a standalone
search query that can be used to search a document
knowledge base.

Use the conversation history to resolve references
such as:

- it
- this
- that
- they
- them
- which one
- what about it
- the above
- the previous one

IMPORTANT:

- Preserve the user's original intent.
- Do not answer the question.
- Do not add unrelated information.
- Do not explain your reasoning.
- Return ONLY the rewritten search query.
- If the question is already standalone, return it
  unchanged.

Conversation history:

{history_text}

Latest user question:

{question}

Standalone search query:
"""


    # ----------------------------------------------
    # Ask local Llama to rewrite
    # ----------------------------------------------

    response = ollama.chat(

        model="llama3.2",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    rewritten = response[
        "message"
    ][
        "content"
    ].strip()


    # ----------------------------------------------
    # Safety fallback
    # ----------------------------------------------

    if not rewritten:

        return question


    return rewritten


# ==================================================
# 6. INTERACTIVE CHAT LOOP
# ==================================================

while True:

    question = input(
        "\nAsk a question: "
    )


    # ==================================================
    # 7. EXIT
    # ==================================================

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # ==================================================
    # 8. REWRITE QUESTION
    # ==================================================

    rewritten_query = rewrite_question(
        question,
        conversation_history
    )


    # ==================================================
    # 9. SHOW REWRITTEN QUERY
    # ==================================================

    print(
        "\nSearch query:"
    )

    print(
        rewritten_query
    )


    # ==================================================
    # 10. CREATE EMBEDDING
    # ==================================================

    question_embedding = embedding_model.encode(
        rewritten_query
    ).tolist()


    # ==================================================
    # 11. RETRIEVE FROM CHROMADB
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
    # 12. GET DOCUMENTS
    # ==================================================

    documents = results[
        "documents"
    ][0]


    metadatas = results[
        "metadatas"
    ][0]


    # ==================================================
    # 13. CREATE RERANKING PAIRS
    # ==================================================

    pairs = [

        [
            rewritten_query,
            document
        ]

        for document in documents

    ]


    # ==================================================
    # 14. RERANK
    # ==================================================

    scores = reranker.predict(
        pairs
    )


    # ==================================================
    # 15. COMBINE RESULTS
    # ==================================================

    reranked_results = list(

        zip(
            documents,
            metadatas,
            scores
        )

    )


    # ==================================================
    # 16. SORT
    # ==================================================

    reranked_results.sort(

        key=lambda x: x[2],

        reverse=True

    )


    # ==================================================
    # 17. SELECT TOP 3
    # ==================================================

    top_results = (
        reranked_results[:3]
    )


    # ==================================================
    # 18. BUILD SOURCE-AWARE CONTEXT
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
    # 19. BUILD CONVERSATION HISTORY
    # ==================================================

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
    # 20. BUILD ANSWER PROMPT
    # ==================================================

    prompt = f"""
You are a helpful conversational question-answering assistant.

Answer the user's current question using the provided
source documents and conversation history.

IMPORTANT RULES:

- Use the source documents as the primary source of truth.
- Use conversation history to understand the current question.
- Prefer information explicitly stated in the source documents.
- Do not invent specific facts.
- Do not contradict information explicitly stated in the sources.
- If the source documents do not contain enough information,
  say so clearly.
- Answer the current question directly.
- Do not repeat the user's question.
- Do not describe your reasoning.
- Do not ask the user a question.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention ChromaDB.
- Do not mention the reranker.
- Do not mention query rewriting.
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
    # 21. ASK LLAMA FOR FINAL ANSWER
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
    # 22. GET ANSWER
    # ==================================================

    answer = response[
        "message"
    ][
        "content"
    ]


    # ==================================================
    # 23. DISPLAY ANSWER
    # ==================================================

    print(
        "\nAnswer:"
    )

    print(
        answer
    )


    # ==================================================
    # 24. DISPLAY SOURCES
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
    # 25. SAVE CONVERSATION
    # ==================================================

    conversation_history.append(

        {
            "question": question,
            "answer": answer
        }

    )


    # ==================================================
    # 26. KEEP LAST 5 TURNS
    # ==================================================

    if len(
        conversation_history
    ) > 5:

        conversation_history = (
            conversation_history[-5:]
        )