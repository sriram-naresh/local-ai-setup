import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama


# ============================================================
# CONFIG
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2"

RETRIEVAL_TOP_K = 15
FINAL_TOP_K = 5
MAX_HISTORY = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Connecting to Chroma...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("RAG system ready.")


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = []


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(question):

    if not conversation_history:

        return question

    recent_history = conversation_history[
        -MAX_HISTORY:
    ]

    history_text = ""

    for user_msg, assistant_msg in recent_history:

        history_text += (
            f"User: {user_msg}\n"
            f"Assistant: {assistant_msg}\n\n"
        )

    prompt = f"""
You are a search query rewriting assistant.

Convert the user's latest question into a standalone
search query that can be understood without the
conversation history.

Do not answer the question.

Keep important technical terms, names, products,
services, and document-specific terms.

Conversation:

{history_text}

Latest question:
{question}

Standalone search query:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    rewritten = response["message"]["content"].strip()

    print()
    print("Original question:")
    print(question)

    print()
    print("Rewritten query:")
    print(rewritten)

    return rewritten


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query):

    query_embedding = embedding_model.encode(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=RETRIEVAL_TOP_K,
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    if not documents:

        return []

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            documents,
            metadatas,
            ids,
            scores,
        ),
        key=lambda x: x[3],
        reverse=True,
    )

    return ranked[:FINAL_TOP_K]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for index, (
        document,
        metadata,
        chunk_id,
        score,
    ) in enumerate(results, start=1):

        source = metadata.get(
            "filename",
            "Unknown source",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown",
        )

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Chunk: {chunk_index}

Content:
{document}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
):

    prompt = f"""
You are a helpful local RAG assistant.

Answer the user's question using ONLY the
provided document context.

Important rules:

1. Do not invent information.
2. If the answer is not present in the documents,
   clearly say that the information was not found.
3. Give a direct and useful answer.
4. When using information from a source, mention
   the source filename naturally when useful.
5. Do not mention internal retrieval, embeddings,
   reranking, or Chroma unless the user asks.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"].strip()


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    seen = set()

    source_number = 1

    for (
        document,
        metadata,
        chunk_id,
        score,
    ) in results:

        filename = metadata.get(
            "filename",
            "Unknown",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown",
        )

        source_key = (
            filename,
            chunk_index,
        )

        if source_key in seen:

            continue

        seen.add(source_key)

        print(
            f"[{source_number}] "
            f"{filename} "
            f"(chunk {chunk_index})"
        )

        source_number += 1

    print("=" * 60)


# ============================================================
# MAIN CHAT LOOP
# ============================================================

print()
print("=" * 60)
print("        LOCAL CONVERSATIONAL RAG")
print("=" * 60)
print("Type 'exit' to quit.")
print("=" * 60)


while True:

    try:

        question = input(
            "\nYou: "
        ).strip()

    except KeyboardInterrupt:

        print("\nGoodbye!")

        break

    if not question:

        continue

    if question.lower() in {
        "exit",
        "quit",
    }:

        print("Goodbye!")

        break

    # --------------------------------------------------------
    # QUERY REWRITE
    # --------------------------------------------------------

    search_query = rewrite_query(
        question
    )

    # --------------------------------------------------------
    # RETRIEVE + RERANK
    # --------------------------------------------------------

    results = retrieve_documents(
        search_query
    )

    if not results:

        answer = (
            "I couldn't find relevant information "
            "in the indexed documents."
        )

        print()
        print("Assistant:")
        print(answer)

        conversation_history.append(
            (
                question,
                answer,
            )
        )

        continue

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("Assistant:")
    print(answer)

    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    display_sources(
        results
    )

    # --------------------------------------------------------
    # SAVE CONVERSATION
    # --------------------------------------------------------

    conversation_history.append(
        (
            question,
            answer,
        )
    )

    if len(conversation_history) > MAX_HISTORY:

        conversation_history = (
            conversation_history[
                -MAX_HISTORY:
            ]
        )