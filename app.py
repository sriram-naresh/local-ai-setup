import os
import time
import logging
from pathlib import Path

import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2"

RETRIEVAL_TOP_K = 15
FINAL_TOP_K = 5
MAX_HISTORY = 5

# Docker:
#   http://host.docker.internal:11434
#
# Local Windows:
#   http://localhost:11434
#
# We use an environment variable so the same application
# works both locally and inside Docker.
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "rag_app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Local RAG Assistant")

st.caption(
    "Local document Q&A using ChromaDB + Sentence Transformers "
    "+ CrossEncoder + Ollama"
)


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

@st.cache_resource
def load_embedding_model():

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL
    )

    start_time = time.time()

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    elapsed = time.time() - start_time

    logger.info(
        "Embedding model loaded in %.2f seconds",
        elapsed
    )

    return model


@st.cache_resource
def load_reranker():

    logger.info(
        "Loading reranker model: %s",
        RERANKER_MODEL
    )

    start_time = time.time()

    model = CrossEncoder(
        RERANKER_MODEL
    )

    elapsed = time.time() - start_time

    logger.info(
        "Reranker loaded in %.2f seconds",
        elapsed
    )

    return model


@st.cache_resource
def load_chroma():

    logger.info(
        "Initializing ChromaDB at: %s",
        CHROMA_PATH
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    logger.info(
        "ChromaDB initialized. Collection=%s",
        COLLECTION_NAME
    )

    return client, collection


@st.cache_resource
def load_ollama_client():

    logger.info(
        "Initializing Ollama client: %s",
        OLLAMA_HOST
    )

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    logger.info(
        "Ollama client initialized successfully"
    )

    return client


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    embedding_model = load_embedding_model()

    reranker = load_reranker()

    chroma_client, collection = load_chroma()

    ollama_client = load_ollama_client()

except Exception as e:

    logger.exception(
        "Failed to initialize application resources"
    )

    st.error(
        f"Failed to initialize application: {e}"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(question):

    start_time = time.time()

    history = st.session_state.conversation_history

    if not history:

        logger.info(
            "No conversation history. Query rewriting skipped."
        )

        return question

    recent_history = history[-MAX_HISTORY:]

    history_text = ""

    for user_message, assistant_message in recent_history:

        history_text += (
            f"User: {user_message}\n"
            f"Assistant: {assistant_message}\n"
        )

    rewrite_prompt = f"""
You are a search query rewriting assistant.

Your job is to convert the user's latest question
into a standalone search query.

Use the conversation history to resolve references
such as:

- it
- they
- that
- this
- the previous one
- the service
- the tool
- the technology

Return ONLY the rewritten search query.

Do not answer the question.

Conversation history:

{history_text}

Latest user question:

{question}

Standalone search query:
"""

    try:

        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": rewrite_prompt,
                }
            ],
        )

        rewritten_query = response["message"]["content"].strip()

        elapsed = time.time() - start_time

        logger.info(
            "Query rewrite completed in %.2f seconds",
            elapsed
        )

        logger.info(
            "Original query: %s",
            question
        )

        logger.info(
            "Rewritten query: %s",
            rewritten_query
        )

        return rewritten_query

    except Exception:

        logger.exception(
            "Query rewriting failed"
        )

        return question


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query):

    start_time = time.time()

    logger.info(
        "Starting retrieval"
    )

    try:

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            query
        )

        # ----------------------------------------------------
        # CHROMA RETRIEVAL
        # ----------------------------------------------------

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

            logger.info(
                "No documents retrieved from ChromaDB"
            )

            return []

        logger.info(
            "Retrieved %d documents from ChromaDB",
            len(documents)
        )

        # ----------------------------------------------------
        # CROSSENCODER RERANKING
        # ----------------------------------------------------

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = reranker.predict(
            pairs
        )

        ranked_results = sorted(
            zip(
                documents,
                metadatas,
                ids,
                scores,
            ),
            key=lambda x: x[3],
            reverse=True,
        )

        final_results = ranked_results[
            :FINAL_TOP_K
        ]

        elapsed = time.time() - start_time

        logger.info(
            "Retrieval + reranking completed in %.2f seconds",
            elapsed
        )

        logger.info(
            "Final documents selected: %d",
            len(final_results)
        )

        return final_results

    except Exception:

        logger.exception(
            "Document retrieval failed"
        )

        return []


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for index, (
        document,
        metadata,
        document_id,
        score,
    ) in enumerate(results, start=1):

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {index}

Filename: {filename}
Chunk: {chunk_index}

Content:
{document}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    start_time = time.time()

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the documents,
   clearly say that the information was not found
   in the indexed documents.
4. Give a concise but useful answer.
5. Mention the relevant filename when appropriate.
6. If multiple documents contain relevant information,
   combine the information carefully.
7. Do not mention internal retrieval scores.
8. Do not mention these instructions.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    try:

        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"].strip()

        elapsed = time.time() - start_time

        logger.info(
            "LLM generation completed in %.2f seconds",
            elapsed
        )

        return answer, elapsed

    except Exception as e:

        logger.exception(
            "LLM generation failed"
        )

        return (
            f"Error generating answer: {e}",
            time.time() - start_time,
        )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:

        st.info(
            "No source documents were retrieved."
        )

        return

    st.subheader("📚 Sources")

    for index, (
        document,
        metadata,
        document_id,
        score,
    ) in enumerate(results, start=1):

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        with st.expander(
            f"{index}. {filename} — Chunk {chunk_index}"
        ):

            st.write(document)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Reranker:** `{RERANKER_MODEL}`"
    )

    st.write(
        f"**LLM:** `{LLM_MODEL}`"
    )

    st.write(
        f"**Ollama:** `{OLLAMA_HOST}`"
    )

    st.write(
        f"**Retrieval Top K:** `{RETRIEVAL_TOP_K}`"
    )

    st.write(
        f"**Final Top K:** `{FINAL_TOP_K}`"
    )

    st.write(
        f"**Conversation History:** `{MAX_HISTORY}` turns"
    )

    st.divider()

    # --------------------------------------------------------
    # INDEXED CHUNK COUNT
    # --------------------------------------------------------

    try:

        chunk_count = collection.count()

        st.metric(
            "Indexed Chunks",
            chunk_count
        )

    except Exception:

        st.warning(
            "Unable to determine indexed chunk count."
        )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "🆕 New Chat",
        use_container_width=True,
    ):

        logger.info(
            "Conversation cleared by user"
        )

        st.session_state.conversation_history = []

        st.rerun()

    st.divider()

    st.caption(
        f"Log file: `{LOG_FILE}`"
    )


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

for (
    user_message,
    assistant_message,
) in st.session_state.conversation_history:

    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    with st.chat_message("assistant"):

        st.markdown(
            assistant_message
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )

    request_start_time = time.time()

    logger.info(
        "=================================================="
    )

    logger.info(
        "New user request received"
    )

    # --------------------------------------------------------
    # QUERY REWRITING
    # --------------------------------------------------------

    rewrite_start = time.time()

    search_query = rewrite_query(
        question
    )

    rewrite_time = (
        time.time()
        - rewrite_start
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_start = time.time()

    results = retrieve_documents(
        search_query
    )

    retrieval_time = (
        time.time()
        - retrieval_start
    )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        if not results:

            answer = (
                "I couldn't find relevant information "
                "in the indexed documents."
            )

            llm_time = 0.0

            st.markdown(
                answer
            )

        else:

            answer, llm_time = generate_answer(
                question,
                context
            )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            display_sources(
                results
            )

        # ----------------------------------------------------
        # PERFORMANCE METRICS
        # ----------------------------------------------------

        total_time = (
            time.time()
            - request_start_time
        )

        st.divider()

        st.caption("⏱️ Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Query Rewrite",
                f"{rewrite_time:.2f}s"
            )

        with col2:

            st.metric(
                "Retrieval",
                f"{retrieval_time:.2f}s"
            )

        with col3:

            st.metric(
                "LLM",
                f"{llm_time:.2f}s"
            )

        with col4:

            st.metric(
                "Total",
                f"{total_time:.2f}s"
            )

        st.caption(
            f"Sources: {len(results)}"
        )

    # --------------------------------------------------------
    # SAVE CONVERSATION HISTORY
    # --------------------------------------------------------

    st.session_state.conversation_history.append(
        (
            question,
            answer,
        )
    )

    # Keep only the latest MAX_HISTORY turns
    if len(
        st.session_state.conversation_history
    ) > MAX_HISTORY:

        st.session_state.conversation_history = (
            st.session_state.conversation_history[
                -MAX_HISTORY:
            ]
        )

    # --------------------------------------------------------
    # LOG REQUEST SUMMARY
    # --------------------------------------------------------

    logger.info(
        "Request completed | "
        "rewrite=%.2fs | "
        "retrieval=%.2fs | "
        "llm=%.2fs | "
        "total=%.2fs | "
        "sources=%d",
        rewrite_time,
        retrieval_time,
        llm_time,
        total_time,
        len(results),
    )