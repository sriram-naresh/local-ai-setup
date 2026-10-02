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
# 5. QUERY REWRITING FUNCTION
# ==================================================

def rewrite_question(question, history):

    # ----------------------------------------------
    # No history = no rewriting needed
    # ----------------------------------------------

    if not history:

        return question


    # ----------------------------------------------
    # Build conversation history
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
You are a search-query rewriting assistant.

Your job is to convert the user's latest question
into a standalone search query for a document
knowledge base.

Use the conversation history only to resolve
references such as:

- it
- this
- that
- they
- them
- which one
- what about it
- the above
- the previous one

IMPORTANT RULES:

1. Preserve the user's original intent.

2. Preserve important words and concepts from the
   original question.

3. Do NOT remove important entities, technologies,
   products, certifications, names, or topics.

4. Do NOT answer the question.

5. Do NOT add unrelated information.

6. Do NOT guess what the user means beyond what can
   reasonably be determined from the conversation.

7. If the question is already standalone, return it
   unchanged.

8. Return ONLY the standalone search query.

Examples:

Conversation:
User: What is Amazon EKS?
User: What storage options are available?

Good rewrite:
What storage options are available for Amazon EKS?

Conversation:
User: What certifications do I have?
User: Which one is related to Terraform?

Good rewrite:
Which certification is related to Terraform?

Bad rewrite:
Kubernetes Certified Administrator or AWS Certified Devops Engineer Professional

Conversation:

{history_text}

Latest user question:

{question}

Standalone search query:
"""


    # ----------------------------------------------
    # Ask local Llama
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
    # Fallback
    # ----------------------------------------------

    if not rewritten:

        return question


    return rewritten


# ==================================================
# 6. MAIN CHAT LOOP
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
    # 9. DISPLAY SEARCH QUERY
    # ==================================================

    print(
        "\nSearch query:"
    )

    print(
        rewritten_query
    )


    # ==================================================
    # 10. CREATE QUESTION EMBEDDING
    # ==================================================

    question_embedding = embedding_model.encode(

        rewritten_query

    ).tolist()


    # ==================================================
    # 11. RETRIEVE 15 CANDIDATES
    # ==================================================

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=15,

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
    # 16. SORT BY RERANKER SCORE
    # ==================================================

    reranked_results.sort(

        key=lambda x: x[2],

        reverse=True

    )


    # ==================================================
    # 17. KEEP BEST 5
    # ==================================================

    top_results = (
        reranked_results[:5]
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
    # 20. FINAL ANSWER PROMPT
    # ==================================================

    prompt = f"""
You are a helpful conversational question-answering assistant.

Answer the user's current question using the provided
source documents and conversation history.

IMPORTANT RULES:

1. The source documents are the primary source of truth.

2. If the source documents explicitly contain the answer,
   use that information directly.

3. Do not claim that information is missing when the
   information is explicitly present in the source documents.

4. Do not invent specific facts.

5. Do not contradict information explicitly stated in
   the source documents.

6. Use conversation history to understand references such
   as "it", "this", "that", "which one", and similar phrases.

7. If multiple source documents contain the same information,
   combine the information naturally.

8. When the question asks for a list, provide a clear list.

9. When the question asks for a definition, use the definition
   supported by the source documents.

10. Do not expand acronyms incorrectly.

11. If an acronym is explicitly defined in the source,
    use that definition.

12. If the source does not contain enough information,
    clearly say that the available documents do not provide
    enough information.

13. Do not mention RAG.

14. Do not mention embeddings.

15. Do not mention ChromaDB.

16. Do not mention the reranker.

17. Do not mention query rewriting.

18. Do not describe your reasoning.

19. Do not ask the user a question.

20. Keep the answer concise but complete.

CONVERSATION HISTORY:

{history_text}

SOURCE DOCUMENTS:

{context}

CURRENT USER QUESTION:

{question}

FINAL ANSWER:
"""


    # ==================================================
    # 21. ASK LLAMA
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