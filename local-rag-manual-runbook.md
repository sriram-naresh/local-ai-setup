# Local RAG - Manual Runbook

## 1. Project

**Project folder**

```text
Windows:
C:\Users\srira\Documents\local-RAG\my-rag

Git Bash:
/c/Users/srira/Documents/local-RAG/my-rag
```

**Main components**

```text
Documents / Downloads / Desktop
        |
        v
   rag_watcher.py
        |
        v
    ingestion.py
        |
        v
     ChromaDB
        |
        v
  chat_rag_v2.py
        |
        v
    Ollama / llama3.2
```

### Main scripts

| Script | Purpose |
|---|---|
| `ingestion.py` | Reads files, chunks text, creates embeddings, stores data in ChromaDB |
| `rebuild_rag.py` | Deletes/recreates the Chroma database and indexes supported files from scratch |
| `rag_watcher.py` | Watches folders and automatically indexes created/modified/deleted/moved files |
| `chat_rag_v2.py` | Conversational RAG chat with query rewriting and reranking |
| `chroma_db/` | Persistent local ChromaDB data |

### Current supported file types

```text
.txt
.pdf
.docx
```

Excel and PowerPoint are not currently included.

---

# 2. Start the Environment

Open **Git Bash**.

Go to the project:

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
```

Activate the virtual environment:

```bash
source venv/Scripts/activate
```

You should see something similar to:

```text
(venv) MINGW64 ...
```

Optional Python check:

```bash
python --version
```

Expected environment used by this project:

```text
Python 3.13.x
```

---

# 3. Check Ollama

The RAG chat uses the local Ollama model:

```text
llama3.2
```

Check Ollama:

```bash
ollama --version
```

Check installed models:

```bash
ollama list
```

You should see:

```text
llama3.2
```

Test the model:

```bash
ollama run llama3.2
```

Type a simple question, for example:

```text
What is Kubernetes?
```

Exit Ollama with:

```text
/bye
```

---

# 4. Check Required Tools

## Tesseract OCR

Check:

```bash
tesseract --version
```

Expected installation location:

```text
C:\Program Files\Tesseract-OCR
```

If Git Bash cannot find it:

```bash
export PATH="$PATH:/c/Program Files/Tesseract-OCR"
```

Then:

```bash
tesseract --version
```

---

## Poppler

Poppler is required for OCR fallback on scanned PDFs.

Current installation used by the project:

```text
C:\Program Files\poppler-26.09.0\Library\bin
```

Add it to the current Git Bash session:

```bash
export PATH="$PATH:/c/Program Files/poppler-26.09.0/Library/bin"
```

Check:

```bash
pdfinfo -v
```

If `pdfinfo` works, Poppler is available.

---

# 5. Normal Daily Startup

For normal usage, do these steps.

## Terminal 1 - Activate environment

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
source venv/Scripts/activate
```

## Terminal 2 - Start the watcher

From the same project folder:

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
source venv/Scripts/activate
python rag_watcher.py
```

Expected:

```text
Watcher is running.
```

Keep this terminal running.

The watcher monitors the configured user folders and automatically handles:

- New files
- Modified files
- Deleted files
- Moved files

A short debounce is used so multiple filesystem events do not immediately cause repeated ingestion.

---

## Terminal 3 - Start the RAG chat

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
source venv/Scripts/activate
python chat_rag_v2.py
```

Expected:

```text
RAG system ready.
```

Then ask questions:

```text
You: what is kubernetes
```

or:

```text
You: what is terraform
```

---

# 6. How the Chat Pipeline Works

`chat_rag_v2.py` currently follows this flow:

```text
User question
     |
     v
Conversation history
     |
     v
Local LLM query rewriting
     |
     v
Embedding model
all-MiniLM-L6-v2
     |
     v
ChromaDB retrieval
top 15
     |
     v
CrossEncoder reranking
ms-marco-MiniLM-L-6-v2
     |
     v
Final top 5 chunks
     |
     v
Ollama llama3.2
     |
     v
Answer + source information
```

The conversational history keeps the latest five turns.

---

# 7. Manual Ingestion of a File

If you want to index one file manually:

```bash
python -c "import ingestion; ingestion.ingest_file(r'C:\path\to\your\file.pdf')"
```

Example:

```bash
python -c "import ingestion; ingestion.ingest_file(r'C:\Users\srira\Documents\example.pdf')"
```

The ingestion process:

```text
File
  |
  v
Read text
  |
  +--> Normal PDF extraction
  |
  +--> OCR fallback for scanned PDF
  |
  v
Chunking
  |
  v
Embeddings
  |
  v
ChromaDB
```

The current chunking configuration is approximately:

```text
Chunk size: 1000 characters
Overlap:     150 characters
```

---

# 8. Full Rebuild of ChromaDB

Use a rebuild when:

- The Chroma database is empty
- Metadata needs to be recreated
- You changed ingestion logic
- You changed chunking logic
- You changed the metadata structure
- You want to re-index all supported files from scratch

## IMPORTANT

A rebuild removes the existing local `chroma_db` and recreates it.

Stop these first:

```text
rag_watcher.py
chat_rag_v2.py
```

Do not rebuild while another process is actively using the Chroma database.

---

## Run rebuild

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
source venv/Scripts/activate
python rebuild_rag.py
```

The script asks:

```text
Type REBUILD to continue:
```

Enter:

```text
REBUILD
```

The script then:

1. Removes the existing local Chroma database.
2. Creates a fresh Chroma database.
3. Scans the configured user folders.
4. Reads supported files.
5. Performs OCR fallback when required.
6. Creates chunks.
7. Generates embeddings.
8. Stores records in ChromaDB.
9. Prints final statistics.

Wait until the rebuild completely finishes.

---

# 9. Verify ChromaDB After Rebuild

Run:

```bash
python -c "import os, chromadb; print('CWD:', os.getcwd()); c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count()); print('Sample:', col.get(limit=3, include=['documents','metadatas']))"
```

Important result:

```text
Chunks: <number greater than 0>
```

If:

```text
Chunks: 0
```

the RAG will not be able to retrieve documents.

---

# 10. Verify Chroma Collection

A shorter check:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count())"
```

Expected:

```text
Chunks: 123
```

The exact number depends on the files currently indexed.

---

# 11. Test Retrieval Directly

If Chroma has chunks but chat does not return results, test retrieval directly:

```bash
python -c "import chromadb; from sentence_transformers import SentenceTransformer; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); m=SentenceTransformer('all-MiniLM-L6-v2'); e=m.encode(['what is kubernetes']).tolist(); r=col.query(query_embeddings=e,n_results=5,include=['documents','metadatas','distances']); print(r)"
```

This helps determine whether the problem is:

```text
Chroma retrieval
```

or:

```text
chat_rag_v2.py
```

---

# 12. Watcher Testing

Start:

```bash
python rag_watcher.py
```

Keep it running.

Create a small test file inside a watched folder.

Example:

```bash
echo "Kubernetes is used for container orchestration." > /c/Users/srira/Documents/watcher-rag-test.txt
```

The watcher should report something similar to:

```text
[MODIFIED] ...watcher-rag-test.txt
[INDEXING] ...watcher-rag-test.txt
INGESTING: watcher-rag-test.txt
Created 1 chunks.
Generating embeddings...
Storing new version...
New version stored successfully.
Successfully indexed 1 chunks.
```

Then query the RAG:

```text
What is Kubernetes?
```

After testing, delete the temporary test file:

```bash
rm /c/Users/srira/Documents/watcher-rag-test.txt
```

The watcher should process the deletion.

---

# 13. Stop the Watcher

In the watcher terminal:

```text
Ctrl+C
```

Expected behavior:

```text
Stopping watcher...
Watcher stopped.
```

This is the normal clean shutdown.

---

# 14. Check Running Python Processes

If Chroma says a database file is locked, check running Python processes:

```bash
tasklist | grep -i python
```

If necessary, force-stop Python processes:

```bash
taskkill //F //IM python.exe
```

Then start only the process you actually need.

**Warning:** `taskkill //F //IM python.exe` stops all Python processes on Windows, so use it only when appropriate.

---

# 15. Common Troubleshooting

## Problem: `Chunks: 0`

### Check current directory

```bash
pwd
```

It should be:

```text
/c/Users/srira/Documents/local-RAG/my-rag
```

Check the database:

```bash
ls -la
```

You should see:

```text
chroma_db
```

Then:

```bash
python -c "import os, chromadb; print('CWD:', os.getcwd()); c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count())"
```

If still zero, run:

```bash
python rebuild_rag.py
```

and enter:

```text
REBUILD
```

---

## Problem: `Collection does not exist`

Run the rebuild:

```bash
python rebuild_rag.py
```

Then verify:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); print(c.list_collections())"
```

---

## Problem: Chroma database is locked

Typical error:

```text
WinError 32
The process cannot access the file because it is being used by another process.
```

Stop the watcher and chat:

```text
Ctrl+C
```

Then check:

```bash
tasklist | grep -i python
```

If needed:

```bash
taskkill //F //IM python.exe
```

Then retry:

```bash
python rebuild_rag.py
```

---

## Problem: RAG says "I couldn't find relevant information"

First check:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count())"
```

If zero:

```bash
python rebuild_rag.py
```

If greater than zero, run the direct retrieval test:

```bash
python -c "import chromadb; from sentence_transformers import SentenceTransformer; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); m=SentenceTransformer('all-MiniLM-L6-v2'); e=m.encode(['what is kubernetes']).tolist(); r=col.query(query_embeddings=e,n_results=5,include=['documents','metadatas','distances']); print(r)"
```

If direct retrieval works but chat does not, investigate:

```text
chat_rag_v2.py
```

If direct retrieval returns nothing, investigate:

```text
ChromaDB
embeddings
ingestion
```

---

## Problem: `Unknown (chunk Unknown)` in sources

This generally indicates that the stored records do not have the expected metadata.

Rebuild the database using:

```bash
python rebuild_rag.py
```

Then verify:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print(col.get(limit=3, include=['documents','metadatas']))"
```

Metadata should contain fields such as:

```text
source
filename
extension
chunk_index
file_hash
```

---

# 16. PDF Extraction Problems

## Normal PDF

The ingestion code first tries normal PDF text extraction.

## Scanned PDF

If no usable text is extracted, OCR is attempted using:

```text
Poppler
+
Tesseract
```

Check Tesseract:

```bash
tesseract --version
```

Check Poppler:

```bash
pdfinfo -v
```

If Poppler is not found:

```bash
export PATH="$PATH:/c/Program Files/poppler-26.09.0/Library/bin"
```

If Tesseract is not found:

```bash
export PATH="$PATH:/c/Program Files/Tesseract-OCR"
```

Then retry ingestion/rebuild.

---

# 17. Encrypted / Password-Protected PDFs

Some PDFs may fail with an error similar to:

```text
File has not been decrypted
```

This means the PDF is encrypted/password protected.

OCR may also fail if the PDF cannot be rendered.

Do not assume the RAG can index a protected PDF.

Possible options:

1. Open the PDF normally and provide the required password.
2. Create an accessible copy if you have permission.
3. Remove the protection using an appropriate authorized workflow.
4. Re-run ingestion on the accessible copy.

The current ingestion process reports the failure rather than silently indexing empty content.

---

# 18. DOCX Problems

The project supports `.docx`.

If a DOCX is corrupt or invalid, ingestion may fail.

A common example is an Office temporary file beginning with:

```text
~$
```

The rebuild process skips Office temporary files beginning with:

```text
~$
```

If a DOCX fails:

1. Open it in Microsoft Word.
2. Confirm it is readable.
3. Save it as a new `.docx`.
4. Retry ingestion.

---

# 19. FontTools / PDF Warnings

During PDF processing you may see warnings related to PDF fonts, including `fontTools`.

For example, warnings about fonts not being installed do not automatically mean the entire ingestion failed.

Look for the actual final result:

```text
Successfully indexed ...
```

or:

```text
PDF extraction failed
```

The success/failure message is more important than a warning alone.

---

# 20. Watcher Does Not Detect Changes

Check that the watcher is running:

```bash
python rag_watcher.py
```

Check that the file is inside one of the configured folders:

```text
Documents
Downloads
Desktop
```

The current configuration warns if Desktop does not exist.

Create a test file:

```bash
echo "watcher test" > /c/Users/srira/Documents/watcher-rag-test.txt
```

Watch the terminal for:

```text
[MODIFIED]
```

and:

```text
[INDEXING]
```

If nothing happens:

1. Confirm the watcher is still running.
2. Confirm the file extension is supported.
3. Confirm the file is not inside an excluded directory.
4. Try modifying the file again.

---

# 21. Files and Folders Excluded From Indexing

The current rebuild/watcher configuration excludes directories such as:

```text
.git
.svn
.hg
venv
.venv
env
.env
node_modules
__pycache__
.pytest_cache
.mypy_cache
chroma_db
$Recycle.Bin
```

This prevents virtual environments, source-control internals, generated files, and the RAG database itself from being indexed.

---

# 22. Check Installed Python Packages

Run:

```bash
pip list
```

Important packages for this project include:

```text
chromadb
sentence-transformers
ollama
pypdf
python-docx
pytesseract
pdf2image
pillow
watchdog
```

If a package is missing:

```bash
pip install <package-name>
```

For example:

```bash
pip install watchdog
```

---

# 23. Reinstall Project Dependencies

If the virtual environment becomes corrupted, install the core packages again:

```bash
pip install chromadb sentence-transformers ollama pypdf python-docx pytesseract pdf2image pillow watchdog
```

Then verify:

```bash
python -c "import chromadb, sentence_transformers, ollama, pypdf, docx, pytesseract, pdf2image, PIL, watchdog; print('Core imports OK')"
```

---

# 24. Check Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

Test:

```bash
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); print(m.get_sentence_embedding_dimension())"
```

Expected:

```text
384
```

---

# 25. Check Reranker

The project uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker is used after initial Chroma retrieval.

Current flow:

```text
Chroma retrieves top 15
        |
        v
CrossEncoder scores them
        |
        v
Top 5 passed to LLM
```

---

# 26. Important Chroma Path Check

The project uses:

```text
./chroma_db
```

That means the database path is relative to the current working directory.

Always start the project from:

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
```

before running:

```bash
python rebuild_rag.py
python rag_watcher.py
python chat_rag_v2.py
```

This avoids accidentally creating/using a different `chroma_db` in another directory.

---

# 27. Recommended Daily Workflow

For normal use:

### Step 1

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
```

### Step 2

```bash
source venv/Scripts/activate
```

### Step 3

Check Ollama:

```bash
ollama list
```

### Step 4

Start watcher:

```bash
python rag_watcher.py
```

### Step 5

In another Git Bash terminal:

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
source venv/Scripts/activate
python chat_rag_v2.py
```

### Step 6

Ask questions.

New/changed supported files in the watched folders should be automatically indexed.

---

# 28. When to Rebuild vs. When Not to Rebuild

## Do NOT rebuild for every new document

The watcher is designed for incremental ingestion.

If you add:

```text
new-document.pdf
```

the watcher should index it automatically.

## Rebuild when

```text
Chroma is empty
OR
metadata/schema changed
OR
chunking changed
OR
embedding strategy changed
OR
ingestion logic changed significantly
OR
you intentionally want a clean full index
```

---

# 29. Quick Health Check

Run these commands in order:

```bash
cd /c/Users/srira/Documents/local-RAG/my-rag
```

```bash
source venv/Scripts/activate
```

```bash
python --version
```

```bash
ollama list
```

```bash
tesseract --version
```

```bash
pdfinfo -v
```

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chroma chunks:', col.count())"
```

If the chunk count is greater than zero, test:

```bash
python chat_rag_v2.py
```

Then:

```text
what is kubernetes
```

---

# 30. Emergency Reset

If you intentionally want to start the local RAG database from scratch:

### Stop all RAG Python processes

```bash
tasklist | grep -i python
```

Stop them if required:

```bash
taskkill //F //IM python.exe
```

### Delete the local Chroma database

From the project directory:

```bash
rm -rf chroma_db
```

### Rebuild

```bash
python rebuild_rag.py
```

Enter:

```text
REBUILD
```

### Verify

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count())"
```

---

# 31. Architecture Summary

The current local system is:

```text
                    USER FILES
              /       |       \
       Documents   Downloads   Desktop
              \       |       /
                    Watchdog
                       |
                 debounce events
                       |
                       v
                 ingestion.py
                       |
             +---------+---------+
             |                   |
          Read/OCR            Chunking
             |                   |
             +---------+---------+
                       |
                       v
              Sentence Transformer
              all-MiniLM-L6-v2
                       |
                       v
                    ChromaDB
                       |
                       v
                chat_rag_v2.py
                       |
                 Query Rewrite
                       |
                       v
              Chroma Top 15
                       |
                       v
                 CrossEncoder
                       |
                  Top 5 chunks
                       |
                       v
                 Ollama Llama 3.2
                       |
                       v
                    Answer
                       +
                   Sources
```

---

# 32. Production Direction

This local project is currently a single-machine development RAG.

A future production architecture could separate:

```text
RAG API
Ingestion Worker
Vector Database
Object/File Storage
LLM Service
Monitoring
```

and run workloads using containers/Kubernetes.

The current scripts are useful as the foundation for that transition.

---

# 33. Quick Command Cheat Sheet

```bash
# Go to project
cd /c/Users/srira/Documents/local-RAG/my-rag

# Activate venv
source venv/Scripts/activate

# Check Python
python --version

# Check Ollama
ollama list

# Check OCR
tesseract --version

# Check Poppler
pdfinfo -v

# Check Chroma
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_collection('my_documents'); print('Chunks:', col.count())"

# Full rebuild
python rebuild_rag.py

# Start watcher
python rag_watcher.py

# Start RAG chat
python chat_rag_v2.py

# Check Python processes
tasklist | grep -i python

# Force-stop all Python processes (use carefully)
taskkill //F //IM python.exe

# Add OCR tools to current Git Bash PATH
export PATH="$PATH:/c/Program Files/Tesseract-OCR"
export PATH="$PATH:/c/Program Files/poppler-26.09.0/Library/bin"
```

---

## Final troubleshooting order

When something is not working, use this order:

```text
1. Confirm project directory
        |
        v
2. Activate venv
        |
        v
3. Check Ollama
        |
        v
4. Check Chroma chunk count
        |
        +---- 0 ----> rebuild_rag.py
        |
        v
5. Test direct Chroma retrieval
        |
        v
6. Test chat_rag_v2.py
        |
        v
7. If file ingestion is the issue:
       check Tesseract / Poppler
        |
        v
8. If database is locked:
       stop Python processes
        |
        v
9. Rebuild only when necessary
```
