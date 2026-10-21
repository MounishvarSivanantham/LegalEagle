# ⚖️ LegalDocuLex-AI: System Overview, Tool Stack & Architecture

---

## 1. What is this Project?

### Problem Statement
In legal, corporate, and compliance workflows, reviewing contracts (such as Master Service Agreements, NDAs, Employment Contracts, and Vendor Agreements) is time-consuming, expensive, and error-prone:
- **Length & Density:** Contracts routinely span tens or hundreds of pages filled with dense, archaic legal jargon ("legalese").
- **High-Stakes Liability:** Missing a 30-day notice requirement, an uncapped indemnification clause, or an unfavorable governing law provision can lead to costly legal disputes.
- **The LLM Blindspot (Hallucination):** Standard Large Language Models (like vanilla ChatGPT) cannot reliably answer questions about private documents out-of-the-box. When asked about specific contract terms, generic LLMs tend to **hallucinate** terms, fabricate legal precedents, confuse clauses between pages, or fail to state where in the document a fact was found.
- **Data Privacy Risks:** Pasting confidential corporate contracts into open web tools risks data exposure and compliance breaches.

### The Solution: LegalDocuLex-AI
**LegalDocuLex-AI** is a local, privacy-first **Retrieval-Augmented Generation (RAG)** assistant specifically engineered for legal document exploration. 

Instead of feeding an entire raw document blindly into an AI model, LegalDocuLex-AI:
1. Parses contracts directly from memory.
2. Breaks agreements into overlapping semantic chunks with strictly preserved 1-based page numbers.
3. Converts text into dense 384-dimensional mathematical embeddings locally on your machine.
4. Stores and indexes those vectors in a persistent local database using cosine similarity.
5. Performs semantic search when a question is asked to find the exact clauses that matter.
6. Feeds only those verified passages into a hardened LLM prompt that demands plain-English translation and mandatory page citations (`[contract.pdf, Page 2]`).
7. Refuses to guess when evidence is insufficient and automatically includes educational legal disclaimers.

---

## 2. Tools & Technologies Used, Why They Were Chosen, and Their Work

| Tool / Technology | Category | Why It Was Chosen | Exact Role / Work in This Project |
| :--- | :--- | :--- | :--- |
| **Python (3.10+)** | Core Language | Industry standard for AI, RAG pipelines, data science, and backend scripting. | Houses the complete codebase, modular services, data structures, and test suites. |
| **PyMuPDF (`fitz`)** | PDF Ingestion & Parsing | 10x faster than PDFMiner or PyPDF2; backed by MuPDF's high-performance C library; handles corrupted, password-protected, and empty PDFs gracefully. | Implemented in `services/pdf_parser.py`. Reads raw bytes from memory (`stream=file_bytes`), extracts clean text page-by-page, tracks 1-based page numbers, and enforces a 25MB file-size limit. |
| **Custom Sliding-Window Chunker** | Text Preprocessing | Fixed-page chunking prevents "context bleed" where page numbers become uncertain across long agreements. Overlap prevents clauses split by line breaks from losing context. | Implemented in `services/chunking.py`. Slices pages into ~500-word chunks with ~80-word overlap. Assigns deterministic chunk IDs like `filename_p1_c1`. |
| **Sentence-Transformers (`all-MiniLM-L6-v2`)** | Semantic Dense Embeddings | Free, open-source, runs 100% locally on CPU without GPU or cloud fees; generates fast, high-quality 384-dimensional dense semantic vectors. | Implemented in `services/embeddings.py`. Loads model into a cached singleton. Converts contract clauses and user questions into 384-dimensional vectors capturing conceptual legal meaning. |
| **ChromaDB** | Vector Database & HNSW Indexing | Lightweight, serverless, embedded database that requires no external server setup; persists directly to disk via SQLite; native cosine distance support. | Implemented in `services/vector_store.py`. Persists embeddings and metadata (`filename`, `page_number`, `chunk_id`, `word_count`). Performs sub-millisecond nearest-neighbor search with document-specific filtering and isolated deletion. |
| **OpenAI API (`gpt-4o-mini`)** | Generative LLM & Question Answering | State-of-the-art instruction following, fast response latency, extremely cost-effective ($0.15 / 1M input tokens), and excels at grounded synthesis. | Implemented in `services/qa.py`. Takes retrieved passages and user questions, formats strict system instructions, generates plain-English answers with page bracket citations, and handles API errors securely. |
| **Streamlit** | Web Interface & User Experience | Enables building sleek, reactive, full-featured Python web apps with zero HTML/CSS/JS build steps; fast prototyping and intuitive session state. | Implemented in `app.py`. Delivers a dark navy/blue legal dashboard with 3 distinct sections: 💬 Chat with citation drill-downs, 📑 Document Ingestion & Vector Library, and ℹ️ Architectural Overview. |
| **python-dotenv** | Secret & Config Management | Prevents sensitive credentials (such as `OPENAI_API_KEY`) from ever being hardcoded in scripts or committed to Git. | Reads `.env` on startup; sanitized so that no API keys or secret tokens are ever exported, printed, or rendered in UI error messages. |
| **Pytest** | Automated Quality Assurance | Robust, extensible testing framework with assertions, fixtures, and execution reporting. | Implemented in `tests/`. Runs 53 automated unit, security, and RAG evaluation tests covering edge cases, memory limits, injection defense, and metric benchmarks. |

---

## 3. Deep-Dive: Each Tool and Its Specific Work

### 3.1. PyMuPDF (`services/pdf_parser.py`)
- **What it does:** When a user uploads a contract, PyMuPDF opens the PDF file bytes directly from memory without saving temporary files to disk.
- **Why it matters:** 
  - Standard PDF parsers frequently fail on complex multi-column layouts, password alerts, or zero-byte corrupted files.
  - PyMuPDF provides high speed and returns clean unicode strings while preserving original document layout and pagination.
- **Output:** A structured `PDFParseResult` object holding the filename, page count, total character count, any warnings (e.g., scanned/unreadable pages), and a list of dictionary pages:
  ```python
  {"page_number": 1, "text": "This Service Agreement..."}
  ```

### 3.2. Semantic Chunker (`services/chunking.py`)
- **What it does:** Language models cannot read an entire 80-page contract in a single prompt without losing focus ("lost in the middle" phenomenon) and driving up API costs.
- **Why it matters:** 
  - Chunking breaks the document into bite-sized semantic passages (~500 words).
  - The **80-word overlap** guarantees that clauses spanning paragraph breaks (like indemnification conditions or termination clauses) are not abruptly cut in half.
  - Crucially, chunking is executed **per-page**. This guarantees that every chunk has a 100% accurate source `page_number`.
- **Output:** A list of enriched chunk dictionaries:
  ```python
  {
      "chunk_id": "service_agreement_p2_c1",
      "text": "3. TERM AND TERMINATION. Either party may terminate with 30 days notice...",
      "page_number": 2,
      "filename": "service_agreement.pdf",
      "word_count": 166
  }
  ```

### 3.3. Sentence-Transformers `all-MiniLM-L6-v2` (`services/embeddings.py`)
- **What it does:** Computers cannot calculate mathematical similarities on raw text strings. This model translates words into **384 numbers** (a dense vector) representing conceptual meaning.
- **Why it matters:**
  - If a user asks *"How can I cancel the agreement?"*, keyword search might fail if the contract only uses the word *"Terminate"*.
  - Because `all-MiniLM-L6-v2` embeds semantic concepts, the vector for *"cancel the agreement"* is mathematically close in high-dimensional space to the vector for *"terminate this contract"*.
  - Running locally ensures zero cloud costs for vectorization and complete client confidentiality.

### 3.4. ChromaDB Vector Store (`services/vector_store.py`)
- **What it does:** Indexes all chunk vectors using HNSW (Hierarchical Navigable Small World) graphs configured with cosine space.
- **Why it matters:**
  - Persists data to the local `chroma_db/` directory so documents remain available across app restarts.
  - **Document Isolation & Safe Deletion:** Supports document-specific filtering (`where={"filename": "nda.pdf"}`) so questions about one contract never accidentally pull clauses from another.
  - Re-indexing a document replaces only that document's records without touching other contracts.

### 3.5. OpenAI `gpt-4o-mini` & Hardened QA Engine (`services/qa.py`)
- **What it does:** The reasoning engine that reads retrieved excerpts and articulates the answer for the human user.
- **Why it matters:**
  - Plain LLMs hallucinate; LegalDocuLex-AI injects strict system instructions:
    1. **Strict Grounding:** Base answers *only* on the provided excerpt evidence. Never assume or extrapolate terms.
    2. **Untrusted Data Defense:** Contract text is treated strictly as data to analyze, preventing prompt injection attacks hidden inside malicious contracts.
    3. **Mandatory Citations:** Every single factual claim must cite `[filename, Page X]`.
    4. **Refusal Mechanism:** If the question cannot be answered from the contract (e.g. asking about stock options when the contract is a vendor agreement), the model explicitly outputs: *"The provided contract excerpts do not contain sufficient information to answer this question."*
    5. **Educational Disclaimer:** Appends non-legal advice disclaimers to ensure compliance.

### 3.6. Streamlit User Interface (`app.py`)
- **What it does:** Serves an interactive, responsive web application.
- **Features:**
  - **💬 Chat Interface:** Natural conversation flow with session memory, quick sample question chips, collapsible citation evidence panels, and one-click transcript exports to Markdown (`.md`).
  - **📑 Documents Tab:** Drag-and-drop PDF uploader, page-by-page text inspector, chunking & embedding generator, and ChromaDB library with single-click targeted deletion.
  - **🔍 Semantic Search Tester:** A testing utility allowing legal analysts to run raw vector similarity searches against indexed documents and inspect cosine distances before querying the LLM.

---

## 4. Complete Project Architecture

The architecture follows a modular, uni-directional RAG pipeline:

```
                                  USER INTERFACE LAYER (Streamlit)
               ┌───────────────────────────────────────────────────────────────────┐
               │  💬 Chat & Q&A      📑 Document Ingestion      🔍 Search Tester  │
               └───────────────┬───────────────────────────────────┬───────────────┘
                               │                                   │
              Ingestion Stream │                                   │ Query Stream
                               ▼                                   ▼
                   ┌───────────────────────┐           ┌───────────────────────┐
                   │  Raw PDF Document     │           │  User Question        │
                   │  (Contract PDF)       │           │  ("What is notice?")  │
                   └───────────┬───────────┘           └───────────┬───────────┘
                               │                                   │
                               ▼                                   │
                   ┌───────────────────────┐                       │
                   │  PyMuPDF Parser       │                       │
                   │  (Memory Stream)      │                       │
                   │  - 1-based page nums  │                       │
                   │  - 25MB safety cap    │                       │
                   └───────────┬───────────┘                       │
                               │                                   │
                               ▼                                   │
                   ┌───────────────────────┐                       │
                   │  Semantic Chunker     │                       │
                   │  - ~500 words / chunk │                       │
                   │  - ~80-word overlap   │                       │
                   │  - Preserved metadata │                       │
                   └───────────┬───────────┘                       │
                               │                                   │
                               ▼                                   ▼
                   ┌───────────────────────────────────────────────────┐
                   │          all-MiniLM-L6-v2 (Local Embedder)        │
                   │          Generates 384-dimensional vectors        │
                   └───────────┬───────────────────────────────────┬───┘
                               │                                   │
                               ▼ (Indexed Vectors)                 ▼ (Query Vector)
                   ┌───────────────────────────────────────────────────┐
                   │               ChromaDB Vector Store               │
                   │            (Local SQLite + HNSW Graph)            │
                   │  - Cosine similarity matching                     │
                   │  - Metadata: filename, page_number, chunk_id      │
                   └───────────────────────────┬───────────────────────┘
                                               │
                                               ▼ Top-K Relevant Passages
                   ┌───────────────────────────────────────────────────┐
                   │              Prompt Hardening & Assembly          │
                   │  - Untrusted data containment                     │
                   │  - Mandatory citation format `[doc, Page X]`      │
                   │  - Insufficient evidence refusal guardrails        │
                   └───────────────────────────┬───────────────────────┘
                                               │
                                               ▼ Augmented Prompt
                   ┌───────────────────────────────────────────────────┐
                   │               OpenAI gpt-4o-mini                  │
                   │               Reasoning & Synthesis               │
                   └───────────────────────────┬───────────────────────┘
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────────┐
               │ Grounded Plain-English Answer + Page Citations + Legal Disclaimer  │
               └───────────────────────────────────────────────────────────────────┘
```

---

## 5. Security & Reliability Highlights

1. **Zero Secret Leakage:**
   - OpenAI keys are never hardcoded or printed.
   - `.env` and `.streamlit/secrets.toml` are strictly ignored by `.gitignore`.
   - Exported Markdown transcripts strip all API keys, headers, or auth tokens.
2. **Denial-of-Service Defense:**
   - `DEFAULT_MAX_PDF_SIZE_BYTES = 25 MB` prevents memory exhaustion attacks from maliciously inflated PDF files.
3. **Prompt Injection Hardening:**
   - System instructions explicitly treat retrieved contract text as untrusted source material to prevent malicious instructions embedded in contract clauses from hijacking the LLM.
4. **Targeted Data Isolation:**
   - Deleting a contract deletes only that file's vector IDs using ChromaDB metadata filtering (`where={"filename": target_filename}`), ensuring no cross-document data contamination.
