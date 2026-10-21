# ⚖️ LegalDocuLex-AI – AI-Powered Contract Assistant

**LegalDocuLex-AI** is a local, privacy-focused legal contract assistant powered by **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF contracts, inspect extracted clauses with exact pagination, generate dense vector embeddings, perform semantic search, and ask natural language questions with strict grounding and verifiable page citations.

---

## 🏗️ Architecture & Pipeline Overview

```
[Contract PDF] 
       │
       ▼
 [PyMuPDF Parser] ──────────► Extracts text, preserves 1-based page numbers
       │
       ▼
[Semantic Chunker] ─────────► Splits text into ~500-word chunks (~80-word overlap)
       │
       ▼
 [all-MiniLM-L6-v2] ────────► Generates 384-dimensional dense semantic vectors
       │
       ▼
 [ChromaDB Vector Store] ───► Stores vectors & metadata (filename, page, chunk_id)
       │
       ▼
[Semantic Retrieval] ───────► Cosine similarity top-K passage retrieval
       │
       ▼
 [OpenAI gpt-4o-mini] ──────► Grounded plain-English answer with page citations
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell, macOS, or Linux

### 2. Environment Setup
```powershell
# Clone the repository and enter the directory
cd "LegalEagle - AI"

# Create a Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```
> **Security Notice:** `.env` is strictly excluded by `.gitignore` and is never committed. LegalDocuLex-AI never prints, logs, or exports secret keys.

### 4. Running the Web Application
```powershell
streamlit run app.py --server.fileWatcherType none
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Automated Verification

LegalDocuLex-AI features a multi-tiered test suite categorized into fast, offline unit tests, security audits, and quantitative RAG evaluations.

### 1. Fast Unit Tests (Offline, Mocked APIs)
Runs in seconds with **zero network calls and no model downloads**:
```powershell
pytest tests/test_unit_pytest.py -v
```
**Coverage:**
- PDF extraction, corrupted PDFs, blank pages, and upload size limits.
- Chunk word counts (~500 words) and overlap (~80 words).
- 384-dimensional embedding validation.
- ChromaDB indexing, retrieval, deduplication/re-indexing, and document filtering.
- Mocked LLM question answering, citation verification, and insufficient evidence handling.
- Chat history state and Markdown/Text conversation exports.

### 2. Security & Reliability Tests
```powershell
pytest tests/test_security_reliability.py -v
```
**Coverage:**
- `.gitignore` verification for `.env` and secret tokens.
- Automated regex audit ensuring zero hardcoded API keys in code files.
- PDF upload size limit validation (25 MB limit).
- Prompt hardening ensuring extracted contract text is treated strictly as untrusted input.
- Zero secret leakage in exported conversation transcripts.
- Graceful handling of missing keys, timeouts, rate limits, and vector DB failures.

### 3. Quantitative RAG Evaluation Benchmark
```powershell
python tests/evaluation/rag_evaluator.py
# Or via pytest:
pytest tests/test_rag_evaluation.py -v
```
Evaluates retrieval recall, citation precision, and grounded fact verification against a deterministic synthetic contract (`synthetic_contract.pdf`) and structured ground-truth dataset (`eval_dataset.json`).

### 4. Legacy Stage Test Suites
Each individual development stage includes standalone regression scripts:
```powershell
python tests/test_pdf_parser.py             # Stage 2: PDF Parsing (5/5 passing)
python tests/test_chunking_embeddings.py    # Stage 3: Chunks & Vectors (5/5 passing)
python tests/test_vector_store.py          # Stage 4: ChromaDB Indexing (5/5 passing)
python tests/test_qa.py                    # Stage 5: Grounded LLM Q&A (6/6 passing)
python tests/test_stage6_management.py      # Stage 6: Interface & Deletion (4/4 passing)
```

---

## 📊 RAG Evaluation Benchmark Results

The quantitative evaluation was conducted using `eval_dataset.json` across 6 test queries on `synthetic_contract.pdf` (Top-K = 3):

| Metric | Target | Actual Score | Status |
| :--- | :---: | :---: | :---: |
| **Retrieval Recall@3** | ≥ 80.0% | **100.0%** (5/5) | PASS ✅ |
| **Citation Correctness** | ≥ 80.0% | **100.0%** (5/5) | PASS ✅ |
| **Grounded Fact Coverage** | ≥ 80.0% | **100.0%** (5/5) | PASS ✅ |
| **Out-of-Domain Refusal Rate** | 100.0% | **100.0%** (1/1) | PASS ✅ |

### Benchmark Query Breakdown:
1. `eval_01_payment_fee` (Base fee $18,500, Net 30): **PASS** (Expected Page 2 | Retrieved Pages: [2, 1, 3])
2. `eval_02_security_retainer` (Retainer deposit $7,500): **PASS** (Expected Page 2 | Retrieved Pages: [2, 3, 1])
3. `eval_03_termination_convenience` (45 days notice): **PASS** (Expected Page 3 | Retrieved Pages: [3, 2, 1])
4. `eval_04_confidentiality_survival` (three (3) years): **PASS** (Expected Page 3 | Retrieved Pages: [3, 1, 2])
5. `eval_05_governing_law` (Delaware, New Castle County): **PASS** (Expected Page 3 | Retrieved Pages: [3, 1, 2])
6. `eval_06_unrelated_out_of_domain` (401k / Stock options): **PASS** (Refusal & Insufficient Evidence Verified)

> **Important Benchmark Note:**
> Evaluation scores are measured strictly on a synthetic test contract with deterministic ground truth to validate RAG pipeline mechanics. These results do not guarantee identical accuracy across arbitrary, bespoke, or complex real-world legal contracts.

---

## 🔒 Security, Privacy & Integrity Commitments

- **Local Vector Storage:** All document embeddings and index metadata reside locally in the `chroma_db/` directory.
- **Untrusted Input Defense:** The system prompt treats all extracted contract text as untrusted source material and explicitly refuses instructions embedded inside contract text.
- **Targeted Deletion Isolation:** Deleting a document removes only its specific vector IDs from ChromaDB. Other contracts and the database collection remain completely unaffected.
- **Upload Guardrails:** Configurable upload size limit (`DEFAULT_MAX_PDF_SIZE_BYTES = 25 MB`) prevents memory exhaustion and denial-of-service attacks.
- **No Compliance Claims:** LegalDocuLex-AI has not undergone independent third-party security audits (e.g. SOC2, ISO 27001, HIPAA) and makes no claims of formal regulatory compliance.

---

## ⚠️ Legal Disclaimer

LegalDocuLex-AI is an educational and assistive prototype designed solely for document exploration and research. **LegalDocuLex-AI does NOT provide legal advice, legal representation, or formal attorney opinions.** Users must never rely on automated AI outputs as a substitute for counsel from a qualified, licensed attorney.
