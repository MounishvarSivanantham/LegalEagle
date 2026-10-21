# ============================================================
# app.py  –  LegalEagle: AI-Powered Contract Assistant
# ============================================================
# Step 6: Polished Interface & Document / Chat Management
#
# Key Features:
#   1. Responsive Navy/White/Blue Theme with Sidebar Navigation:
#      - 💬 Chat (Q&A with persistent citations, follow-ups, export)
#      - 📑 Documents (Upload, Chunk, Embed, Index, Library, Safe Delete)
#      - ℹ️ About (RAG explanation & qualified legal disclaimer)
#   2. Session-State Chat History with New Conversation & Download.
#   3. ChromaDB Document Management & Document-Specific Filtering.
#   4. Safe Deletion of only selected document's vectors.
#   5. Strictly grounded citations with page references & excerpts.
#
# Run with:
#     streamlit run app.py
# ============================================================

import streamlit as st
import datetime
import os
from typing import Optional, List, Dict, Any

# Import custom services
from services.pdf_parser import extract_text_from_pdf, get_text_preview
from services.chunking import chunk_parse_result, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from services.embeddings import (
    generate_chunk_embeddings,
    EXPECTED_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
)
from services.vector_store import (
    index_document_chunks,
    search_contract_chunks,
    get_collection_stats,
    get_document_metadata_list,
    delete_document_vectors,
    DEFAULT_COLLECTION_NAME,
)
from services.qa import (
    answer_contract_question,
    get_openai_api_key,
    export_conversation_to_markdown,
    export_conversation_to_text,
    DEFAULT_OPENAI_MODEL,
)

# ── 1. Streamlit Page Configuration ───────────────────────────
st.set_page_config(
    page_title="LegalEagle – AI Contract Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Custom CSS: Navy, White, and Subtle Blue Theme ─────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Header Banner ---- */
    .main-header {
        background: linear-gradient(135deg, #0A192F 0%, #172A45 50%, #1E3A8A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 14px rgba(10, 25, 47, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header h1 {
        color: #F8FAFC;
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header .tagline {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }

    /* ---- Sidebar Customization ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070D1F 0%, #0B192F 100%);
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F1F5F9 !important;
    }

    /* ---- Metric Stat Cards ---- */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 110px;
        background: #0D1B2A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .metric-card .value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38BDF8;
    }

    /* ---- Document Management Card ---- */
    .doc-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .doc-card h4 {
        color: #F8FAFC;
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
    }

    /* ---- Legal Disclaimer Banner ---- */
    .disclaimer-banner {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0;
        color: #CBD5E1;
        font-size: 0.85rem;
    }
    .disclaimer-banner strong {
        color: #FCD34D;
    }

    /* ---- Badges ---- */
    .badge-primary {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-success {
        background-color: #064E3B;
        color: #6EE7B7;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 3. Session State Initialization ───────────────────────────
def init_session_state():
    """Initialize persistent session variables across Streamlit reruns."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of dicts with role, content, sources, timestamp

    if "current_nav" not in st.session_state:
        st.session_state.current_nav = "💬 Chat"

    if "selected_doc_filter" not in st.session_state:
        st.session_state.selected_doc_filter = "All Documents"

    # Current PDF upload state
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "pdf_size" not in st.session_state:
        st.session_state.pdf_size = 0
    if "upload_time" not in st.session_state:
        st.session_state.upload_time = None
    if "parse_result" not in st.session_state:
        st.session_state.parse_result = None

    # Chunking & Embeddings state
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embedding_status" not in st.session_state:
        st.session_state.embedding_status = None
    if "embedding_error" not in st.session_state:
        st.session_state.embedding_error = None

    # ChromaDB indexing state
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "index_status" not in st.session_state:
        st.session_state.index_status = None
    if "index_error" not in st.session_state:
        st.session_state.index_error = None
    if "indexed_chunk_count" not in st.session_state:
        st.session_state.indexed_chunk_count = 0

    # Search tester state
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""


init_session_state()


# ── 4. Sidebar: Navigation, Document Overview & Settings ─────
with st.sidebar:
    st.markdown("## ⚖️ LegalEagle")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; margin-top:-0.5rem;'>"
        "AI-Powered Legal Contract Assistant</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation Radio
    nav_selection = st.radio(
        "Navigation",
        options=["💬 Chat", "📑 Documents", "ℹ️ About"],
        index=["💬 Chat", "📑 Documents", "ℹ️ About"].index(st.session_state.current_nav)
        if st.session_state.current_nav in ["💬 Chat", "📑 Documents", "ℹ️ About"]
        else 0,
        key="sidebar_nav_radio",
        label_visibility="collapsed",
    )
    st.session_state.current_nav = nav_selection
    st.divider()

    # Vector Store Quick Stats
    st.markdown("### 🗄️ Vector Knowledge Base")
    db_stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
    total_vectors = db_stats.get("total_chunks", 0)
    indexed_doc_names = db_stats.get("documents", [])

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Indexed Docs</div>
                <div class="value">{len(indexed_doc_names)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Chunks</div>
                <div class="value">{total_vectors}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if indexed_doc_names:
        with st.expander("📁 Indexed Contracts", expanded=False):
            for doc in indexed_doc_names:
                st.markdown(f"• `{doc}`")
    else:
        st.caption("No contracts indexed yet. Visit the **Documents** tab to upload and index.")

    st.divider()

    # OpenAI API Key Configuration
    st.markdown("### 🔑 API Configuration")
    detected_key = get_openai_api_key()
    if detected_key:
        st.success("OpenAI Key Active ✅")
    else:
        st.warning("OpenAI Key Missing ⚠️")
        user_key = st.text_input(
            "Session API Key:",
            type="password",
            placeholder="sk-proj-...",
            help="Set your OpenAI API key for this session. Alternatively, set OPENAI_API_KEY in your .env file.",
            key="sidebar_user_key",
        )
        if user_key and user_key.strip():
            os.environ["OPENAI_API_KEY"] = user_key.strip()
            st.rerun()

    # Footer note
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#475569; font-size:0.75rem; text-align:center;'>"
        "LegalEagle v0.6 · Step 6<br>For informational use only</p>",
        unsafe_allow_html=True,
    )


# ── 5. Main Header ────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <div>
            <h1>⚖️ LegalEagle</h1>
            <div class="tagline">Grounded Contract Analysis & Intelligent Clause Retrieval</div>
        </div>
        <div>
            <span class="badge-primary">v0.6 Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── 6. Page Renderers ─────────────────────────────────────────

def render_chat_page():
    """Render the primary LegalEagle conversational interface."""
    st.markdown("### 💬 Contract Assistant Chat")

    # Fetch currently indexed documents for the filter selector
    db_stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
    indexed_docs = db_stats.get("documents", [])

    # Top Toolbar: Document Context Filter, New Chat, Export
    col_filter, col_new, col_export = st.columns([3, 1.2, 1.5])

    with col_filter:
        options = ["All Documents (Cross-Contract Search)"] + indexed_docs
        current_idx = 0
        if st.session_state.selected_doc_filter in indexed_docs:
            current_idx = indexed_docs.index(st.session_state.selected_doc_filter) + 1

        selected_filter = st.selectbox(
            "Document Context Filter:",
            options=options,
            index=current_idx,
            help="Filter search and question answering to a specific contract or search across all indexed contracts.",
            key="chat_doc_filter_select",
        )
        if selected_filter == "All Documents (Cross-Contract Search)":
            st.session_state.selected_doc_filter = None
        else:
            st.session_state.selected_doc_filter = selected_filter

    with col_new:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 New Chat", use_container_width=True, help="Reset conversation history without deleting indexed documents."):
            st.session_state.chat_history = []
            st.rerun()

    with col_export:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.session_state.chat_history:
            # Generate export data on demand
            md_content = export_conversation_to_markdown(
                st.session_state.chat_history,
                document_filter=st.session_state.selected_doc_filter,
            )
            time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Export Chat (.md)",
                data=md_content,
                file_name=f"legaleagle_transcript_{time_tag}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download full conversation transcript with citations and without secrets.",
            )

    # Empty State Notice if no documents indexed
    if not indexed_docs:
        st.info(
            "💡 **No contracts are currently indexed in ChromaDB.**\n\n"
            "To get accurate, grounded answers with citations, switch to the **📑 Documents** tab to upload and index your contract PDF."
        )

    st.markdown("---")

    # Display Chat History Messages
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:2rem; text-align:center; color:#94A3B8; margin:1rem 0;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🦅</div>
                <h3 style="color:#F8FAFC; margin:0.25rem 0;">Welcome to LegalEagle</h3>
                <p style="margin:0;">Ask any question about your contracts. Answers are strictly grounded in retrieved clauses with verified page citations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Suggested Prompts
        st.markdown("##### 💡 Suggested Questions:")
        chip_col1, chip_col2 = st.columns(2)
        with chip_col1:
            if st.button("What are the contract termination terms?", use_container_width=True):
                st.session_state.suggested_prompt = "What are the contract termination terms and notice requirements?"
            if st.button("What is the payment fee and billing schedule?", use_container_width=True):
                st.session_state.suggested_prompt = "What is the payment fee and billing schedule?"
        with chip_col2:
            if st.button("What confidentiality covenants exist?", use_container_width=True):
                st.session_state.suggested_prompt = "What confidentiality obligations apply and how long do they survive?"
            if st.button("Which governing law and jurisdiction applies?", use_container_width=True):
                st.session_state.suggested_prompt = "Which governing law and jurisdiction controls this agreement?"
    else:
        for idx, msg in enumerate(st.session_state.chat_history):
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            time_display = f" &nbsp;·&nbsp; <span style='font-size:0.75rem; color:#64748B;'>{timestamp}</span>" if timestamp else ""

            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You**{time_display}", unsafe_allow_html=True)
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**LegalEagle**{time_display}", unsafe_allow_html=True)
                    st.write(content)

                    # Persistent Citations / Source References
                    sources = msg.get("sources", [])
                    if sources:
                        with st.expander(f"📚 {len(sources)} Grounded Source Passage(s) Cited", expanded=False):
                            for s_i, src in enumerate(sources, 1):
                                doc_name = src.get("filename", "Unknown Document")
                                page_num = src.get("page_number", "?")
                                sim_val = src.get("similarity", 0.0)
                                dist_val = src.get("distance", 0.0)
                                chunk_text = src.get("text", "").strip()

                                st.markdown(
                                    f"**Source [{s_i}]** &nbsp;|&nbsp; 📄 **`{doc_name}`** (Page {page_num}) "
                                    f"&nbsp;|&nbsp; Similarity: `{sim_val * 100:.1f}%` (Distance: `{dist_val:.4f}`)"
                                )
                                st.info(chunk_text)

    # Chat Input Box
    prompt_input = st.chat_input("Ask a question about your contract clauses …")

    # Handle suggested prompt click
    if "suggested_prompt" in st.session_state and st.session_state.suggested_prompt:
        prompt_input = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None

    if prompt_input:
        user_query = prompt_input.strip()
        now_str = datetime.datetime.now().strftime("%H:%M")

        # 1. Append user turn to session chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": now_str,
        })

        # 2. Call QA service with document filter and past turns
        with st.spinner("Analyzing contract clauses and retrieving grounded evidence..."):
            filter_to_use = st.session_state.selected_doc_filter
            qa_res = answer_contract_question(
                question=user_query,
                filename=filter_to_use,
                top_k=4,
                collection_name=DEFAULT_COLLECTION_NAME,
                chat_history=st.session_state.chat_history[:-1],
            )

        # 3. Append assistant turn
        if qa_res.error:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Notice:** {qa_res.error}",
                "sources": qa_res.sources,
                "timestamp": now_str,
                "is_error": True,
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": qa_res.answer,
                "sources": qa_res.sources,
                "timestamp": now_str,
                "is_error": False,
            })

        st.rerun()


def render_documents_page():
    """Render Document Management, Ingestion, and Vector Store Inspection."""
    st.markdown("### 📑 Document & Vector Store Management")
    st.markdown(
        "Upload PDF contracts, inspect extracted text, generate 384-d semantic embeddings, "
        "and manage documents indexed in local persistent ChromaDB storage."
    )

    tab_upload, tab_library, tab_search = st.tabs([
        "📤 Upload & Process Contract",
        "📚 Indexed Documents Library",
        "🔍 Semantic Search Tester",
    ])

    # ── TAB 1: Upload & Process Contract ─────────────────────
    with tab_upload:
        st.markdown("#### 1. Upload Contract PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Select a legal contract in PDF format.",
            key="doc_tab_uploader",
        )

        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id != st.session_state.get("last_uploaded_id"):
                st.session_state.last_uploaded_id = file_id
                st.session_state.pdf_uploaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_size = uploaded_file.size
                st.session_state.upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.chunks = []
                st.session_state.embedding_status = None
                st.session_state.embedding_error = None
                st.session_state.is_indexed = False
                st.session_state.index_status = None
                st.session_state.index_error = None
                st.session_state.indexed_chunk_count = 0

                with st.spinner("📖 Extracting text with PyMuPDF..."):
                    file_bytes = uploaded_file.read()
                    parse = extract_text_from_pdf(file_bytes, uploaded_file.name)
                    st.session_state.parse_result = parse

        if st.session_state.pdf_uploaded and st.session_state.parse_result:
            parse = st.session_state.parse_result

            # Document metrics
            size_kb = st.session_state.pdf_size / 1024
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="label">Filename</div>
                        <div class="value" style="font-size:0.85rem;">{st.session_state.pdf_name}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">File Size</div>
                        <div class="value">{size_kb:.1f} KB</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Total Pages</div>
                        <div class="value">{parse.page_count}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Total Characters</div>
                        <div class="value">{parse.total_chars:,}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if parse.error:
                st.error(f"❌ Extraction error: {parse.error}")
            if parse.warnings:
                for w in parse.warnings:
                    st.warning(f"⚠️ {w}")

            # Extracted Text Preview
            with st.expander("📝 Extracted Text Preview", expanded=False):
                preview = get_text_preview(parse, max_chars=800)
                if preview:
                    st.code(preview, language=None)
                else:
                    st.caption("No text extracted.")

            # Page-by-Page Viewer
            with st.expander("📄 Page-by-Page Text Inspector", expanded=False):
                for p in parse.pages:
                    st.markdown(f"**Page {p['page_number']}** ({len(p['text']):,} characters):")
                    st.text_area(f"Page {p['page_number']} Content", p["text"], height=100, disabled=True, key=f"tab1_p_{p['page_number']}")

            # Step 2: Chunking & Embeddings
            st.markdown("#### 2. Generate Semantic Chunks & Embeddings")
            col_c_btn, col_c_info = st.columns([1, 2])
            with col_c_btn:
                chunk_embed_btn = st.button("⚡ Generate Embeddings", type="primary", use_container_width=True)

            if chunk_embed_btn:
                with st.spinner("Chunking document (~500 words, ~80 overlap) and encoding with all-MiniLM-L6-v2..."):
                    raw_chunks = chunk_parse_result(parse, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
                    if not raw_chunks:
                        st.warning("No text available to chunk.")
                    else:
                        embedded_chunks, emb_err = generate_chunk_embeddings(raw_chunks)
                        if emb_err:
                            st.error(f"Embedding error: {emb_err}")
                            st.session_state.embedding_status = "error"
                            st.session_state.embedding_error = emb_err
                        else:
                            st.session_state.chunks = embedded_chunks
                            st.session_state.embedding_status = "success"
                            st.session_state.embedding_error = None
                            st.success(f"✅ Generated {len(embedded_chunks)} embedded chunks ({EXPECTED_EMBEDDING_DIM}-d).")

            # Step 3: Indexing into ChromaDB
            if st.session_state.chunks:
                st.markdown("#### 3. Index into ChromaDB Vector Store")
                col_i_btn, col_i_info = st.columns([1, 2])
                with col_i_btn:
                    idx_btn = st.button("📥 Index in ChromaDB", type="primary", use_container_width=True)

                if idx_btn:
                    with st.spinner("Storing chunks in persistent ChromaDB collection..."):
                        indexed_cnt, idx_err = index_document_chunks(
                            st.session_state.chunks,
                            filename=st.session_state.pdf_name,
                        )
                        if idx_err:
                            st.error(f"Indexing error: {idx_err}")
                            st.session_state.index_status = "error"
                            st.session_state.index_error = idx_err
                        else:
                            st.session_state.index_status = "success"
                            st.session_state.is_indexed = True
                            st.session_state.indexed_chunk_count = indexed_cnt
                            st.success(f"✅ Successfully indexed {indexed_cnt} chunks for `{st.session_state.pdf_name}`.")
                            st.rerun()

    # ── TAB 2: Indexed Documents Library & Safe Deletion ─────
    with tab_library:
        st.markdown("#### 📚 Active Vector Store Documents")
        doc_list = get_document_metadata_list(DEFAULT_COLLECTION_NAME)

        if not doc_list:
            st.info("No documents are currently indexed in the vector store.")
        else:
            st.markdown(f"Found **{len(doc_list)}** indexed contract(s):")

            for doc in doc_list:
                fname = doc["filename"]
                chunks_cnt = doc["chunk_count"]
                page_cnt = doc["page_count"]
                pages_str = ", ".join(map(str, doc["pages"]))
                total_w = doc["total_words"]

                with st.container():
                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <h4>📄 {fname}</h4>
                            <span class="badge-success">Indexed in ChromaDB</span>
                            <div style="margin-top:0.5rem; color:#CBD5E1; font-size:0.88rem;">
                                <strong>Pages:</strong> {page_cnt} (Pages: {pages_str}) &nbsp;|&nbsp;
                                <strong>Vectors / Chunks:</strong> {chunks_cnt} &nbsp;|&nbsp;
                                <strong>Total Words:</strong> {total_w:,}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Safe Deletion with Explicit Confirmation
                    col_del_check, col_del_btn = st.columns([3, 1])
                    with col_del_check:
                        confirm_delete = st.checkbox(
                            f"Confirm deletion of `{fname}` from vector store",
                            key=f"del_confirm_{fname}",
                            help="Safely removes this document's vectors only. Other contracts and the database remain completely intact.",
                        )
                    with col_del_btn:
                        delete_clicked = st.button(
                            f"🗑️ Delete Document",
                            key=f"del_btn_{fname}",
                            type="secondary",
                            disabled=not confirm_delete,
                            use_container_width=True,
                        )

                    if delete_clicked:
                        with st.spinner(f"Safely removing vectors for '{fname}'..."):
                            del_count, del_err = delete_document_vectors(fname)
                            if del_err:
                                st.error(f"Failed to delete document: {del_err}")
                            else:
                                st.success(f"✅ Removed {del_count} vectors for `{fname}`. Other documents preserved.")
                                # If the deleted file was the currently uploaded session doc, reset state
                                if st.session_state.pdf_name == fname:
                                    st.session_state.is_indexed = False
                                    st.session_state.indexed_chunk_count = 0
                                st.rerun()

                    st.markdown("<hr style='border:0; border-top:1px solid #1E293B; margin:1rem 0;'>", unsafe_allow_html=True)

    # ── TAB 3: Semantic Search Tester ────────────────────────
    with tab_search:
        st.markdown("#### 🔍 Test Semantic Retrieval Across Documents")
        st.markdown("Query ChromaDB directly to inspect retrieved passages and cosine similarity scores.")

        doc_options = ["All Documents"] + [d["filename"] for d in doc_list] if doc_list else ["All Documents"]
        col_sq, col_sf, col_sk = st.columns([3, 2, 1])

        with col_sq:
            test_query = st.text_input(
                "Search query:",
                placeholder="e.g. termination notice period, compensation fee...",
                key="test_semantic_q",
            )
        with col_sf:
            test_doc_filter = st.selectbox(
                "Filter by document:",
                options=doc_options,
                key="test_semantic_filter",
            )
        with col_sk:
            top_k_select = st.number_input("Top K results:", min_value=1, max_value=5, value=3)

        if st.button("🔎 Run Semantic Search", type="primary"):
            if not test_query.strip():
                st.warning("Please enter a search query.")
            else:
                filter_val = None if test_doc_filter == "All Documents" else test_doc_filter
                with st.spinner("Searching vector database..."):
                    results, s_err = search_contract_chunks(
                        query=test_query.strip(),
                        top_k=top_k_select,
                        filename_filter=filter_val,
                    )
                    if s_err:
                        st.error(f"Search failed: {s_err}")
                    elif not results:
                        st.info("No matching passages found.")
                    else:
                        st.success(f"Found {len(results)} matching passage(s):")
                        for r_idx, res in enumerate(results, 1):
                            sim_pct = res.get("similarity", 0.0) * 100
                            dist_val = res.get("distance", 0.0)
                            with st.expander(f"Result {r_idx}: `{res['filename']}` — Page {res['page_number']} (Similarity: {sim_pct:.1f}%)", expanded=(r_idx == 1)):
                                st.markdown(f"**Chunk ID:** `{res['chunk_id']}` &nbsp;|&nbsp; **Words:** {res['word_count']} &nbsp;|&nbsp; **Cosine Distance:** {dist_val:.4f}")
                                st.info(res["text"])


def render_about_page():
    """Render educational explanation of RAG and qualified legal disclaimer."""
    st.markdown("### ℹ️ About LegalEagle & RAG Architecture")

    # Legal Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ IMPORTANT LEGAL DISCLAIMER:</strong><br>
            LegalEagle is an AI-powered document exploration and assistive technology tool designed for informational and educational purposes only.
            <strong>LegalEagle does NOT provide legal advice, legal opinions, or formal attorney representation.</strong>
            Contract terms, liability limitations, and legal obligations are complex and subject to jurisdictional laws.
            Users should never rely on AI outputs as a substitute for counsel from a qualified, licensed attorney.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🏗️ How Retrieval-Augmented Generation (RAG) Works in LegalEagle")
    st.markdown(
        "Standard large language models (LLMs) can hallucinate details or fail to recall exact clauses from lengthy agreements. "
        "LegalEagle implements a grounded **Retrieval-Augmented Generation (RAG)** pipeline to guarantee factual fidelity:"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ##### 1. 📖 Document Ingestion
            Contracts are parsed using **PyMuPDF**, extracting clean text page-by-page while tracking original pagination and flagging scanned or unreadable pages.

            ##### 2. 🧩 Clause Chunking
            Text is split into semantic chunks (~500 words with ~80-word overlap). Overlap ensures legal clauses that span paragraph or page breaks remain intact.

            ##### 3. 🧠 Dense Embeddings
            Each chunk is transformed into a 384-dimensional dense semantic vector using the `sentence-transformers/all-MiniLM-L6-v2` model.
            """
        )

    with col2:
        st.markdown(
            """
            ##### 4. 🗄️ ChromaDB Vector Indexing
            Vectors and rich metadata (filename, page number, chunk ID) are stored in a persistent local **ChromaDB** collection with HNSW cosine similarity search.

            ##### 5. 🔍 Semantic Retrieval
            When a question is asked, LegalEagle converts the query into an embedding and retrieves the top-K most relevant contract excerpts.

            ##### 6. ⚖️ Grounded LLM Response
            Retrieved passages feed into OpenAI `gpt-4o-mini` with strict system instructions: explain in plain English, cite exact page numbers, acknowledge insufficient evidence, and never fabricate citations.
            """
        )

    st.markdown("---")
    st.markdown("#### 🔒 Security, Privacy & Integrity Commitments")
    st.markdown(
        """
        - **Local Vector Storage:** Embeddings and indexed chunks remain in your local workspace directory (`chroma_db`).
        - **Document Isolation & Safe Deletion:** Re-indexing replaces only that document's records, and deletion safely removes only the chosen contract without wiping other documents.
        - **Zero Secret Leakage:** API keys are read from environment variables or Streamlit secrets and are strictly excluded from conversation exports, logs, and UI displays.
        - **Accurate Citations:** LegalEagle cites only filenames and pages present in retrieved passages.
        """
    )


# ── 7. Router ─────────────────────────────────────────────────
if st.session_state.current_nav == "💬 Chat":
    render_chat_page()
elif st.session_state.current_nav == "📑 Documents":
    render_documents_page()
elif st.session_state.current_nav == "ℹ️ About":
    render_about_page()
