# ============================================================
# services/vector_store.py  –  ChromaDB Vector Store & Semantic Search
# ============================================================
# In Retrieval-Augmented Generation (RAG), vector databases store
# document chunk embeddings and index them using approximate nearest
# neighbor algorithms (such as HNSW).
#
# When a lawyer or user asks a question, we:
#   1. Convert the query into an embedding vector using all-MiniLM-L6-v2.
#   2. Perform a semantic similarity search against the stored vectors.
#   3. Retrieve the top-K most relevant chunks with accurate page citations.
#
# Key Features:
#   • Persistent storage in the local "chroma_db" folder.
#   • Default collection: "legal_contracts".
#   • Document-level replacement (re-indexing a document replaces only its
#     own records without deleting or affecting other documents).
#   • Enforces 384-dimensional vector validation.
#   • Graceful handling of empty queries and empty collections.
# ============================================================

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PERSIST_DIR: str = "chroma_db"
DEFAULT_COLLECTION_NAME: str = "legal_contracts"
EXPECTED_EMBEDDING_DIM: int = 384

# Module-level cached ChromaDB client to avoid reopening persistent storage
_cached_client = None
_cached_persist_dir = None


def get_chroma_client(persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Get or initialize a persistent ChromaDB client.

    Parameters
    ----------
    persist_directory : str
        Local folder path where vector index and sqlite data are stored.

    Returns
    -------
    chromadb.PersistentClient
    """
    global _cached_client, _cached_persist_dir

    if _cached_client is not None and _cached_persist_dir == persist_directory:
        return _cached_client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "The 'chromadb' package is not installed. "
            "Please install it with: pip install chromadb"
        ) from e

    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    _cached_client = client
    _cached_persist_dir = persist_directory
    return client


def get_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
):
    """
    Retrieve or create the specified collection in persistent ChromaDB storage.
    Configured with cosine distance for semantic similarity.
    """
    client = get_chroma_client(persist_directory)
    # Use cosine distance for normalized embeddings
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_document_chunks(
    chunks: List[Dict[str, Any]],
    filename: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Tuple[int, Optional[str]]:
    """
    Index document chunks into the ChromaDB collection.

    Re-indexing behavior:
      Replaces existing records for the specified `filename` without
      modifying records belonging to other documents.

    Parameters
    ----------
    chunks : List[Dict[str, Any]]
        List of chunk dicts from services.chunking and services.embeddings.
        Each chunk must have 'chunk_id', 'text', and 'embedding' (length 384).
    filename : Optional[str]
        Document filename. If None, inferred from the first chunk.
    collection_name : str
        Target collection name.
    persist_directory : str
        Storage path.

    Returns
    -------
    Tuple[int, Optional[str]]
        (number_of_indexed_chunks, error_message)
    """
    if not chunks:
        return 0, "No chunks provided to index."

    # Infer or normalize filename
    target_filename = filename or chunks[0].get("filename", "unknown_document.pdf")

    # ── Step 1: Validate dimensions and required fields ─────
    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, c in enumerate(chunks):
        chunk_text = c.get("text", "").strip()
        if not chunk_text:
            continue

        emb = c.get("embedding")
        if emb is None:
            return 0, f"Chunk '{c.get('chunk_id', idx)}' is missing an embedding."

        if len(emb) != EXPECTED_EMBEDDING_DIM:
            return 0, (
                f"Embedding dimension mismatch in chunk '{c.get('chunk_id', idx)}': "
                f"expected {EXPECTED_EMBEDDING_DIM}, got {len(emb)}."
            )

        chunk_id = str(c.get("chunk_id", f"{target_filename}_chunk_{idx}"))
        ids.append(chunk_id)
        documents.append(chunk_text)
        embeddings.append(emb)

        # ChromaDB accepts only str, int, float, or bool in metadatas
        metadata_entry = {
            "filename": str(c.get("filename", target_filename)),
            "page_number": int(c.get("page_number", 1)),
            "chunk_id": chunk_id,
            "word_count": int(c.get("word_count", len(chunk_text.split()))),
        }
        metadatas.append(metadata_entry)

    if not ids:
        return 0, "All chunks were empty or lacked valid text."

    # ── Step 2: Replace only this document's previous records
    try:
        collection = get_collection(collection_name, persist_directory)

        # Find existing records for this specific document
        existing = collection.get(where={"filename": target_filename})
        existing_ids = existing.get("ids", []) if existing else []

        if existing_ids:
            logger.info(
                f"Replacing {len(existing_ids)} existing chunks for document '{target_filename}'."
            )
            collection.delete(ids=existing_ids)

        # ── Step 3: Add new chunks to collection ────────────
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids), None

    except Exception as e:
        error_msg = f"Failed to index chunks into ChromaDB: {e}"
        logger.error(error_msg)
        return 0, error_msg


def search_contract_chunks(
    query: str,
    top_k: int = 5,
    filename_filter: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Perform semantic vector similarity search using sentence-transformers query embedding.

    Parameters
    ----------
    query : str
        The user's legal or factual question (e.g., "What are the termination conditions?").
    top_k : int
        Maximum number of relevant chunks to retrieve (default: 5).
    filename_filter : Optional[str]
        If provided, restrict search to chunks from this specific contract.
    collection_name : str
        Collection to search.
    persist_directory : str
        Storage path.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Optional[str]]
        - List of result dictionaries sorted by relevance.
        - Error message if search could not be completed, or None on success.
    """
    # ── Handle empty query ──────────────────────────────────
    if not query or not isinstance(query, str) or not query.strip():
        return [], "Query is empty."

    clean_query = query.strip()

    try:
        collection = get_collection(collection_name, persist_directory)
        total_items = collection.count()

        # ── Handle empty database / collection ──────────────
        if total_items == 0:
            return [], None

        # ── Step 1: Embed query using Sentence Transformers ──
        from services.embeddings import generate_single_embedding

        query_embedding = generate_single_embedding(clean_query)
        if query_embedding is None:
            return [], "Could not generate embedding for query."

        if len(query_embedding) != EXPECTED_EMBEDDING_DIM:
            return [], (
                f"Query embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}, "
                f"got {len(query_embedding)}."
            )

        # ── Step 2: Determine appropriate n_results ─────────
        where_filter = {"filename": filename_filter} if filename_filter else None

        if where_filter:
            # Check how many records match this file
            matching = collection.get(where=where_filter)
            matching_count = len(matching.get("ids", [])) if matching else 0
            if matching_count == 0:
                return [], None
            n_results = min(top_k, matching_count)
        else:
            n_results = min(top_k, total_items)

        if n_results <= 0:
            return [], None

        # ── Step 3: Query ChromaDB collection ───────────────
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        # ── Step 4: Parse & structure results ───────────────
        formatted_results: List[Dict[str, Any]] = []

        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_metas = results.get("metadatas", [[]])[0]
        retrieved_dists = results.get("distances", [[]])[0]
        retrieved_ids = results.get("ids", [[]])[0]

        for doc_id, doc_text, meta, dist in zip(
            retrieved_ids, retrieved_docs, retrieved_metas, retrieved_dists
        ):
            meta_dict = meta if isinstance(meta, dict) else {}
            distance = float(dist) if dist is not None else 0.0

            # Cosine similarity: score between 0.0 and 1.0 (approx 1 - cosine distance)
            similarity = round(max(0.0, min(1.0, 1.0 - distance)), 4)

            formatted_results.append({
                "chunk_id": str(meta_dict.get("chunk_id", doc_id)),
                "text": str(doc_text),
                "filename": str(meta_dict.get("filename", "")),
                "page_number": int(meta_dict.get("page_number", 1)),
                "word_count": int(meta_dict.get("word_count", len(doc_text.split()))),
                "distance": round(distance, 4),
                "similarity": similarity,
            })

        return formatted_results, None

    except Exception as e:
        error_msg = f"Semantic search failed: {e}"
        logger.error(error_msg)
        return [], error_msg


def get_collection_stats(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Dict[str, Any]:
    """
    Return statistics about the vector collection (total chunks, unique documents).
    """
    try:
        collection = get_collection(collection_name, persist_directory)
        count = collection.count()
        if count == 0:
            return {"total_chunks": 0, "documents": []}

        # Inspect stored metadatas to list distinct documents
        data = collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        filenames = sorted(list({m.get("filename") for m in metas if m and "filename" in m}))

        return {
            "total_chunks": count,
            "documents": filenames,
        }
    except Exception as e:
        logger.error(f"Error fetching collection stats: {e}")
        return {"total_chunks": 0, "documents": [], "error": str(e)}


def get_document_metadata_list(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> List[Dict[str, Any]]:
    """
    Return detailed per-document statistics stored in the ChromaDB collection.

    Returns
    -------
    List[Dict[str, Any]]
        Each entry contains:
          - 'filename': Document filename
          - 'chunk_count': Number of vector chunks stored for this document
          - 'page_count': Number of distinct pages spanned by this document
          - 'pages': Sorted list of page numbers
          - 'total_words': Sum of word counts across all chunks
    """
    try:
        collection = get_collection(collection_name, persist_directory)
        count = collection.count()
        if count == 0:
            return []

        data = collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])

        docs_map: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            if not m or not isinstance(m, dict):
                continue
            fname = str(m.get("filename", "unknown"))
            if fname not in docs_map:
                docs_map[fname] = {
                    "filename": fname,
                    "chunk_count": 0,
                    "pages": set(),
                    "total_words": 0,
                }
            docs_map[fname]["chunk_count"] += 1
            if "page_number" in m:
                docs_map[fname]["pages"].add(int(m["page_number"]))
            if "word_count" in m:
                docs_map[fname]["total_words"] += int(m.get("word_count", 0))

        result = []
        for fname in sorted(docs_map.keys()):
            info = docs_map[fname]
            sorted_pages = sorted(list(info["pages"]))
            result.append({
                "filename": fname,
                "chunk_count": info["chunk_count"],
                "page_count": len(sorted_pages),
                "pages": sorted_pages,
                "total_words": info["total_words"],
            })

        return result

    except Exception as e:
        logger.error(f"Error fetching detailed document metadata: {e}")
        return []


def delete_document_vectors(
    filename: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Tuple[int, Optional[str]]:
    """
    Safely delete ONLY the vector chunks and metadata belonging to the specified document.

    Safety Guarantee:
      Does NOT touch other documents in the database and never deletes the entire collection.

    Parameters
    ----------
    filename : str
        Target contract filename to delete.
    collection_name : str
        Target collection.
    persist_directory : str
        ChromaDB folder path.

    Returns
    -------
    Tuple[int, Optional[str]]
        (number_of_deleted_chunks, error_message)
    """
    if not filename or not isinstance(filename, str) or not filename.strip():
        return 0, "No document filename specified for deletion."

    target_name = filename.strip()

    try:
        collection = get_collection(collection_name, persist_directory)
        existing = collection.get(where={"filename": target_name})
        existing_ids = existing.get("ids", []) if existing else []

        if not existing_ids:
            return 0, None  # Nothing to delete, successful no-op

        collection.delete(ids=existing_ids)
        logger.info(
            f"Successfully deleted {len(existing_ids)} chunks for document '{target_name}'."
        )
        return len(existing_ids), None

    except Exception as e:
        err = f"Failed to delete document '{target_name}' from vector store: {e}"
        logger.error(err)
        return 0, err


def clear_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> bool:
    """Delete all records in the collection (useful for tests or hard resets)."""
    try:
        client = get_chroma_client(persist_directory)
        client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")
        return False


