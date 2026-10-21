# ============================================================
# services/embeddings.py  –  Vector Embedding Module
# ============================================================
# In AI and RAG systems, text cannot be understood directly by
# mathematical distance formulas. We convert text chunks into
# dense numerical vectors called "embeddings".
#
# Each embedding is an array of floating-point numbers (e.g. 384 dimensions)
# that captures the semantic meaning of that contract clause.
#
# Model used:
#   - sentence-transformers/all-MiniLM-L6-v2
#   - Produces 384-dimensional embeddings
#   - Lightweight, extremely fast, and ideal for local legal tech prototypes
# ============================================================

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM: int = 384

# Module-level cache so we load the model once into memory
_cached_model = None
_cached_model_name = None


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load and cache the SentenceTransformer model.

    Returns
    -------
    SentenceTransformer instance, or raises RuntimeError if loading fails.
    """
    global _cached_model, _cached_model_name

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "The 'sentence-transformers' package is not installed. "
            "Please run: pip install sentence-transformers"
        ) from e

    try:
        logger.info(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)
        _cached_model = model
        _cached_model_name = model_name
        return _cached_model
    except Exception as e:
        error_msg = (
            f"Failed to load embedding model '{model_name}'. "
            f"If you are offline, ensure the model has been downloaded previously. "
            f"Details: {e}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def generate_single_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Optional[List[float]]:
    """
    Generate a 384-dimensional embedding vector for a single string.

    Parameters
    ----------
    text : str
        The input text string.
    model_name : str
        Name of the sentence-transformers model (default: all-MiniLM-L6-v2).

    Returns
    -------
    Optional[List[float]]
        List of 384 floats, or None if the input text is empty or invalid.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return None

    model = load_embedding_model(model_name)
    embedding = model.encode(text.strip(), convert_to_numpy=True)
    return embedding.tolist()


def generate_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Generate and attach embeddings to a list of text chunks.

    Parameters
    ----------
    chunks : List[Dict[str, Any]]
        List of chunk dictionaries (from `services.chunking`).
        Each chunk must have at least a "text" key.
    model_name : str
        Name of the sentence-transformers model.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Optional[str]]
        - A new list of chunk dicts, each with an "embedding" key (list of 384 floats).
        - An error string if embedding generation failed, or None on success.
    """
    if not chunks:
        return [], None

    # Filter out empty or whitespace-only chunks defensively
    valid_chunks = [c for c in chunks if c.get("text", "").strip()]
    if not valid_chunks:
        return [], "No non-empty text chunks found to embed."

    try:
        model = load_embedding_model(model_name)
    except Exception as e:
        return [], str(e)

    texts_to_embed = [c["text"].strip() for c in valid_chunks]

    try:
        # Encode all chunks in batches
        raw_embeddings = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        embedded_chunks = []
        for chunk, emb in zip(valid_chunks, raw_embeddings):
            # Create a shallow copy so we do not mutate the caller's objects
            chunk_copy = dict(chunk)
            chunk_copy["embedding"] = emb.tolist()
            chunk_copy["embedding_dim"] = len(emb)
            embedded_chunks.append(chunk_copy)

        return embedded_chunks, None

    except Exception as e:
        error_msg = f"Error during embedding computation: {e}"
        logger.error(error_msg)
        return [], error_msg
