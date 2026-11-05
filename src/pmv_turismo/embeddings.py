from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pmv_turismo.common import POI_URN_PREFIX, load_poi_list, processed_dir

DEFAULT_MODEL_ID = "google/embeddinggemma-300m"

logger = logging.getLogger(__name__)


def _format_poi_text(poi: Dict[str, Any]) -> str:
    """Build the text to embed using name, description, and categories."""
    name = str(poi.get("name") or "").strip()
    desc = str(poi.get("description") or "").strip()
    categories = [str(c) for c in (poi.get("category") or []) if c]
    subcategory = poi.get("subcategory")
    if subcategory and subcategory not in categories:
        categories.append(str(subcategory).replace("_", " "))
    category_str = ", ".join(categories) or "unknown"
    text = f"Name: {name}\nDescription: {desc}\nCategory: {category_str}"
    if poi.get("heritageDesignation"):
        text += "\nProtected heritage"
    return text


def embeddings_path(config: dict) -> Path:
    emb_cfg = config.get("embeddings", {})
    return Path(emb_cfg.get("output_path", processed_dir(config) / "embeddings.jsonl"))


def generate_embeddings(
    config: dict,
    poi_data: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate embeddings for POIs and store them in a JSONL file."""
    # Imported lazily: torch/sentence-transformers are only needed for this step.
    from sentence_transformers import SentenceTransformer

    emb_cfg = config.get("embeddings", {})
    model_id = emb_cfg.get("model_id", DEFAULT_MODEL_ID)
    device = emb_cfg.get("device", "cpu")
    batch_size = int(emb_cfg.get("batch_size", 16))
    normalize = bool(emb_cfg.get("normalize", True))
    show_progress = bool(emb_cfg.get("show_progress", True))
    output_path = embeddings_path(config)

    poi_items = poi_data or load_poi_list(processed_dir(config) / "poi.jsonld")

    texts: List[str] = []
    poi_ids: List[str] = []
    for poi in poi_items:
        poi_id = poi.get("id")
        if not poi_id:
            continue
        poi_ids.append(str(poi_id))
        texts.append(_format_poi_text(poi))

    if not texts:
        raise ValueError("No POIs with ids found to embed")

    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    encode = model.encode_document if hasattr(model, "encode_document") else model.encode
    vectors = encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for poi_id, text, vec in zip(poi_ids, texts, vectors):
            record = {
                "id": poi_id,
                "model": model_id,
                "embedding": [float(x) for x in vec.tolist()],
                "text": text,
            }
            output.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output


_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
# A request arriving during the API warm-up waits for it instead of loading a second copy.
_MODEL_LOCK = threading.Lock()
_QUERY_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def encode_query(config: dict, text: str) -> Optional[np.ndarray]:
    """Unit vector for a free-text interest, in the same space as the POIs.

    The model is loaded on first use and kept in memory; vectors are cached
    per text so repeated requests from the same profile are free. Returns
    None when the optional `embeddings` extra is not installed, so callers
    simply ignore the text.
    """
    emb_cfg = config.get("embeddings", {})
    model_id = emb_cfg.get("model_id", DEFAULT_MODEL_ID)
    key = (model_id, text.strip())
    if key in _QUERY_CACHE:
        return _QUERY_CACHE[key]

    model_key = (model_id, emb_cfg.get("device", "cpu"))
    with _MODEL_LOCK:
        if model_key not in _MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("sentence-transformers not installed; free-text interests are ignored")
                return None
            _MODEL_CACHE[model_key] = SentenceTransformer(model_id, device=model_key[1], trust_remote_code=True)
    model = _MODEL_CACHE[model_key]
    encode = model.encode_query if hasattr(model, "encode_query") else model.encode
    vec = np.asarray(encode([key[1]], convert_to_numpy=True)[0], dtype=np.float32)
    vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
    _QUERY_CACHE[key] = vec
    return vec


class EmbeddingIndex:
    """Embedding matrix plus id lookup (accepts bare OSM ids or NGSI-LD URNs)."""

    def __init__(self, matrix: np.ndarray, id_to_idx: Dict[str, int]):
        self.matrix = matrix
        self.id_to_idx = id_to_idx
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        self.units = matrix / norms

    def index_of(self, poi_id: str) -> Optional[int]:
        if poi_id in self.id_to_idx:
            return self.id_to_idx[poi_id]
        return self.id_to_idx.get(f"{POI_URN_PREFIX}{poi_id}")

    def indices_of(self, poi_ids: List[str]) -> List[int]:
        return [idx for pid in poi_ids if (idx := self.index_of(pid)) is not None]

    def resolve(self, poi_id: str) -> Optional[Tuple[str, np.ndarray]]:
        for key in (poi_id, f"{POI_URN_PREFIX}{poi_id}"):
            if key in self.id_to_idx:
                return key, self.matrix[self.id_to_idx[key]]
        return None


_INDEX_CACHE: Dict[str, Tuple[float, EmbeddingIndex]] = {}


def load_embedding_index(path: str | Path) -> EmbeddingIndex:
    """Load embeddings JSONL, cached in-process until the file changes."""
    embedding_path = Path(path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embedding_path}")
    mtime = embedding_path.stat().st_mtime
    cached = _INDEX_CACHE.get(embedding_path.as_posix())
    if cached and cached[0] == mtime:
        return cached[1]

    id_to_idx: Dict[str, int] = {}
    vectors: List[List[float]] = []
    for line in embedding_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        poi_id = obj.get("id")
        vec = obj.get("embedding")
        if poi_id and isinstance(vec, list):
            id_to_idx[str(poi_id)] = len(vectors)
            vectors.append(vec)
    if not vectors:
        raise ValueError(f"No embeddings loaded from {embedding_path}")

    index = EmbeddingIndex(np.asarray(vectors, dtype=np.float32), id_to_idx)
    _INDEX_CACHE[embedding_path.as_posix()] = (mtime, index)
    return index
