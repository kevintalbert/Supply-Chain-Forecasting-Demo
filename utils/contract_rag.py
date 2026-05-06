"""Lightweight RAG: chunk mock contract, retrieve clauses, fuse with structured pricing."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


def _chunk_text(text: str, max_chars: int = 450) -> List[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    chunks = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def extract_pdf_text(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def build_index(
    pdf_path: str,
    models_dir: str,
) -> Dict[str, Any]:
    """Embed contract chunks; persist for model serving."""
    os.makedirs(models_dir, exist_ok=True)
    text = extract_pdf_text(pdf_path)
    chunks = _chunk_text(text)
    encoder_name = None
    embeddings = None
    vectorizer = None

    if _ST_AVAILABLE:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, normalize_embeddings=True)
            encoder_name = "all-MiniLM-L6-v2"
            vectorizer = None
        except Exception:
            embeddings = None

    if embeddings is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=2048, stop_words="english")
        embeddings = vectorizer.fit_transform(chunks).toarray().astype(np.float32)
        encoder_name = "tfidf_fallback"

    payload = {
        "chunks": chunks,
        "embeddings": embeddings,
        "encoder": encoder_name,
        "vectorizer": vectorizer,
        "source_pdf": pdf_path,
    }
    path = os.path.join(models_dir, "contract_rag_index.joblib")
    import joblib

    joblib.dump(payload, path)
    meta = {"rag_index": path, "num_chunks": len(chunks), "encoder": encoder_name}
    with open(os.path.join(models_dir, "rag_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_index(models_dir: str):
    import joblib

    path = os.path.join(models_dir, "contract_rag_index.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def retrieve(
    query: str,
    index: Dict[str, Any],
    top_k: int = 3,
) -> List[RetrievedChunk]:
    chunks: Sequence[str] = index["chunks"]
    emb = index["embeddings"]
    enc = index["encoder"]

    if enc != "tfidf_fallback" and _ST_AVAILABLE:
        model = SentenceTransformer(enc)
        q = model.encode([query], normalize_embeddings=True)[0]
        sim = emb @ q
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        vec = index.get("vectorizer")
        if vec is None:
            raise ValueError("RAG index missing vectorizer for TF-IDF mode")
        qv = vec.transform([query]).toarray().astype(np.float32)
        sim = cosine_similarity(qv, emb)[0]

    top_idx = np.argsort(-sim)[:top_k]
    return [
        RetrievedChunk(text=chunks[int(i)], score=float(sim[int(i)]), chunk_id=int(i))
        for i in top_idx
    ]


def structured_price_context(
    nsn: str,
    contract_id: str,
    price_df,
    transactions_df,
    spike_month: Optional[str] = None,
) -> Dict[str, Any]:
    """Join warehouse-style frames for narrative grounding."""
    ph = price_df[
        (price_df["nsn"] == nsn) & (price_df["contract_id"] == contract_id)
    ].sort_values("date")
    if ph.empty:
        return {"error": "no price history for nsn/contract"}
    ph = ph.copy()
    ph["mom_change"] = ph["unit_price"].pct_change()
    if spike_month:
        target = pd.Timestamp(spike_month)
    else:
        # largest positive MoM in recent window
        tail = ph.tail(18)
        target = tail.loc[tail["mom_change"].idxmax(), "date"]
    row = ph[ph["date"] == target].iloc[0]
    prev_row = ph[ph["date"] < target].tail(1)
    prev_price = float(prev_row["unit_price"].iloc[0]) if len(prev_row) else None

    tx = transactions_df[
        (transactions_df["nsn"] == nsn)
        & (transactions_df["contract_id"] == contract_id)
    ]
    tx_near = tx[
        (tx["order_date"] >= target - pd.Timedelta(days=45))
        & (tx["order_date"] <= target + pd.Timedelta(days=45))
    ]

    return {
        "nsn": nsn,
        "contract_id": contract_id,
        "spike_month": str(target.date()),
        "unit_price": float(row["unit_price"]),
        "prior_price": prev_price,
        "mom_pct": float(row["mom_change"]) if pd.notna(row["mom_change"]) else None,
        "market_index": float(row["market_index"]),
        "demand_quantity": float(row["demand_quantity"]),
        "nearby_orders": int(len(tx_near)),
        "shipping_codes_nearby": tx_near["shipping_code"].dropna().unique().tolist()[:5],
    }
