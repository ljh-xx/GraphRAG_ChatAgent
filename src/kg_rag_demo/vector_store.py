from __future__ import annotations

import re
import shutil
from collections import Counter
from typing import Any

import chromadb

from .config import CHROMA_DIR, Settings
from .llm import LLMClient
from .models import ChunkRecord

# 检测文本中是否包含中文字符
_HAS_CJK = re.compile(r"[一-鿿㐀-䶿]")


class VectorStore:
    def __init__(self, settings: Settings, llm: LLMClient, reset: bool = False) -> None:
        self.settings = settings
        self.llm = llm
        if reset and CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        self.client: chromadb.ClientAPI | None = None
        self.collection: Any | None = None
        self._open_client()

    def get_existing_ids(self) -> set[str]:
        """返回 Chroma 中已存在的 chunk_id 集合，用于断点续传."""
        collection = self._require_collection()
        count = collection.count()
        if count == 0:
            return set()
        result = collection.get(include=[], limit=count)
        return set(result.get("ids", []))

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> int:
        if not chunks:
            return 0

        collection = self._require_collection()
        texts = [chunk.text for chunk in chunks]
        embeddings = self.llm.embed_documents(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._to_metadata(chunk) for chunk in chunks]
        documents = texts

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.retrieval_k
        query_embedding = self.llm.embed_query(question)
        result = self._query_collection(
            collection=self._require_collection(),
            query_embedding=query_embedding,
            top_k=k,
        )
        hits: list[dict[str, Any]] = []

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            item = dict(meta or {})
            item["chunk_id"] = chunk_id
            item["text"] = text
            item["distance"] = distance
            hits.append(item)
        return hits

    def query_bm25(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.retrieval_k
        rows = self.get_all_chunks()
        if not rows:
            return []
        query_terms = self._tokenize_for_sparse(question)
        if not query_terms:
            return rows[:k]

        doc_freq: Counter[str] = Counter()
        tokenized_docs: list[list[str]] = []
        for row in rows:
            tokens = self._tokenize_for_sparse(str(row.get("text", "")))
            tokenized_docs.append(tokens)
            for token in set(tokens):
                doc_freq[token] += 1

        doc_count = max(1, len(tokenized_docs))
        avgdl = sum(len(tokens) for tokens in tokenized_docs) / doc_count
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, int]] = []

        for idx, tokens in enumerate(tokenized_docs):
            tf = Counter(tokens)
            dl = max(1, len(tokens))
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = self._bm25_idf(doc_count, doc_freq.get(term, 0))
                numerator = tf[term] * (k1 + 1)
                denominator = tf[term] + k1 * (1 - b + b * dl / max(1.0, avgdl))
                score += idf * (numerator / max(1e-9, denominator))
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[dict[str, Any]] = []
        for score, idx in scored[:k]:
            item = dict(rows[idx])
            item["bm25_score"] = score
            hits.append(item)
        return hits

    def get_all_chunks(self) -> list[dict[str, Any]]:
        collection = self._require_collection()
        count = collection.count()
        if count <= 0:
            return []
        result = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        rows: list[dict[str, Any]] = []
        for chunk_id, text, meta in zip(ids, docs, metas):
            item = dict(meta or {})
            item["chunk_id"] = chunk_id
            item["text"] = text
            rows.append(item)
        return rows

    def self_check(self, query_text: str) -> dict[str, Any]:
        query_embedding = self.llm.embed_query(query_text)
        try:
            self.close()
            return self._reopen_and_query(query_embedding)
        finally:
            self._open_client()

    def close(self) -> None:
        if self.client is None:
            return
        self.client.close()
        self.client = None
        self.collection = None

    def _open_client(self) -> None:
        if self.client is not None:
            return
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=self.settings.collection_name)

    def _require_collection(self) -> Any:
        if self.collection is None:
            self._open_client()
        if self.collection is None:
            raise RuntimeError("Chroma collection 未成功初始化。")
        return self.collection

    @staticmethod
    def _query_collection(collection: Any, query_embedding: list[float], top_k: int) -> dict[str, Any]:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def _reopen_and_query(self, query_embedding: list[float]) -> dict[str, Any]:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = client.get_collection(name=self.settings.collection_name)
            count = collection.count()
            if count == 0:
                raise RuntimeError("collection 为空，没有可查询的数据。")

            result = self._query_collection(
                collection=collection,
                query_embedding=query_embedding,
                top_k=1,
            )
            ids = result.get("ids", [[]])[0]
            if not ids:
                raise RuntimeError("能够重新打开 collection，但检索结果为空。")

            return {
                "count": count,
                "top_hit_id": ids[0],
            }
        except Exception as exc:
            raise RuntimeError(f"向量库重开自检失败：{exc}") from exc
        finally:
            client.close()

    @staticmethod
    def _to_metadata(chunk: ChunkRecord) -> dict[str, Any]:
        metadata = {
            "doc_id": chunk.doc_id,
            "source_path": chunk.source_path,
            "title": chunk.title,
            "modality": chunk.modality,
            "page_number": chunk.page_number or -1,
            "order": chunk.order,
        }
        metadata.update(chunk.metadata)
        return metadata

    @staticmethod
    def _tokenize_for_sparse(text: str) -> list[str]:
        normalized = " ".join(text.lower().split())
        if not normalized:
            return []
        if _HAS_CJK.search(normalized):
            try:
                import jieba

                return [t for t in jieba.lcut(normalized) if t.strip()]
            except ImportError:
                pass  # jieba 未安装，回退到单字切分
            return [ch for ch in normalized if not ch.isspace()]
        return [token for token in normalized.split(" ") if token]

    @staticmethod
    def _bm25_idf(doc_count: int, doc_freq: int) -> float:
        import math

        return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
