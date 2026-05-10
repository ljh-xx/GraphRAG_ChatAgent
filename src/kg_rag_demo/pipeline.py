from __future__ import annotations
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypedDict

from .chunking import chunk_documents
from .config import DATA_DIR, LONG_MEMORY_PATH, SHORT_MEMORY_PATH, Settings, ensure_project_dirs
from .graph_store import GraphStore
from .llm import LLMClient
from .memory_store import MemoryStore, MemoryTurn
from .models import ChunkRecord, QueryAnswer, RelationEdge
from .parsers import DocumentParser
from .vector_store import VectorStore


class AskState(TypedDict):
    question: str
    intent: str
    intent_reason: str
    should_write_long_memory: bool
    long_memory_write_reason: str
    used_retrieval: bool
    memory_context: str
    search_queries: list[str]
    dense_query_hits: dict[str, list[dict[str, Any]]]
    bm25_query_hits: dict[str, list[dict[str, Any]]]
    chunk_hits: list[dict[str, object]]
    query_entities: list[str]
    graph_query_entities: list[str]
    graph_hits: list[dict[str, object]]
    answer: str
    final_prompt: str
    debug_info: dict[str, Any]


@dataclass
class AskRuntimeConfig:
    enable_query_rewrite: bool = True
    enable_hybrid_retrieval: bool = True
    enable_rerank: bool = True
    enable_evidence_compression: bool = True
    enable_graph_entity_synonyms: bool = True
    enable_graph_multi_hop: bool = True
    enable_graph_hit_dedup: bool = True
    enable_graph_hit_rerank: bool = True
    enable_graph_hit_truncate: bool = True
    debug_mode: bool = False
    retrieval_k: int = 5
    retrieval_candidate_k: int = 12
    rrf_k: int = 60
    graph_max_hops: int = 2
    graph_synonyms_per_entity: int = 2
    graph_top_k: int = 8
    graph_score_exact_match_weight: float = 3.0
    graph_score_partial_match_weight: float = 1.5
    graph_score_hop_weight: float = 1.0
    graph_score_relation_weight: float = 0.4
    graph_score_evidence_weight: float = 0.3
    graph_score_source_weight: float = 0.2


class DemoPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        progress_callback: Callable[[str], None] | None = None,
        ask_config: AskRuntimeConfig | None = None,
        *,
        enable_graph: bool = True,
        reset_vector_store: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.progress_callback = progress_callback or print
        self.enable_graph = enable_graph
        ensure_project_dirs()
        self.parser = DocumentParser(progress_callback=self._log)
        self.llm = LLMClient(self.settings)
        self.vector_store = VectorStore(self.settings, self.llm, reset=reset_vector_store)
        self.memory_store = MemoryStore(
            short_term_path=SHORT_MEMORY_PATH,
            long_term_path=LONG_MEMORY_PATH,
            short_term_max_turns=self.settings.short_memory_max_turns,
            long_term_max_turns=self.settings.long_memory_max_turns,
        )
        self.graph_store: GraphStore | None = None
        if enable_graph:
            self.graph_store = GraphStore(
                uri=self.settings.neo4j_uri,
                username=self.settings.neo4j_username,
                password=self.settings.neo4j_password,
            )
        self.ask_config = ask_config or AskRuntimeConfig(
            enable_query_rewrite=self.settings.enable_query_rewrite,
            enable_hybrid_retrieval=self.settings.enable_hybrid_retrieval,
            enable_rerank=self.settings.enable_rerank,
            enable_evidence_compression=self.settings.enable_evidence_compression,
            enable_graph_entity_synonyms=self.settings.enable_graph_entity_synonyms,
            enable_graph_multi_hop=self.settings.enable_graph_multi_hop,
            enable_graph_hit_dedup=self.settings.enable_graph_hit_dedup,
            enable_graph_hit_rerank=self.settings.enable_graph_hit_rerank,
            enable_graph_hit_truncate=self.settings.enable_graph_hit_truncate,
            debug_mode=self.settings.debug_mode,
            retrieval_k=self.settings.retrieval_k,
            retrieval_candidate_k=max(self.settings.retrieval_k, self.settings.retrieval_candidate_k),
            rrf_k=self.settings.rrf_k,
            graph_max_hops=self.settings.graph_max_hops,
            graph_synonyms_per_entity=self.settings.graph_synonyms_per_entity,
            graph_top_k=self.settings.graph_top_k,
            graph_score_exact_match_weight=self.settings.graph_score_exact_match_weight,
            graph_score_partial_match_weight=self.settings.graph_score_partial_match_weight,
            graph_score_hop_weight=self.settings.graph_score_hop_weight,
            graph_score_relation_weight=self.settings.graph_score_relation_weight,
            graph_score_evidence_weight=self.settings.graph_score_evidence_weight,
            graph_score_source_weight=self.settings.graph_score_source_weight,
        )
        self.ask_graph = self._build_ask_graph()

    def close(self) -> None:
        self.vector_store.close()
        if self.graph_store is not None:
            self.graph_store.close()

    def ingest(self, data_dir: str | None = None) -> dict[str, int]:
        graph_store = self._require_graph_store()
        docs, chunks, vector_inserted = self._ingest_vectors(data_dir=data_dir, prefix="ingest")

        self._log("[ingest] 初始化 Neo4j 约束")
        graph_store.ensure_constraints()

        # 断点续传：跳过已处理的 chunk
        processed_ids = graph_store.get_processed_chunk_ids()
        remaining = [c for c in chunks if c.chunk_id not in processed_ids]
        skipped = len(chunks) - len(remaining)
        if skipped > 0:
            self._log(f"[ingest] 检测到 {skipped} 个已处理 chunk，自动跳过")

        if not remaining:
            self._log("[ingest] 没有新的 Chunk 需要处理，图谱构建跳过")
            return {
                "documents": len(docs),
                "vector_chunks": vector_inserted,
                "graph_chunks": len(chunks),
                "new_graph_chunks": 0,
                "relations": 0,
                "skipped_chunks": skipped,
            }

        total = len(remaining)
        original_total = len(chunks)
        graph_extract_workers = max(1, self.settings.graph_extract_workers)
        self._log(f"[graph] 开始处理 {total} 个新 Chunk（跳过 {skipped} 个），并发数: {graph_extract_workers}")

        relation_count = 0
        future_to_chunk: dict[Future[list[RelationEdge]], tuple[int, ChunkRecord]] = {}
        completed = 0
        next_idx = 1
        heartbeat_seconds = max(1.0, self.settings.graph_progress_heartbeat_seconds)
        last_heartbeat = time.monotonic()

        def submit(chunk_index: int, chunk_record: ChunkRecord, executor: ThreadPoolExecutor) -> None:
            nonlocal next_idx
            self._log(
                f"[graph] 处理 chunk {chunk_index}/{total} (总进度 {skipped + chunk_index}/{original_total}): "
                f"{chunk_record.title} | 页码 {chunk_record.page_number if chunk_record.page_number is not None else '-'} | 长度 {len(chunk_record.text)}"
            )
            self._log(f"[graph] 提交抽取任务: {chunk_record.chunk_id}")
            future = executor.submit(self._extract_graph_relations, chunk_record.text)
            future_to_chunk[future] = (chunk_index, chunk_record)
            next_idx += 1

        with ThreadPoolExecutor(max_workers=graph_extract_workers, thread_name_prefix="graph-extract") as executor:
            while next_idx <= total and len(future_to_chunk) < graph_extract_workers:
                submit(next_idx, remaining[next_idx - 1], executor)

            while future_to_chunk:
                done, _ = wait(
                    list(future_to_chunk.keys()),
                    timeout=heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    pending = total - next_idx + 1
                    self._log(
                        f"[graph] 抽取进行中，已完成 {completed}/{total} (总进度 {skipped + completed}/{original_total})，"
                        f"进行中 {len(future_to_chunk)}，待提交 {max(0, pending)}"
                    )
                    last_heartbeat = time.monotonic()
                    continue

                for future in done:
                    index, chunk = future_to_chunk.pop(future)
                    try:
                        relations = future.result()
                    except Exception as exc:
                        completed += 1
                        self._log(f"[graph] 抽取失败 {index}/{total}: {chunk.chunk_id} | {exc}")
                    else:
                        completed += 1
                        self._log(f"[graph] 抽取完成 {index}/{total}，得到 {len(relations)} 条关系: {chunk.chunk_id}")
                        graph_store.upsert_chunk(chunk)
                        relation_count += graph_store.upsert_relations(chunk, relations)
                        self._log(f"[graph] 已累计写入 {relation_count} 条关系")

                    while next_idx <= total and len(future_to_chunk) < graph_extract_workers:
                        submit(next_idx, remaining[next_idx - 1], executor)

                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_seconds:
                    pending = total - next_idx + 1
                    self._log(
                        f"[graph] 抽取进行中，已完成 {completed}/{total} (总进度 {skipped + completed}/{original_total})，"
                        f"进行中 {len(future_to_chunk)}，待提交 {max(0, pending)}"
                    )
                    last_heartbeat = now

        self._log("[ingest] 全部流程完成")
        return {
            "documents": len(docs),
            "vector_chunks": vector_inserted,
            "graph_chunks": original_total,
            "new_graph_chunks": total,
            "relations": relation_count,
            "skipped_chunks": skipped,
        }

    def ingest_vectors(self, data_dir: str | None = None) -> dict[str, object]:
        docs, chunks, inserted_chunks = self._ingest_vectors(data_dir=data_dir, prefix="vector")
        self_check = self._self_check_vector_store(chunks)
        self._log("[vector] 全部流程完成")
        return {
            "documents": len(docs),
            "chunks": inserted_chunks,
            "vector_self_check": self_check["status"],
        }

    def ask(self, question: str) -> QueryAnswer:
        initial_state: AskState = {
            "question": question,
            "intent": "need_retrieve",
            "intent_reason": "",
            "should_write_long_memory": False,
            "long_memory_write_reason": "",
            "used_retrieval": False,
            "memory_context": "",
            "search_queries": [],
            "dense_query_hits": {},
            "bm25_query_hits": {},
            "chunk_hits": [],
            "query_entities": [],
            "graph_query_entities": [],
            "graph_hits": [],
            "answer": "",
            "final_prompt": "",
            "debug_info": {},
        }
        if self.ask_graph is None:
            result = self._ask_without_graph(initial_state)
        else:
            result = self.ask_graph.invoke(initial_state)
        result["debug_info"] = {
            **result["debug_info"],
            "route_summary": {
                "intent": result["intent"],
                "intent_reason": result["intent_reason"],
                "used_retrieval": result["used_retrieval"],
                "long_memory_should_write": result["should_write_long_memory"],
                "long_memory_write_reason": result["long_memory_write_reason"],
            },
        }
        self._remember_turn(result)
        return QueryAnswer(
            answer=result["answer"],
            chunk_hits=result["chunk_hits"],
            graph_hits=result["graph_hits"],
            query_entities=result["query_entities"],
            debug_info=result["debug_info"],
        )

    def debug_context(self, question: str) -> dict[str, object]:
        answer = self.ask(question)
        return asdict(answer)

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _extract_graph_relations(self, text: str) -> list[RelationEdge]:
        _, relations = self.llm.extract_graph(text)
        return relations

    def _ingest_vectors(self, data_dir: str | None, prefix: str) -> tuple[list[object], list[ChunkRecord], int]:
        target_dir = data_dir or str(DATA_DIR)
        self._log(f"[{prefix}] 开始构建向量索引，数据目录: {target_dir}")
        docs = self.parser.parse_directory(target_dir)
        self._log(f"[{prefix}] 文档解析完成，共得到 {len(docs)} 条记录，开始切块")
        chunks = chunk_documents(docs, self.settings)
        self._log(f"[{prefix}] 切块完成，共生成 {len(chunks)} 个 chunk")

        # 跳过已入库的 chunk，避免重复 embedding
        existing_ids = self.vector_store.get_existing_ids()
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            self._log(f"[{prefix}] 检测到 {skipped} 个已入库 chunk，跳过 embedding")

        if new_chunks:
            batch_size = self.settings.vector_batch_size
            total = len(new_chunks)
            inserted = 0
            for i in range(0, total, batch_size):
                batch = new_chunks[i : i + batch_size]
                inserted += self.vector_store.upsert_chunks(batch)
                self._log(f"[{prefix}] 向量写入进度: {min(i + batch_size, total)}/{total}")
            self._log(f"[{prefix}] 向量写入完成，本次写入 {inserted} 个（跳过 {skipped} 个）")
        else:
            inserted = 0
            self._log(f"[{prefix}] 所有 chunk 均已入库，无需写入")

        return docs, chunks, inserted

    def _self_check_vector_store(self, chunks: list[ChunkRecord]) -> dict[str, str]:
        if not chunks:
            self._log("[vector] 跳过自检：没有生成任何 chunk")
            return {"status": "skipped"}

        probe_text = chunks[0].text.strip() or chunks[0].title.strip() or "测试"
        self._log("[vector] 开始自检：重新打开 Chroma 并执行一次检索")
        check_result = self.vector_store.self_check(probe_text)
        self._log(
            f"[vector] 自检通过：collection={check_result['count']}，"
            f"top_hit={check_result['top_hit_id']}"
        )
        return {"status": "passed"}

    def _require_graph_store(self) -> GraphStore:
        if self.graph_store is None:
            raise RuntimeError("当前 pipeline 未启用 Neo4j，无法构建知识图谱。")
        return self.graph_store

    def _build_ask_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError:
            self._log("[ask] 未安装 langgraph，回退为顺序执行。")
            return None

        graph = StateGraph(AskState)
        graph.add_node("prepare_queries", self._ask_prepare_queries)
        graph.add_node("classify_long_memory_write", self._ask_classify_long_memory_write)
        graph.add_node("classify_intent", self._ask_classify_intent)
        graph.add_node("vector_retrieve", self._ask_vector_retrieve)
        graph.add_node("extract_entities", self._ask_extract_entities)
        graph.add_node("graph_retrieve", self._ask_graph_retrieve)
        graph.add_node("answer_by_memory", self._ask_answer_by_memory)
        graph.add_node("answer_without_retrieval", self._ask_answer_without_retrieval)
        graph.add_node("answer_question", self._ask_answer_question)

        graph.add_edge(START, "prepare_queries")
        graph.add_edge("prepare_queries", "classify_long_memory_write")
        graph.add_edge("classify_long_memory_write", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._ask_route_after_intent,
            {
                "vector_retrieve": "vector_retrieve",
                "answer_by_memory": "answer_by_memory",
                "answer_without_retrieval": "answer_without_retrieval",
            },
        )
        graph.add_edge("vector_retrieve", "extract_entities")
        graph.add_conditional_edges(
            "extract_entities",
            self._ask_route_after_entity_extract,
            {
                "graph_retrieve": "graph_retrieve",
                "answer_question": "answer_question",
            },
        )
        graph.add_edge("graph_retrieve", "answer_question")
        graph.add_edge("answer_by_memory", "answer_question")
        graph.add_edge("answer_without_retrieval", "answer_question")
        graph.add_edge("answer_question", END)
        return graph.compile()

    def _ask_without_graph(self, initial_state: AskState) -> AskState:
        state = self._ask_prepare_queries(initial_state)
        state = self._ask_classify_long_memory_write(state)
        state = self._ask_classify_intent(state)
        intent_route = self._ask_route_after_intent(state)
        if intent_route == "answer_by_memory":
            state = self._ask_answer_by_memory(state)
        elif intent_route == "answer_without_retrieval":
            state = self._ask_answer_without_retrieval(state)
        else:
            state = self._ask_vector_retrieve(state)
            state = self._ask_extract_entities(state)
            if self._ask_route_after_entity_extract(state) == "graph_retrieve":
                state = self._ask_graph_retrieve(state)
        state = self._ask_answer_question(state)
        return state

    def _ask_prepare_queries(self, state: AskState) -> AskState:
        question = state["question"].strip()
        self._log(f"[ask] 收到问题: {question}")
        memory_context = self._build_agent_memory_context()
        return {
            **state,
            "question": question,
            "memory_context": memory_context,
            "debug_info": {
                **state["debug_info"],
                "search_queries": [],
                "memory_context_preview": memory_context[:800],
            },
        }

    def _ask_classify_long_memory_write(self, state: AskState) -> AskState:
        recent_turns = self.memory_store.recent_context(limit=self.settings.memory_recent_turns_for_intent)
        try:
            decision = self.llm.classify_long_memory_write_intent(state["question"], recent_turns)
            should_write = bool(decision.get("should_write", False))
            reason = str(decision.get("reason", "")).strip() or "fallback"
        except Exception as exc:
            should_write = False
            reason = f"classifier_error:{exc}"
        if self.ask_config.debug_mode:
            self._log(f"[ask][debug] 长期记忆写入意图: {should_write} | {reason}")
        return {
            **state,
            "should_write_long_memory": should_write,
            "long_memory_write_reason": reason,
            "debug_info": {
                **state["debug_info"],
                "long_memory_should_write": should_write,
                "long_memory_write_reason": reason,
            },
        }

    def _ask_classify_intent(self, state: AskState) -> AskState:
        recent_turns = self.memory_store.recent_context(limit=self.settings.memory_recent_turns_for_intent)
        memory_match, similarity = self._match_retrieved_memory(state["question"])
        if memory_match is not None and similarity >= self.settings.memory_reuse_similarity_threshold:
            intent = "already_retrieved"
            reason = f"memory_similarity={similarity:.3f}"
        else:
            domain_keywords = [
                item.strip()
                for item in self.settings.domain_keywords.split(",")
                if item.strip()
            ]
            classify = self.llm.classify_intent_with_context(
                question=state["question"],
                recent_turns=recent_turns,
                domain_keywords=domain_keywords,
            )
            intent = classify["intent"]
            reason = classify["reason"]
            if intent == "already_retrieved" and memory_match is None:
                intent = "need_retrieve"
                reason = "no_memory_match_fallback_to_retrieve"

        intent_zh = {
            "unrelated": "无关",
            "already_retrieved": "已检索",
            "need_retrieve": "需检索",
        }.get(intent, "需检索")
        if self.ask_config.debug_mode:
            self._log(f"[ask][debug] 意图识别: {intent_zh}({intent}) | {reason}")
            if memory_match is not None:
                self._log(
                    f"[ask][debug] 相关短期记忆命中: turn_id={memory_match.turn_id}, similarity={similarity:.3f}"
                )
        return {
            **state,
            "intent": intent,
            "intent_reason": reason,
            "debug_info": {
                **state["debug_info"],
                "intent_result": intent,
                "intent_result_zh": intent_zh,
                "intent_reason": reason,
                "memory_match_similarity": similarity,
                "memory_match_turn_id": memory_match.turn_id if memory_match else "",
            },
        }

    @staticmethod
    def _ask_route_after_intent(state: AskState) -> str:
        if state["intent"] == "already_retrieved":
            return "answer_by_memory"
        if state["intent"] == "unrelated":
            return "answer_without_retrieval"
        return "vector_retrieve"

    def _ask_vector_retrieve(self, state: AskState) -> AskState:
        search_queries = self._build_search_queries(state["question"])
        if self.ask_config.debug_mode:
            self._log("[ask][debug] 本次检索查询如下：")
            for idx, query in enumerate(search_queries, start=1):
                self._log(f"[ask][debug] query_{idx}: {query}")
        dense_query_hits: dict[str, list[dict[str, Any]]] = {}
        bm25_query_hits: dict[str, list[dict[str, Any]]] = {}
        dense_candidates: list[dict[str, Any]] = []
        for query in search_queries:
            dense_hits = self.vector_store.query(query, top_k=self.ask_config.retrieval_candidate_k)
            dense_query_hits[query] = dense_hits
            for hit in dense_hits:
                item = dict(hit)
                item["source_query"] = query
                item["retriever"] = "dense"
                dense_candidates.append(item)

            if self.ask_config.debug_mode:
                self._log(f"[ask][debug] dense | query={query} 命中 {len(dense_hits)} 个 chunk")
                for idx, hit in enumerate(dense_hits, start=1):
                    self._log(
                        f"[ask][debug]   - {idx}. {hit.get('chunk_id', '')} | "
                        f"title={hit.get('title', '')} | distance={hit.get('distance', '')}"
                    )

            if self.ask_config.enable_hybrid_retrieval:
                bm25_hits = self.vector_store.query_bm25(query, top_k=self.ask_config.retrieval_candidate_k)
                bm25_query_hits[query] = bm25_hits
                if self.ask_config.debug_mode:
                    self._log(f"[ask][debug] bm25  | query={query} 命中 {len(bm25_hits)} 个 chunk")
                    for idx, hit in enumerate(bm25_hits, start=1):
                        self._log(
                            f"[ask][debug]   - {idx}. {hit.get('chunk_id', '')} | "
                            f"title={hit.get('title', '')} | bm25={hit.get('bm25_score', '')}"
                        )

        if self.ask_config.enable_hybrid_retrieval:
            merged = self._fuse_hybrid_rrf(dense_query_hits, bm25_query_hits)
        else:
            merged = self._dedupe_chunks(dense_candidates)

        chunk_hits = self._rerank_and_truncate(state["question"], merged)
        self._log(
            f"[ask] 检索完成，候选去重/融合后 {len(merged)} 条，"
            f"最终保留 {len(chunk_hits)} 条"
        )
        debug_info = {
            **state["debug_info"],
            "search_queries": search_queries,
            "dense_query_hits": {
                query: [
                    {
                        "chunk_id": hit.get("chunk_id"),
                        "title": hit.get("title"),
                        "distance": hit.get("distance"),
                        "source_path": hit.get("source_path"),
                    }
                    for hit in hits
                ]
                for query, hits in dense_query_hits.items()
            },
            "bm25_query_hits": {
                query: [
                    {
                        "chunk_id": hit.get("chunk_id"),
                        "title": hit.get("title"),
                        "bm25_score": hit.get("bm25_score"),
                        "source_path": hit.get("source_path"),
                    }
                    for hit in hits
                ]
                for query, hits in bm25_query_hits.items()
            },
            "merged_candidate_count": len(merged),
            "final_chunk_ids": [item.get("chunk_id") for item in chunk_hits],
            "rerank_enabled": self.ask_config.enable_rerank,
            "hybrid_enabled": self.ask_config.enable_hybrid_retrieval,
        }
        return {
            **state,
            "search_queries": search_queries,
            "chunk_hits": chunk_hits,
            "dense_query_hits": dense_query_hits,
            "bm25_query_hits": bm25_query_hits,
            "debug_info": debug_info,
            "used_retrieval": True,
        }

    def _ask_extract_entities(self, state: AskState) -> AskState:
        query_entities = self.llm.extract_query_entities(state["question"])
        graph_query_entities = list(query_entities)
        synonym_map: dict[str, list[str]] = {}
        if query_entities and self.ask_config.enable_graph_entity_synonyms:
            synonym_map = self.llm.expand_graph_query_entities(
                query_entities,
                max_synonyms_per_entity=self.ask_config.graph_synonyms_per_entity,
            )
            for entity in query_entities:
                for synonym in synonym_map.get(entity, []):
                    if synonym not in graph_query_entities:
                        graph_query_entities.append(synonym)

        self._log(f"[ask] 问题实体抽取结果: {query_entities or '无'}")
        if self.ask_config.enable_graph_entity_synonyms:
            self._log(f"[ask] 图谱检索实体扩展: {graph_query_entities or '无'}")
        if self.ask_config.debug_mode:
            self._log(f"[ask][debug] 实体同义词映射: {synonym_map or {}}")
        return {
            **state,
            "query_entities": query_entities,
            "graph_query_entities": graph_query_entities,
            "debug_info": {
                **state["debug_info"],
                "query_entities": query_entities,
                "graph_query_entities": graph_query_entities,
                "entity_synonym_map": synonym_map,
            },
        }

    def _ask_route_after_entity_extract(self, state: AskState) -> str:
        if self.graph_store is not None and state["graph_query_entities"]:
            return "graph_retrieve"
        self._log("[ask] 跳过图谱检索（未启用图谱或未抽取到实体）")
        return "answer_question"

    def _ask_graph_retrieve(self, state: AskState) -> AskState:
        max_hops = self.ask_config.graph_max_hops if self.ask_config.enable_graph_multi_hop else 1
        graph_query_limit = max(self.ask_config.graph_top_k * 4, self.settings.retrieval_k * 4)
        graph_hits = self._require_graph_store().query_entity_relations(
            state["graph_query_entities"],
            limit=graph_query_limit,
            max_hops=max_hops,
        )
        processed_graph_hits, graph_post_debug = self._postprocess_graph_hits(
            query_entities=state["graph_query_entities"],
            graph_hits=graph_hits,
        )
        self._log(
            f"[ask] 图谱检索完成（max_hops={max_hops}），原始 {len(graph_hits)} 条，"
            f"后处理后 {len(processed_graph_hits)} 条"
        )
        if self.ask_config.debug_mode:
            self._log(
                "[ask][debug] 图后处理统计: "
                f"dedup={graph_post_debug.get('dedup_applied')} "
                f"({graph_post_debug.get('raw_count')} -> {graph_post_debug.get('dedup_count')}), "
                f"rerank={graph_post_debug.get('rerank_applied')}, "
                f"truncate={graph_post_debug.get('truncate_applied')} "
                f"({graph_post_debug.get('after_rerank_count')} -> {graph_post_debug.get('final_count')})"
            )
        debug_info = {
            **state["debug_info"],
            "graph_max_hops": max_hops,
            "graph_multi_hop_enabled": self.ask_config.enable_graph_multi_hop,
            "graph_query_limit": graph_query_limit,
            "graph_postprocess": graph_post_debug,
            "graph_hit_preview": [
                {
                    "query_entity": item.get("query_entity"),
                    "matched_entity": item.get("matched_entity"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "relation": item.get("relation"),
                    "path_hops": item.get("path_hops"),
                    "graph_score": item.get("graph_score"),
                }
                for item in processed_graph_hits[:20]
            ],
        }
        return {
            **state,
            "graph_hits": processed_graph_hits,
            "debug_info": debug_info,
        }

    def _ask_answer_question(self, state: AskState) -> AskState:
        if state["intent"] in {"already_retrieved", "unrelated"}:
            return state
        chunk_hits = self._compress_evidence(state["question"], state["chunk_hits"], state["debug_info"])
        answer, final_prompt = self.llm.answer_question_with_prompt(
            state["question"],
            chunk_hits,
            state["graph_hits"],
            memory_context=state["memory_context"],
        )
        debug_info = state["debug_info"]
        if self.ask_config.debug_mode:
            debug_info = {
                **debug_info,
                "final_answer_prompt": final_prompt,
            }
            self._log("[ask][debug] 最终回答提示词如下：")
            self._log(final_prompt)
        return {
            **state,
            "chunk_hits": chunk_hits,
            "debug_info": debug_info,
            "answer": answer,
            "final_prompt": final_prompt,
        }

    def _ask_answer_by_memory(self, state: AskState) -> AskState:
        memory_match, similarity = self._match_retrieved_memory(state["question"])
        if memory_match is None:
            # Safety fallback: when no reusable memory, route to retrieval branch.
            return {
                **state,
                "intent": "need_retrieve",
                "intent_reason": "memory_not_found_fallback_to_retrieve",
            }
        answer, memory_prompt = self.llm.answer_from_memory_prompt(
            question=state["question"],
            historical_prompt=memory_match.final_prompt,
            historical_answer=memory_match.answer,
            memory_context=state["memory_context"],
        )
        debug_info = {
            **state["debug_info"],
            "memory_reuse_turn_id": memory_match.turn_id,
            "memory_reuse_similarity": similarity,
        }
        if self.ask_config.debug_mode:
            debug_info["final_answer_prompt"] = memory_prompt
            self._log("[ask][debug] 命中已检索记忆，直接复用历史提示词：")
            self._log(memory_prompt)
        return {
            **state,
            "used_retrieval": False,
            "answer": answer,
            "final_prompt": memory_prompt,
            "debug_info": debug_info,
        }

    def _ask_answer_without_retrieval(self, state: AskState) -> AskState:
        prompt = (
            f"会话记忆:\n{state['memory_context'] or '无'}\n\n"
            f"用户问题：{state['question']}\n"
            "请直接回答该问题。该问题与数据库课程主题无关时，不要调用检索证据。"
        )
        answer = self.llm.chat("你是一个中文问答助手。", prompt)
        debug_info = dict(state["debug_info"])
        if self.ask_config.debug_mode:
            debug_info["final_answer_prompt"] = prompt
            self._log("[ask][debug] 无关问题，跳过检索直接回答。提示词如下：")
            self._log(prompt)
        return {
            **state,
            "used_retrieval": False,
            "answer": answer,
            "final_prompt": prompt,
            "debug_info": debug_info,
        }

    def _postprocess_graph_hits(
        self,
        query_entities: list[str],
        graph_hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw_count = len(graph_hits)
        deduped = graph_hits
        if self.ask_config.enable_graph_hit_dedup:
            deduped = self._dedupe_graph_hits(graph_hits)
        dedup_count = len(deduped)

        ranked = deduped
        if self.ask_config.enable_graph_hit_rerank:
            ranked = self._score_and_sort_graph_hits(query_entities, deduped)
        after_rerank_count = len(ranked)

        final_hits = ranked
        if self.ask_config.enable_graph_hit_truncate:
            final_hits = ranked[: self.ask_config.graph_top_k]
        final_count = len(final_hits)

        debug_info = {
            "raw_count": raw_count,
            "dedup_count": dedup_count,
            "after_rerank_count": after_rerank_count,
            "final_count": final_count,
            "dedup_applied": self.ask_config.enable_graph_hit_dedup,
            "rerank_applied": self.ask_config.enable_graph_hit_rerank,
            "truncate_applied": self.ask_config.enable_graph_hit_truncate,
            "graph_top_k": self.ask_config.graph_top_k,
            "final_hit_ids": [
                f"{item.get('source', '')}|{item.get('relation', '')}|{item.get('target', '')}"
                for item in final_hits
            ],
        }
        return final_hits, debug_info

    @staticmethod
    def _dedupe_graph_hits(graph_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for item in graph_hits:
            key = (
                str(item.get("source", "")).strip(),
                str(item.get("relation", "")).strip(),
                str(item.get("target", "")).strip(),
                str(item.get("evidence", "")).strip(),
            )
            if key not in merged:
                merged[key] = dict(item)
                continue

            existing = merged[key]
            existing_hop = int(existing.get("path_hops", 9999) or 9999)
            current_hop = int(item.get("path_hops", 9999) or 9999)
            if current_hop < existing_hop:
                replacement = dict(item)
                if existing.get("matched_entity") and not replacement.get("matched_entity"):
                    replacement["matched_entity"] = existing.get("matched_entity")
                if existing.get("query_entity") and not replacement.get("query_entity"):
                    replacement["query_entity"] = existing.get("query_entity")
                merged[key] = replacement
        return list(merged.values())

    def _score_and_sort_graph_hits(self, query_entities: list[str], graph_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_set = {item.strip() for item in query_entities if item.strip()}
        scored: list[dict[str, Any]] = []
        for item in graph_hits:
            matched_entity = str(item.get("matched_entity", "")).strip()
            relation = str(item.get("relation", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            hop = int(item.get("path_hops", 1) or 1)
            score = 0.0
            if matched_entity and matched_entity in query_set:
                score += self.ask_config.graph_score_exact_match_weight
            elif matched_entity:
                score += self.ask_config.graph_score_partial_match_weight
            score += self.ask_config.graph_score_hop_weight / max(1, hop)
            if relation:
                score += self.ask_config.graph_score_relation_weight
            if evidence:
                score += self.ask_config.graph_score_evidence_weight
            if str(item.get("source_path", "")).strip():
                score += self.ask_config.graph_score_source_weight
            merged = dict(item)
            merged["graph_score"] = round(score, 6)
            scored.append(merged)
        scored.sort(
            key=lambda x: (
                float(x.get("graph_score", 0.0)),
                -int(x.get("path_hops", 9999) or 9999),
            ),
            reverse=True,
        )
        return scored

    def _build_search_queries(self, question: str) -> list[str]:
        if not self.ask_config.enable_query_rewrite:
            return [question]

        rewrite_payload = self.llm.build_query_variants(question)
        route_1 = rewrite_payload["de_colloquialized_query"]
        route_2 = rewrite_payload["synonym_keyword_query"]
        route_3 = self.llm.generate_hypothetical_answer_query(question)

        ordered = [route_1, route_2, route_3]
        deduped: list[str] = []
        seen: set[str] = set()
        for query in ordered:
            key = query.strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped or [question]

    @staticmethod
    def _dedupe_chunks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_chunk: dict[str, dict[str, Any]] = {}
        for item in candidates:
            chunk_id = str(item.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            existing = best_by_chunk.get(chunk_id)
            if existing is None:
                merged = dict(item)
                merged["matched_queries"] = [item.get("source_query", "")]
                best_by_chunk[chunk_id] = merged
                continue

            prev_distance = float(existing.get("distance", 1e9))
            curr_distance = float(item.get("distance", 1e9))
            if curr_distance < prev_distance:
                replacement = dict(item)
                replacement["matched_queries"] = list(existing.get("matched_queries", []))
                replacement["matched_queries"].append(item.get("source_query", ""))
                best_by_chunk[chunk_id] = replacement
            else:
                existing.setdefault("matched_queries", []).append(item.get("source_query", ""))
        return list(best_by_chunk.values())

    def _fuse_hybrid_rrf(
        self,
        dense_query_hits: dict[str, list[dict[str, Any]]],
        bm25_query_hits: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        rrf_k = max(1, self.ask_config.rrf_k)

        def update_from_ranked_hits(
            query: str,
            ranked_hits: list[dict[str, Any]],
            retriever: str,
        ) -> None:
            for rank, hit in enumerate(ranked_hits, start=1):
                chunk_id = str(hit.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                rrf = 1.0 / (rrf_k + rank)
                item = merged.get(chunk_id)
                if item is None:
                    item = dict(hit)
                    item["matched_queries"] = [query]
                    item["matched_retrievers"] = [retriever]
                    item["fusion_score"] = rrf
                    merged[chunk_id] = item
                    continue
                item["fusion_score"] = float(item.get("fusion_score", 0.0)) + rrf
                item.setdefault("matched_queries", [])
                if query not in item["matched_queries"]:
                    item["matched_queries"].append(query)
                item.setdefault("matched_retrievers", [])
                if retriever not in item["matched_retrievers"]:
                    item["matched_retrievers"].append(retriever)

                prev_distance = float(item.get("distance", 1e9))
                curr_distance = float(hit.get("distance", 1e9))
                if curr_distance < prev_distance:
                    keep = dict(hit)
                    keep["fusion_score"] = item["fusion_score"]
                    keep["matched_queries"] = item["matched_queries"]
                    keep["matched_retrievers"] = item["matched_retrievers"]
                    merged[chunk_id] = keep

        for query, hits in dense_query_hits.items():
            update_from_ranked_hits(query, hits, retriever="dense")
        for query, hits in bm25_query_hits.items():
            update_from_ranked_hits(query, hits, retriever="bm25")

        fused = list(merged.values())
        fused.sort(key=lambda item: float(item.get("fusion_score", 0.0)), reverse=True)
        return fused

    def _rerank_and_truncate(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not self.ask_config.enable_rerank:
            if any("fusion_score" in item for item in candidates):
                return sorted(
                    candidates,
                    key=lambda item: float(item.get("fusion_score", 0.0)),
                    reverse=True,
                )[: self.ask_config.retrieval_k]
            return sorted(candidates, key=lambda item: float(item.get("distance", 1e9)))[: self.ask_config.retrieval_k]

        texts = [str(item.get("text", "")) for item in candidates]
        try:
            scores = self.llm.rerank(question, texts)
            rerank_method = "bge_cross_encoder"
        except Exception as exc:
            self._log(f"[ask] BGE 重排失败，回退为向量重排: {exc}")
            scores = self._embedding_rerank_scores(question, texts)
            rerank_method = "embedding_fallback"

        reranked: list[dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            merged = dict(item)
            merged["rerank_score"] = float(score)
            merged["rerank_method"] = rerank_method
            reranked.append(merged)

        reranked.sort(key=lambda item: float(item.get("rerank_score", -1e9)), reverse=True)
        return reranked[: self.ask_config.retrieval_k]

    def _embedding_rerank_scores(self, question: str, texts: list[str]) -> list[float]:
        query_embedding = self.llm.embed_query(question)
        doc_embeddings = self.llm.embed_documents(texts)
        scores: list[float] = []
        for embedding in doc_embeddings:
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            scores.append(float(score))
        return scores

    def _compress_evidence(
        self,
        question: str,
        chunk_hits: list[dict[str, Any]],
        debug_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not chunk_hits:
            return []
        if not self.ask_config.enable_evidence_compression:
            return chunk_hits

        try:
            keep_ids = self.llm.select_evidence_chunks(
                question=question,
                chunk_contexts=chunk_hits,
                max_chunks=self.ask_config.retrieval_k,
            )
        except Exception as exc:
            self._log(f"[ask] 证据压缩失败，回退到重排结果: {exc}")
            return chunk_hits
        if not keep_ids:
            return chunk_hits
        keep_set = set(keep_ids)
        compressed = [item for item in chunk_hits if str(item.get("chunk_id", "")) in keep_set]
        if compressed:
            debug_info["compressed_chunk_ids"] = [item.get("chunk_id") for item in compressed]
            self._log(f"[ask] 证据压缩完成，保留 {len(compressed)}/{len(chunk_hits)} 条片段")
            return compressed
        return chunk_hits

    def _match_retrieved_memory(self, question: str) -> tuple[MemoryTurn | None, float]:
        candidates = self.memory_store.recent_retrieved_turns(limit=self.settings.short_memory_max_turns)
        if not candidates:
            return None, 0.0
        query_embedding = self.llm.embed_query(question)
        candidate_texts = [item.question for item in candidates]
        embeddings = self.llm.embed_documents(candidate_texts)
        best_turn: MemoryTurn | None = None
        best_score = -1.0
        for turn, emb in zip(candidates, embeddings):
            score = sum(a * b for a, b in zip(query_embedding, emb))
            if score > best_score:
                best_score = float(score)
                best_turn = turn
        if best_turn is None:
            return None, 0.0
        return best_turn, best_score

    def _build_agent_memory_context(self) -> str:
        short_turns = self.memory_store.recent_short_turns_for_prompt(limit=self.settings.short_memory_prompt_turns)
        long_items = self.memory_store.get_long_term_memories()

        lines: list[str] = []
        lines.append("[长期记忆]")
        if not long_items:
            lines.append("- 无")
        else:
            for idx, item in enumerate(long_items, start=1):
                lines.append(f"- L{idx}: {item.content}")

        lines.append("")
        lines.append("[近几轮短期记忆]")
        if not short_turns:
            lines.append("- 无")
        else:
            for idx, turn in enumerate(short_turns, start=1):
                lines.append(f"- S{idx}. 用户: {turn['question']}")
                lines.append(f"      助手: {turn['answer']}")
        return "\n".join(lines)

    def _maintain_long_term_memory(self, state: AskState, turn: MemoryTurn) -> None:
        if not state.get("should_write_long_memory", False):
            return
        try:
            memory_text = self.llm.extract_long_term_memory(
                question=state["question"],
                answer=state["answer"],
            )
            if not memory_text:
                return
            self.memory_store.add_long_term_memory(memory_text, source_turn_id=turn.turn_id)
            total_chars = self.memory_store.total_long_term_chars()
            if total_chars > self.settings.long_memory_max_chars:
                self.memory_store.compress_oldest_half_long_memories()
                total_chars = self.memory_store.total_long_term_chars()
            if self.ask_config.debug_mode:
                self._log(
                    f"[ask][debug] 长期记忆写入完成: chars={total_chars}, "
                    f"reason={state.get('long_memory_write_reason', '')}"
                )
        except Exception as exc:
            self._log(f"[memory] 长期记忆维护失败: {exc}")

    def _remember_turn(self, state: AskState) -> None:
        try:
            turn = self.memory_store.add_turn(
                question=state["question"],
                answer=state["answer"],
                final_prompt=state["final_prompt"],
                intent=state["intent"],
                used_retrieval=state["used_retrieval"],
            )
            self._maintain_long_term_memory(state, turn)
        except Exception as exc:
            self._log(f"[memory] 写入记忆失败: {exc}")
