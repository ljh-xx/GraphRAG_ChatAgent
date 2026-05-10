"""
测试脚本：构建向量库 + 知识图谱库，并验证检索效果

用法：
    python test_ingest_and_search.py              # 完整构建 + 检索测试
    python test_ingest_and_search.py --vector-only  # 仅构建向量库（无需 Neo4j）
    python test_ingest_and_search.py --search-only  # 仅检索（跳过构建）

批处理参数在下方 BatchConfig 中集中管理，可直接修改。
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.stdout.reconfigure(encoding="utf-8")

from kg_rag_demo.config import Settings
from kg_rag_demo.llm import LLMClient
from kg_rag_demo.parsers import DocumentParser
from kg_rag_demo.chunking import chunk_documents
from kg_rag_demo.vector_store import VectorStore
from kg_rag_demo.graph_store import GraphStore
from concurrent.futures import ThreadPoolExecutor, as_completed
from kg_rag_demo.models import RelationEdge
from kg_rag_demo.pipeline import AskRuntimeConfig


# ============================================================================
# 批处理参数 —— 集中修改这里
# ============================================================================
class BatchConfig:
    """批处理相关参数，可根据数据规模和机器配置调整。"""

    # ---------- Chunk 相关 ----------
    chunk_size = 1200       # chunk 最大 token 数（tiktoken cl100k_base 计量）
    chunk_overlap = 180     # chunk 间重叠 token 数（chunk_size 的 15%）

    # ---------- 图谱抽取并发 ----------
    graph_workers = 16       # 并发抽取关系的线程数
                            #   本地 API → 4-8 可充分利用网络 IO
                            #   本地模型 → 1-2，避免多份模型同时加载到显存

    # ---------- 向量写入 ----------
    vector_batch_size = 500  # 每批写入 Chroma 的 chunk 数
                             #   减小 → 内存占用低、进度反馈频繁
                             #   增大 → 写入快但内存峰值高

    # ---------- 检索 ----------
    retrieval_candidate_k = 12   # 每条查询的候选数
    retrieval_k = 5              # 最终返回的 chunk 数
    test_questions = [
        "BERT 模型的核心创新是什么？",
        "什么是知识图谱？",
        "人工智能发展的战略地位体现在哪些方面？",
    ]


# ============================================================================
# 工具函数
# ============================================================================

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{seconds % 60:.0f}s"


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================================
# 步骤 1：解析 + 切块
# ============================================================================

def step_parse_and_chunk(settings: Settings) -> tuple:
    """解析 data/ 下所有文件并切块。"""
    print_separator("步骤 1/4：文档解析 + 切块")

    parser = DocumentParser(progress_callback=lambda msg: print(f"  [parser] {msg}"))
    t0 = time.perf_counter()

    docs = parser.parse_directory("data")
    t1 = time.perf_counter()
    print(f"\n  解析完成：{len(docs)} 条文档记录 ({format_duration(t1 - t0)})")

    total_chars = sum(len(d.text) for d in docs)
    print(f"  总字符数：{total_chars:,}")

    chunks = chunk_documents(docs, settings)
    t2 = time.perf_counter()
    print(f"  切块完成：{len(chunks)} 个 chunk ({format_duration(t2 - t1)})")

    # 按 modality 统计
    from collections import Counter
    modality_counts = Counter(c.modality for c in chunks)
    for modality, count in modality_counts.items():
        avg_len = sum(len(c.text) for c in chunks if c.modality == modality) / max(count, 1)
        print(f"    {modality}: {count} 个, 平均 {avg_len:.0f} 字符/chunk")

    return docs, chunks


# ============================================================================
# 步骤 2：构建向量库
# ============================================================================

def step_build_vector_store(settings: Settings, llm: LLMClient, chunks: list) -> VectorStore:
    """构建 Chroma 向量库，分批写入。"""
    print_separator("步骤 2/4：构建向量库 (ChromaDB)")

    batch_size = BatchConfig.vector_batch_size
    vs = VectorStore(settings, llm, reset=True)
    total = len(chunks)
    t0 = time.perf_counter()

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vs.upsert_chunks(batch)
        progress = min(i + batch_size, total)
        elapsed = time.perf_counter() - t0
        speed = progress / max(elapsed, 0.1)
        print(
            f"  [{progress}/{total}] "
            f"已写入 {progress} 个 chunk | "
            f"速度 {speed:.0f} chunk/s | "
            f"耗时 {format_duration(elapsed)}"
        )

    t1 = time.perf_counter()
    print(f"\n  向量库构建完成 ({format_duration(t1 - t0)})")

    # 自检
    vs.close()
    vs._open_client()
    collection = vs._require_collection()
    stored = collection.count()
    print(f"  向量库中已存储 {stored} 个 chunk")
    return vs


# ============================================================================
# 步骤 3：构建知识图谱
# ============================================================================

def step_build_graph(settings: Settings, llm: LLMClient, chunks: list) -> GraphStore:
    """构建 Neo4j 知识图谱（并发抽取关系）。"""
    print_separator("步骤 3/4：构建知识图谱 (Neo4j)")

    gs = GraphStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    gs.ensure_constraints()
    print("  Neo4j 约束已就绪")

    workers = max(1, BatchConfig.graph_workers)
    total = len(chunks)
    print(f"  并发 worker 数: {workers}，共 {total} 个 chunk")

    # 先将所有 chunk 写入 Neo4j（串行，保证节点存在）
    for idx, chunk in enumerate(chunks, start=1):
        gs.upsert_chunk(chunk)

    # 并发抽取关系
    relation_count = 0
    completed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graph") as executor:
        future_map = {
            executor.submit(llm.extract_graph, chunk.text): (idx, chunk)
            for idx, chunk in enumerate(chunks, start=1)
        }

        for future in as_completed(future_map):
            idx, chunk = future_map[future]
            try:
                _, relations = future.result()
            except Exception as exc:
                print(f"  [graph] 抽取失败 {idx}/{total}: {chunk.chunk_id} | {exc}")
            else:
                if relations:
                    gs.upsert_relations(chunk, relations)
                    relation_count += len(relations)

            completed += 1
            if completed % 5 == 0 or completed == total:
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{completed}/{total}] "
                    f"已写入 {relation_count} 条关系 | "
                    f"耗时 {format_duration(elapsed)}"
                )

    t1 = time.perf_counter()
    print(f"\n  知识图谱构建完成 ({format_duration(t1 - t0)})")
    print(f"  共写入 {relation_count} 条关系")
    return gs


# ============================================================================
# 步骤 4：检索测试
# ============================================================================

def step_search_test(
    settings: Settings,
    vs: VectorStore,
    gs: GraphStore | None,
    chunks: list,
) -> None:
    """对测试问题执行检索，展示效果。"""
    print_separator("步骤 4/4：检索测试")

    cfg = AskRuntimeConfig(
        retrieval_k=BatchConfig.retrieval_k,
        retrieval_candidate_k=BatchConfig.retrieval_candidate_k,
    )
    llm = LLMClient(settings)

    for q_idx, question in enumerate(BatchConfig.test_questions, start=1):
        print(f"\n{'─' * 60}")
        print(f"  问题 {q_idx}: {question}")
        print(f"{'─' * 60}")

        # ---- BM25 检索 ----
        t0 = time.perf_counter()
        bm25_hits = vs.query_bm25(question, top_k=cfg.retrieval_candidate_k)
        bm25_time = time.perf_counter() - t0
        print(f"\n  [BM25] 命中 {len(bm25_hits)} 条 ({bm25_time:.2f}s)")
        for i, hit in enumerate(bm25_hits[:3], start=1):
            snippet = hit["text"][:100].replace("\n", " ")
            print(f"    {i}. score={hit.get('bm25_score', '?'):.4f} | {snippet}...")

        # ---- Dense 检索 ----
        t0 = time.perf_counter()
        dense_hits = vs.query(question, top_k=cfg.retrieval_candidate_k)
        dense_time = time.perf_counter() - t0
        print(f"\n  [Dense] 命中 {len(dense_hits)} 条 ({dense_time:.2f}s)")
        for i, hit in enumerate(dense_hits[:3], start=1):
            snippet = hit["text"][:100].replace("\n", " ")
            print(f"    {i}. dist={hit.get('distance', '?'):.4f} | {snippet}...")

        # ---- 图检索 ----
        if gs is not None:
            entities = llm.extract_query_entities(question)
            print(f"\n  [Graph] 抽取实体: {entities}")
            if entities:
                t0 = time.perf_counter()
                graph_hits = gs.query_entity_relations(entities, limit=cfg.graph_top_k * 4)
                graph_time = time.perf_counter() - t0
                print(f"  [Graph] 命中 {len(graph_hits)} 条关系 ({graph_time:.2f}s)")
                for i, hit in enumerate(graph_hits[:5], start=1):
                    print(
                        f"    {i}. {hit.get('source', '?')} "
                        f"--{hit.get('relation', '?')}--> "
                        f"{hit.get('target', '?')} "
                        f"(hops={hit.get('path_hops', '?')})"
                    )
            else:
                print(f"  [Graph] 未抽到实体，跳过图检索")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="构建并测试向量库和知识图谱库")
    parser.add_argument("--vector-only", action="store_true", help="仅构建向量库")
    parser.add_argument("--search-only", action="store_true", help="仅检索测试（跳过构建）")
    parser.add_argument("--no-graph", action="store_true", help="跳过知识图谱")
    args = parser.parse_args()

    settings = Settings()
    settings.chunk_size = BatchConfig.chunk_size
    settings.chunk_overlap = BatchConfig.chunk_overlap
    settings.graph_extract_workers = BatchConfig.graph_workers
    settings.retrieval_k = BatchConfig.retrieval_k
    settings.retrieval_candidate_k = BatchConfig.retrieval_candidate_k

    print("=" * 60)
    print("  向量库 & 知识图谱库 构建与测试")
    print("=" * 60)
    print(f"  Chunk 大小: {settings.chunk_size} tokens")
    print(f"  Chunk 重叠: {settings.chunk_overlap} tokens")
    print(f"  图谱并发:   {settings.graph_extract_workers} workers")
    print(f"  向量批量:   {BatchConfig.vector_batch_size} chunks/batch")
    print(f"  检索候选:   {settings.retrieval_candidate_k} → 最终 {settings.retrieval_k}")

    llm = LLMClient(settings)

    if args.search_only:
        # 跳过构建，仅打开已有库做检索
        print("\n  [模式] 仅检索，跳过构建")
        vs = VectorStore(settings, llm, reset=False)
        gs = None
        if not args.no_graph:
            try:
                gs = GraphStore(
                    uri=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
                print("  Neo4j 连接成功")
            except Exception as exc:
                print(f"  Neo4j 连接失败（将跳过图检索）: {exc}")
        step_search_test(settings, vs, gs, [])
        vs.close()
        if gs:
            gs.close()
        return

    # --- 构建流程 ---
    overall_t0 = time.perf_counter()

    # 1. 解析 + 切块
    docs, chunks = step_parse_and_chunk(settings)

    # 2. 向量库
    vs = step_build_vector_store(settings, llm, chunks)

    # 3. 知识图谱（可选）
    gs = None
    if not args.no_graph:
        print(f"\n  图谱并发 worker 数: {settings.graph_extract_workers}")
        print(f"  （本地模型建议 1-2，API 建议 4-8）")
        try:
            gs = step_build_graph(settings, llm, chunks)
        except Exception as exc:
            print(f"\n  ⚠ 知识图谱构建失败（请确认 Neo4j 已启动）: {exc}")
            print("  将继续执行向量库检索测试")

    overall_elapsed = time.perf_counter() - overall_t0
    print(f"\n  总构建耗时: {format_duration(overall_elapsed)}")

    # 4. 检索测试
    step_search_test(settings, vs, gs, chunks)

    # 清理
    vs.close()
    if gs:
        gs.close()

    print(f"\n{'=' * 60}")
    print(f"  测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
