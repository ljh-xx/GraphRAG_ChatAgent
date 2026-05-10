from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kg_rag_demo.pipeline import AskRuntimeConfig, DemoPipeline


def infer_ask_stage(log_message: str) -> str:
    text = (log_message or "").strip()
    if "长期记忆写入意图" in text:
        return "正在判断长期记忆写入..."
    if "意图识别" in text:
        return "正在意图识别..."
    if "dense" in text or "bm25" in text or "检索完成" in text or "query_" in text:
        return "正在执行文本检索..."
    if "图谱检索" in text or "实体抽取" in text:
        return "正在执行图谱检索..."
    if "证据压缩" in text or "最终回答提示词" in text:
        return "正在生成回答..."
    if "收到问题" in text:
        return "正在准备问题..."
    return "正在处理中..."


def build_langgraph_workflow_dot() -> str:
    return """
    digraph Workflow {
      rankdir=LR;
      graph [fontname="Microsoft YaHei", bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#EEF4FF", color="#6C8EF5", fontname="Microsoft YaHei"];
      edge [fontname="Microsoft YaHei", color="#8AA0D6"];

      start [shape=circle, fillcolor="#D9E8FF", label="开始"];
      prepare [label="准备问题"];
      classify_ltm [label="长期记忆写入判定"];
      classify_intent [label="检索意图识别"];
      retrieve [label="文本检索"];
      extract_entities [label="实体抽取"];
      graph_retrieve [label="图谱检索"];
      answer_memory [label="记忆复用回答"];
      answer_no_retrieve [label="跳过检索直答"];
      answer [label="生成最终回答"];
      end [shape=doublecircle, fillcolor="#D9E8FF", label="结束"];

      start -> prepare -> classify_ltm -> classify_intent;
      classify_intent -> retrieve [label="需检索"];
      classify_intent -> answer_memory [label="已检索"];
      classify_intent -> answer_no_retrieve [label="无关"];
      retrieve -> extract_entities;
      extract_entities -> graph_retrieve [label="有实体"];
      extract_entities -> answer [label="无实体"];
      graph_retrieve -> answer;
      answer_memory -> answer;
      answer_no_retrieve -> answer;
      answer -> end;
    }
    """


def resolve_source_file_path(raw_path: str) -> Path | None:
    text = (raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    rooted = PROJECT_ROOT / text
    if rooted.exists():
        return rooted
    return None


def open_local_file(path: Path) -> tuple[bool, str]:
    try:
        resolved = path.resolve()
        if os.name == "nt":
            os.startfile(str(resolved))
            return True, ""
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved)])
            return True, ""
        subprocess.Popen(["xdg-open", str(resolved)])
        return True, ""
    except Exception as exc:
        return False, str(exc)


def to_file_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except Exception:
        return ""


def build_graphviz(graph_hits: list[dict[str, str]]) -> str:
    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [fontname="Microsoft YaHei"];',
        'node [shape=ellipse, style=filled, fillcolor="#E8F0FE", fontname="Microsoft YaHei"];',
        'edge [fontname="Microsoft YaHei"];',
    ]
    highlighted: set[str] = set()
    for item in graph_hits:
        if item.get("matched_entity"):
            highlighted.add(item["matched_entity"])

    nodes = {item["source"] for item in graph_hits} | {item["target"] for item in graph_hits}
    for name in nodes:
        fill = "#FFE082" if name in highlighted else "#E8F0FE"
        safe_name = name.replace('"', '\\"')
        lines.append(f'"{safe_name}" [fillcolor="{fill}"];')

    for item in graph_hits:
        source = item["source"].replace('"', '\\"')
        target = item["target"].replace('"', '\\"')
        relation = item["relation"].replace('"', '\\"')
        lines.append(f'"{source}" -> "{target}" [label="{relation}"];')

    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="KG + RAG", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.2rem; }
    .subtitle { color: #5B6B8A; margin-bottom: 0.8rem; }
    .answer-card {
      border: 1px solid #D5E2FF;
      background: linear-gradient(180deg, #F8FBFF 0%, #F2F7FF 100%);
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 12px;
      box-shadow: 0 4px 14px rgba(67, 102, 193, 0.08);
    }
    .answer-card h4 { margin: 0 0 8px 0; color: #2A3B66; }
    .route-card {
      border: 1px solid #E5EAF5;
      border-radius: 12px;
      padding: 10px 12px;
      background: #FCFDFF;
      margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("知识图谱 + RAG")
st.caption("面向中文 PDF / Word / 图片")


@st.cache_resource
def get_pipeline(
    enable_query_rewrite: bool,
    enable_hybrid_retrieval: bool,
    enable_rerank: bool,
    enable_evidence_compression: bool,
    enable_graph_entity_synonyms: bool,
    enable_graph_multi_hop: bool,
    enable_graph_hit_dedup: bool,
    enable_graph_hit_rerank: bool,
    enable_graph_hit_truncate: bool,
    debug_mode: bool,
    retrieval_k: int,
    retrieval_candidate_k: int,
    rrf_k: int,
    graph_max_hops: int,
    graph_synonyms_per_entity: int,
    graph_top_k: int,
    graph_score_exact_match_weight: float,
    graph_score_partial_match_weight: float,
    graph_score_hop_weight: float,
    graph_score_relation_weight: float,
    graph_score_evidence_weight: float,
    graph_score_source_weight: float,
) -> DemoPipeline:
    ask_config = AskRuntimeConfig(
        enable_query_rewrite=enable_query_rewrite,
        enable_hybrid_retrieval=enable_hybrid_retrieval,
        enable_rerank=enable_rerank,
        enable_evidence_compression=enable_evidence_compression,
        enable_graph_entity_synonyms=enable_graph_entity_synonyms,
        enable_graph_multi_hop=enable_graph_multi_hop,
        enable_graph_hit_dedup=enable_graph_hit_dedup,
        enable_graph_hit_rerank=enable_graph_hit_rerank,
        enable_graph_hit_truncate=enable_graph_hit_truncate,
        debug_mode=debug_mode,
        retrieval_k=max(1, retrieval_k),
        retrieval_candidate_k=max(retrieval_k, retrieval_candidate_k),
        rrf_k=max(1, rrf_k),
        graph_max_hops=max(1, graph_max_hops),
        graph_synonyms_per_entity=max(0, graph_synonyms_per_entity),
        graph_top_k=max(1, graph_top_k),
        graph_score_exact_match_weight=float(graph_score_exact_match_weight),
        graph_score_partial_match_weight=float(graph_score_partial_match_weight),
        graph_score_hop_weight=float(graph_score_hop_weight),
        graph_score_relation_weight=float(graph_score_relation_weight),
        graph_score_evidence_weight=float(graph_score_evidence_weight),
        graph_score_source_weight=float(graph_score_source_weight),
    )
    return DemoPipeline(ask_config=ask_config)


if "ui_enable_query_rewrite" not in st.session_state:
    st.session_state.ui_enable_query_rewrite = True
if "ui_enable_rerank" not in st.session_state:
    st.session_state.ui_enable_rerank = True
if "ui_enable_hybrid_retrieval" not in st.session_state:
    st.session_state.ui_enable_hybrid_retrieval = True
if "ui_enable_evidence_compression" not in st.session_state:
    st.session_state.ui_enable_evidence_compression = True
if "ui_enable_graph_entity_synonyms" not in st.session_state:
    st.session_state.ui_enable_graph_entity_synonyms = True
if "ui_enable_graph_multi_hop" not in st.session_state:
    st.session_state.ui_enable_graph_multi_hop = True
if "ui_enable_graph_hit_dedup" not in st.session_state:
    st.session_state.ui_enable_graph_hit_dedup = True
if "ui_enable_graph_hit_rerank" not in st.session_state:
    st.session_state.ui_enable_graph_hit_rerank = True
if "ui_enable_graph_hit_truncate" not in st.session_state:
    st.session_state.ui_enable_graph_hit_truncate = True
if "ui_debug_mode" not in st.session_state:
    st.session_state.ui_debug_mode = False
if "ui_retrieval_k" not in st.session_state:
    st.session_state.ui_retrieval_k = 5
if "ui_candidate_k" not in st.session_state:
    st.session_state.ui_candidate_k = 12
if "ui_graph_max_hops" not in st.session_state:
    st.session_state.ui_graph_max_hops = 2
if "ui_graph_synonyms_per_entity" not in st.session_state:
    st.session_state.ui_graph_synonyms_per_entity = 2
if "ui_graph_top_k" not in st.session_state:
    st.session_state.ui_graph_top_k = 8
if "ui_rrf_k" not in st.session_state:
    st.session_state.ui_rrf_k = 60
if "ui_memory_reuse_similarity_threshold" not in st.session_state:
    st.session_state.ui_memory_reuse_similarity_threshold = 0.84
if "ui_memory_recent_turns_for_intent" not in st.session_state:
    st.session_state.ui_memory_recent_turns_for_intent = 4
if "ui_short_memory_prompt_turns" not in st.session_state:
    st.session_state.ui_short_memory_prompt_turns = 3
if "ui_long_memory_max_chars" not in st.session_state:
    st.session_state.ui_long_memory_max_chars = 500
if "ui_graph_score_exact_match_weight" not in st.session_state:
    st.session_state.ui_graph_score_exact_match_weight = 3.0
if "ui_graph_score_partial_match_weight" not in st.session_state:
    st.session_state.ui_graph_score_partial_match_weight = 1.5
if "ui_graph_score_hop_weight" not in st.session_state:
    st.session_state.ui_graph_score_hop_weight = 1.0
if "ui_graph_score_relation_weight" not in st.session_state:
    st.session_state.ui_graph_score_relation_weight = 0.4
if "ui_graph_score_evidence_weight" not in st.session_state:
    st.session_state.ui_graph_score_evidence_weight = 0.3
if "ui_graph_score_source_weight" not in st.session_state:
    st.session_state.ui_graph_score_source_weight = 0.2


pipeline = get_pipeline(
    enable_query_rewrite=st.session_state.ui_enable_query_rewrite,
    enable_hybrid_retrieval=st.session_state.ui_enable_hybrid_retrieval,
    enable_rerank=st.session_state.ui_enable_rerank,
    enable_evidence_compression=st.session_state.ui_enable_evidence_compression,
    enable_graph_entity_synonyms=st.session_state.ui_enable_graph_entity_synonyms,
    enable_graph_multi_hop=st.session_state.ui_enable_graph_multi_hop,
    enable_graph_hit_dedup=st.session_state.ui_enable_graph_hit_dedup,
    enable_graph_hit_rerank=st.session_state.ui_enable_graph_hit_rerank,
    enable_graph_hit_truncate=st.session_state.ui_enable_graph_hit_truncate,
    debug_mode=st.session_state.ui_debug_mode,
    retrieval_k=int(st.session_state.ui_retrieval_k),
    retrieval_candidate_k=int(st.session_state.ui_candidate_k),
    rrf_k=int(st.session_state.ui_rrf_k),
    graph_max_hops=int(st.session_state.ui_graph_max_hops),
    graph_synonyms_per_entity=int(st.session_state.ui_graph_synonyms_per_entity),
    graph_top_k=int(st.session_state.ui_graph_top_k),
    graph_score_exact_match_weight=float(st.session_state.ui_graph_score_exact_match_weight),
    graph_score_partial_match_weight=float(st.session_state.ui_graph_score_partial_match_weight),
    graph_score_hop_weight=float(st.session_state.ui_graph_score_hop_weight),
    graph_score_relation_weight=float(st.session_state.ui_graph_score_relation_weight),
    graph_score_evidence_weight=float(st.session_state.ui_graph_score_evidence_weight),
    graph_score_source_weight=float(st.session_state.ui_graph_score_source_weight),
)

# Runtime knobs that currently live in Settings.
pipeline.settings.memory_reuse_similarity_threshold = float(st.session_state.ui_memory_reuse_similarity_threshold)
pipeline.settings.memory_recent_turns_for_intent = int(st.session_state.ui_memory_recent_turns_for_intent)
pipeline.settings.short_memory_prompt_turns = int(st.session_state.ui_short_memory_prompt_turns)
pipeline.settings.long_memory_max_chars = int(st.session_state.ui_long_memory_max_chars)

with st.sidebar:
    st.subheader("操作")
    if st.button("构建索引", use_container_width=True):
        with st.spinner("正在解析文档并写入索引..."):
            result = pipeline.ingest()
        skipped = result.get("skipped_chunks", 0)
        skip_msg = f"，跳过 {skipped} 个已处理" if skipped > 0 else ""
        st.success(
            f"完成: 文档 {result['documents']}，"
            f"向量写入 {result['vector_chunks']}，"
            f"图谱处理 {result['new_graph_chunks']}/{result['graph_chunks']}"
            f"{skip_msg}，"
            f"关系 {result['relations']}"
        )

    st.markdown(
        """
        说明:
        - 把资料放入 `data/`
        - 先点“构建索引”
        - 再在右侧输入问题
        """
    )

ask_tab, settings_tab = st.tabs(["问答", "设置"])

with settings_tab:
    st.subheader("设置面板")
    st.caption("按模块分组配置，修改后下一次问答生效。")

    with st.expander("文本检索 (RAG)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.checkbox("启用 query 三路改写", key="ui_enable_query_rewrite")
            st.checkbox("启用 Hybrid Retrieval (Dense + BM25)", key="ui_enable_hybrid_retrieval")
            st.checkbox("启用召回结果重排", key="ui_enable_rerank")
            st.checkbox("启用证据压缩后再生成", key="ui_enable_evidence_compression")
        with c2:
            st.number_input("最终输入模型的 chunk 数 (k)", min_value=1, max_value=30, step=1, key="ui_retrieval_k")
            st.number_input(
                "每路 query 召回候选数 (candidate_k)",
                min_value=1,
                max_value=100,
                step=1,
                key="ui_candidate_k",
            )
            st.number_input("RRF 融合常数 (rrf_k)", min_value=1, max_value=300, step=1, key="ui_rrf_k")

    with st.expander("图谱检索 (KG)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.checkbox("启用图谱实体同义词扩展", key="ui_enable_graph_entity_synonyms")
            st.checkbox("启用图谱多跳检索", key="ui_enable_graph_multi_hop")
            st.checkbox("启用图谱结果去重", key="ui_enable_graph_hit_dedup")
            st.checkbox("启用图谱结果打分排序", key="ui_enable_graph_hit_rerank")
            st.checkbox("启用图谱结果截断", key="ui_enable_graph_hit_truncate")
            st.number_input("图谱检索最大跳数", min_value=1, max_value=4, step=1, key="ui_graph_max_hops")
            st.number_input("每个实体扩展同义词数", min_value=0, max_value=8, step=1, key="ui_graph_synonyms_per_entity")
            st.number_input("图谱结果保留条数 (graph_top_k)", min_value=1, max_value=30, step=1, key="ui_graph_top_k")
        with c2:
            st.caption("KG 打分权重")
            st.number_input("精确实体匹配权重", min_value=0.0, max_value=10.0, step=0.1, key="ui_graph_score_exact_match_weight")
            st.number_input("部分匹配权重", min_value=0.0, max_value=10.0, step=0.1, key="ui_graph_score_partial_match_weight")
            st.number_input("跳数权重", min_value=0.0, max_value=5.0, step=0.1, key="ui_graph_score_hop_weight")
            st.number_input("关系字段权重", min_value=0.0, max_value=5.0, step=0.1, key="ui_graph_score_relation_weight")
            st.number_input("证据字段权重", min_value=0.0, max_value=5.0, step=0.1, key="ui_graph_score_evidence_weight")
            st.number_input("来源字段权重", min_value=0.0, max_value=5.0, step=0.1, key="ui_graph_score_source_weight")

    with st.expander("记忆与路由", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "记忆复用相似度阈值",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="ui_memory_reuse_similarity_threshold",
            )
            st.number_input("意图识别参考短期轮数", min_value=1, max_value=10, step=1, key="ui_memory_recent_turns_for_intent")
        with c2:
            st.number_input("注入提示词的短期轮数", min_value=1, max_value=10, step=1, key="ui_short_memory_prompt_turns")
            st.number_input("长期记忆总字数上限", min_value=100, max_value=5000, step=50, key="ui_long_memory_max_chars")

    with st.expander("记忆管理", expanded=False):
        col_reset_short, col_reset_long = st.columns(2)
        with col_reset_short:
            if st.button("重置短期记忆", use_container_width=True):
                pipeline.memory_store.reset_short_term_memory()
                st.success("短期记忆已清空。")
        with col_reset_long:
            if st.button("重置长期记忆", use_container_width=True):
                pipeline.memory_store.reset_long_term_memory()
                st.success("长期记忆已清空。")

    with st.expander("调试与可观测性", expanded=False):
        st.checkbox("启用 debug 模式", key="ui_debug_mode")

with ask_tab:
    question = st.text_area("请输入你的问题", placeholder="例如：什么是事务？数据库范式有哪些？", height=100)

    if st.button("开始问答", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            ask_status = st.empty()
            ask_status.info("正在准备问题...")
            ask_logs: list[str] = []

            def ui_progress_logger(message: str) -> None:
                ask_logs.append(message)
                ask_status.info(infer_ask_stage(message))

            old_logger = pipeline.progress_callback
            pipeline.progress_callback = ui_progress_logger
            try:
                result = pipeline.ask(question.strip())
            finally:
                pipeline.progress_callback = old_logger
            ask_status.success("处理完成")

            left, right = st.columns([1.2, 1])

            with left:
                intent_zh = result.debug_info.get("intent_result_zh", "-")
                intent_raw = result.debug_info.get("intent_result", "-")
                intent_reason = result.debug_info.get("intent_reason", "-")
                st.info(f"意图识别结果: {intent_zh} ({intent_raw}) | 原因: {intent_reason}")

                with st.container(border=True):
                    st.markdown("#### 回答")
                    st.write(result.answer)

                st.subheader("路由轨迹")
                st.markdown('<div class="route-card">当前对话路由状态</div>', unsafe_allow_html=True)
                route = result.debug_info.get("route_summary", {})
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("意图", route.get("intent", intent_raw))
                col_b.metric("是否执行检索", "是" if route.get("used_retrieval") else "否")
                col_c.metric("长期记忆写入", "是" if route.get("long_memory_should_write") else "否")
                st.caption(
                    f"意图原因: {route.get('intent_reason', intent_reason)} | "
                    f"长期记忆原因: {route.get('long_memory_write_reason', '-')}"
                )
                if ask_logs:
                    with st.expander("执行轨迹"):
                        for line in ask_logs:
                            st.text(line)

                st.subheader("检索片段")
                for idx, item in enumerate(result.chunk_hits, start=1):
                    with st.expander(f"片段 {idx} | {item.get('title', 'unknown')}"):
                        st.write(item.get("text", ""))
                        st.caption(
                            f"来源: {item.get('source_path', '')} | 页码: {item.get('page_number', '')} | "
                            f"距离: {item.get('distance', '')} | 重排分: {item.get('rerank_score', '')}"
                        )
                        local_path = resolve_source_file_path(str(item.get("source_path", "")))
                        if local_path is not None:
                            open_col, path_col = st.columns([1, 2])
                            with open_col:
                                open_key = f"open_local_{idx}_{item.get('chunk_id', idx)}"
                                if st.button("打开本地文件", key=open_key, use_container_width=True):
                                    ok, err = open_local_file(local_path)
                                    if ok:
                                        st.success("已触发系统打开文件。")
                                    else:
                                        st.error(f"打开失败: {err}")
                            with path_col:
                                st.caption(f"本地路径: `{local_path}`")
                                file_uri = to_file_uri(local_path)
                                if file_uri:
                                    st.caption(f"备用链接: {file_uri}")

                if st.session_state.ui_debug_mode:
                    st.subheader("Debug: Query 改写与命中明细")
                    search_queries = result.debug_info.get("search_queries", [])
                    for idx, query in enumerate(search_queries, start=1):
                        st.markdown(f"- query_{idx}: `{query}`")
                    dense_query_hits = result.debug_info.get("dense_query_hits", {})
                    bm25_query_hits = result.debug_info.get("bm25_query_hits", {})
                    for query, hits in dense_query_hits.items():
                        with st.expander(f"Dense 命中 | {query}"):
                            if not hits:
                                st.write("无命中")
                            for hit in hits:
                                st.write(
                                    f"{hit.get('chunk_id', '')} | {hit.get('title', '')} | "
                                    f"distance={hit.get('distance', '')} | source={hit.get('source_path', '')}"
                                )
                    for query, hits in bm25_query_hits.items():
                        with st.expander(f"BM25 命中 | {query}"):
                            if not hits:
                                st.write("无命中")
                            for hit in hits:
                                st.write(
                                    f"{hit.get('chunk_id', '')} | {hit.get('title', '')} | "
                                    f"bm25={hit.get('bm25_score', '')} | source={hit.get('source_path', '')}"
                                )
                    st.caption(
                        f"融合后候选数: {result.debug_info.get('merged_candidate_count', '-')}, "
                        f"最终 chunk_ids: {result.debug_info.get('final_chunk_ids', [])}, "
                        f"压缩后 chunk_ids: {result.debug_info.get('compressed_chunk_ids', [])}"
                    )
                    st.caption(
                        f"意图识别: {result.debug_info.get('intent_result_zh', '-')}"
                        f" ({result.debug_info.get('intent_result', '-')}) | "
                        f"原因: {result.debug_info.get('intent_reason', '-')}"
                    )
                    st.caption(
                        f"长期记忆写入判定: {result.debug_info.get('long_memory_should_write', '-')}"
                        f" | 原因: {result.debug_info.get('long_memory_write_reason', '-')}"
                    )
                    st.caption(
                        f"图谱实体: {result.debug_info.get('query_entities', [])} -> "
                        f"扩展后: {result.debug_info.get('graph_query_entities', [])} | "
                        f"max_hops: {result.debug_info.get('graph_max_hops', '-')}"
                    )
                    synonym_map = result.debug_info.get("entity_synonym_map", {})
                    if synonym_map:
                        st.json({"entity_synonym_map": synonym_map})
                    graph_post = result.debug_info.get("graph_postprocess", {})
                    if graph_post:
                        st.json({"graph_postprocess": graph_post})
                    final_prompt = result.debug_info.get("final_answer_prompt", "")
                    if final_prompt:
                        with st.expander("最终回答提示词 (Debug)"):
                            st.text(final_prompt)

            with right:
                st.subheader("LangGraph 流程")
                st.graphviz_chart(build_langgraph_workflow_dot(), use_container_width=True)

                st.subheader("图谱关系")
                if result.query_entities:
                    st.caption("问题实体: " + "，".join(result.query_entities))
                if not result.graph_hits:
                    st.info("未检索到明显关系。")
                else:
                    st.graphviz_chart(build_graphviz(result.graph_hits), use_container_width=True)
                    for item in result.graph_hits:
                        st.write(
                            f"{item['source']} --{item['relation']}--> {item['target']} "
                            f"(命中实体: {item.get('matched_entity', '')})"
                        )
                        if item.get("evidence"):
                            st.caption(item["evidence"])
