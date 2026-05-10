from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# utf-8-sig: avoid a BOM on the first line breaking OPENAI_API_KEY parsing on Windows editors.
# override=True: if the shell or OS already has OPENAI_API_KEY, still apply values from .env
# (common when another machine has a stale or wrong key in user/system environment).
load_dotenv(encoding="utf-8-sig", override=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
HF_HUB_CACHE_DIR = MODEL_DIR / "hub"

# Default HF cache to project-local D: path to avoid filling C: user cache.
os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE_DIR))


def resolve_local_hf_model(spec: str) -> str:
    """Resolve a Hugging Face model id to local snapshot path when available."""
    raw = (spec or "").strip()
    if not raw:
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.is_dir():
        return str(candidate.resolve())

    rooted = PROJECT_ROOT / raw
    if rooted.is_dir():
        return str(rooted.resolve())

    if "/" not in raw or "\\" in raw or raw.startswith("."):
        return raw

    parts = raw.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return raw

    cache_root = PROJECT_ROOT / "model" / f"models--{parts[0]}--{parts[1]}"
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.is_dir():
        return raw

    ref_main = cache_root / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        snap = snapshots_dir / commit
        if snap.is_dir():
            return str(snap.resolve())

    for child in sorted(snapshots_dir.iterdir()):
        if child.is_dir():
            return str(child.resolve())

    return raw


def resolve_local_embed_model(spec: str) -> str:
    """Backward-compatible wrapper for embedding model resolution."""
    return resolve_local_hf_model(spec)
DATA_DIR = PROJECT_ROOT / "data"
WORK_DIR = PROJECT_ROOT / "workdir"
CHROMA_DIR = WORK_DIR / "chroma"
CACHE_DIR = WORK_DIR / "cache"
MEMORY_DIR = WORK_DIR / "memory"
SHORT_MEMORY_PATH = MEMORY_DIR / "short_term_memory.jsonl"
LONG_MEMORY_PATH = MEMORY_DIR / "long_term_memory.jsonl"


def _parse_api_keys() -> list[str]:
    keys_env = os.getenv("OPENAI_API_KEYS", "")
    if keys_env:
        return [k.strip() for k in keys_env.split(",") if k.strip()]
    single_key = os.getenv("OPENAI_API_KEY", "").strip()
    if single_key:
        return [single_key]
    return []


def _parse_chat_models() -> list[str]:
    models_env = os.getenv("OPENAI_CHAT_MODELS", "")
    if models_env:
        return [m.strip() for m in models_env.split(",") if m.strip()]
    single_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    return [single_model]


@dataclass
class Settings:
    openai_api_keys: list[str] = field(default_factory=_parse_api_keys)
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_models: list[str] = field(default_factory=_parse_chat_models)
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    openai_max_retries: int = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
    openai_retry_backoff_seconds: float = float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    local_embed_model: str = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    vector_batch_size: int = max(50, int(os.getenv("VECTOR_BATCH_SIZE", "500")))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "kg_rag_demo")
    graph_extract_workers: int = max(1, int(os.getenv("GRAPH_EXTRACT_WORKERS", "4")))
    graph_progress_heartbeat_seconds: float = float(os.getenv("GRAPH_PROGRESS_HEARTBEAT_SECONDS", "5"))
    enable_graph_entity_synonyms: bool = os.getenv("ENABLE_GRAPH_ENTITY_SYNONYMS", "1").lower() not in {"0", "false", "no"}
    enable_graph_multi_hop: bool = os.getenv("ENABLE_GRAPH_MULTI_HOP", "1").lower() not in {"0", "false", "no"}
    enable_graph_hit_dedup: bool = os.getenv("ENABLE_GRAPH_HIT_DEDUP", "1").lower() not in {"0", "false", "no"}
    enable_graph_hit_rerank: bool = os.getenv("ENABLE_GRAPH_HIT_RERANK", "1").lower() not in {"0", "false", "no"}
    enable_graph_hit_truncate: bool = os.getenv("ENABLE_GRAPH_HIT_TRUNCATE", "1").lower() not in {"0", "false", "no"}
    graph_max_hops: int = max(1, int(os.getenv("GRAPH_MAX_HOPS", "2")))
    graph_synonyms_per_entity: int = max(0, int(os.getenv("GRAPH_SYNONYMS_PER_ENTITY", "2")))
    graph_top_k: int = max(1, int(os.getenv("GRAPH_TOP_K", "8")))
    graph_score_exact_match_weight: float = float(os.getenv("GRAPH_SCORE_EXACT_MATCH_WEIGHT", "3.0"))
    graph_score_partial_match_weight: float = float(os.getenv("GRAPH_SCORE_PARTIAL_MATCH_WEIGHT", "1.5"))
    graph_score_hop_weight: float = float(os.getenv("GRAPH_SCORE_HOP_WEIGHT", "1.0"))
    graph_score_relation_weight: float = float(os.getenv("GRAPH_SCORE_RELATION_WEIGHT", "0.4"))
    graph_score_evidence_weight: float = float(os.getenv("GRAPH_SCORE_EVIDENCE_WEIGHT", "0.3"))
    graph_score_source_weight: float = float(os.getenv("GRAPH_SCORE_SOURCE_WEIGHT", "0.2"))
    enable_query_rewrite: bool = os.getenv("ENABLE_QUERY_REWRITE", "1").lower() not in {"0", "false", "no"}
    enable_hybrid_retrieval: bool = os.getenv("ENABLE_HYBRID_RETRIEVAL", "1").lower() not in {"0", "false", "no"}
    enable_rerank: bool = os.getenv("ENABLE_RERANK", "1").lower() not in {"0", "false", "no"}
    enable_evidence_compression: bool = os.getenv("ENABLE_EVIDENCE_COMPRESSION", "1").lower() not in {"0", "false", "no"}
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")
    rerank_batch_size: int = max(1, int(os.getenv("RERANK_BATCH_SIZE", "16")))
    rrf_k: int = max(1, int(os.getenv("RRF_K", "60")))
    debug_mode: bool = os.getenv("DEBUG_MODE", "0").lower() in {"1", "true", "yes"}
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "12"))
    short_memory_max_turns: int = max(1, int(os.getenv("SHORT_MEMORY_MAX_TURNS", "8")))
    long_memory_max_turns: int = max(10, int(os.getenv("LONG_MEMORY_MAX_TURNS", "1000")))
    short_memory_prompt_turns: int = max(1, int(os.getenv("SHORT_MEMORY_PROMPT_TURNS", "3")))
    memory_recent_turns_for_intent: int = max(1, int(os.getenv("MEMORY_RECENT_TURNS_FOR_INTENT", "4")))
    memory_reuse_similarity_threshold: float = float(os.getenv("MEMORY_REUSE_SIMILARITY_THRESHOLD", "0.84"))
    long_memory_max_chars: int = max(100, int(os.getenv("LONG_MEMORY_MAX_CHARS", "500")))
    domain_keywords: str = os.getenv(
        "DOMAIN_KEYWORDS",
        "人工智能,机器学习,深度学习,神经网络,大模型,LLM,Transformer,卷积神经网络,循环神经网络,CNN,RNN,LSTM,GRU,注意力机制,预训练模型,微调,LoRA,提示词工程,向量检索,嵌入,语义向量,知识图谱,实体抽取,关系抽取,多轮对话,检索增强,RAG,向量数据库,相似度计算,聚类,分类,回归,决策树,随机森林,强化学习,监督学习,无监督学习,半监督学习,过拟合,欠拟合,归一化,激活函数,损失函数,梯度下降,反向传播,计算机视觉,图像分类,目标检测,语义分割,自然语言处理,NLP,词嵌入,BERT,GPT,文本生成,情感分析,命名实体识别,机器翻译",
    )


def ensure_project_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
