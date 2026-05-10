from __future__ import annotations

import base64
import json
import re
import time
from functools import cached_property
from typing import Any, Iterator

from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import Settings, resolve_local_embed_model, resolve_local_hf_model
from .models import EntityNode, RelationEdge


GRAPH_SCHEMA_PROMPT = """你是一个知识图谱抽取助手。
请从给定文本中抽取实体和关系，并严格输出 JSON。

输出格式 (严格 JSON):
{
  "entities": [{"name": "实体名", "label": "类型"}],
  "relations": [{"source": "源实体", "target": "目标实体", "relation": "关系描述", "evidence": "原文证据"}]
}

要求:
1. 最多抽取 8 个实体和 12 条关系。
2. 实体名必须来自原文，不要凭空编造。
3. 如果内容不足，返回空数组。
4. 实体可以是具体事物（如 AI 模型），也可以是抽象概念（如 数字经济、核心生产力、2030目标）。
5. 关系应体现逻辑连接（如：扮演角色、属于、包含、旨在实现）。
6. 即使文本简短，也请尽力提取其中蕴含的逻辑结构。
7. 只输出 JSON，不要输出解释。
"""


ANSWER_PROMPT = """你是一个严谨的中文知识问答助手。
请基于给定的检索片段和图谱关系回答问题。

要求:
1. 优先使用已给出的资料，不要臆造。
2. 如果资料不足，请明确说“资料不足”。
3. 回答尽量简洁，并在末尾附上引用来源文件名。
"""


QUERY_ENTITY_PROMPT = """你是一个问题实体抽取助手。
请从用户问题中抽取最关键的知识图谱检索实体，并严格输出 JSON。

输出格式:
{
  "entities": ["实体1", "实体2"]
}

要求:
1. 只抽取适合在知识图谱中查找的核心实体或名词短语。
2. 不要抽取“什么”“如何”“为什么”“组成”“有哪些”这类问法词。
3. 最多输出 5 个实体。
4. 如果无法判断，返回空数组。
5. 只输出 JSON，不要输出解释。
"""


QUERY_REWRITE_PROMPT = """你是一个检索查询改写助手，请把用户问题改写为更适合向量检索的查询。
严格输出 JSON，格式如下：
{
  "de_colloquialized_query": "去口语化后的查询",
  "synonym_keyword_query": "同义改写 + 关键词化查询"
}

要求：
1. 两个字段都必须是中文字符串。
2. 不要回答问题本身，只输出用于检索的查询。
3. 关键词查询应保留关键术语并补充常见同义表达。
4. 只输出 JSON，不要输出解释。
"""


HYPOTHETICAL_ANSWER_PROMPT = """你是一个检索增强助手。
请先根据问题生成一段“可能的教材式回答”，用于 HyDE 检索。

要求：
1. 回答控制在 80~160 字。
2. 包含尽可能多的关键术语与概念关系。
3. 不要声明“不确定”，直接给出一个有信息量的假设性答案文本。
"""


class APIResourceManager:
    """API Key + Model 资源池，支持 Key 轮换与模型降级。

    故障转移策略：
    1. 当前 Key 配额耗尽 → 轮换到下一个 Key（同模型）
    2. 所有 Key 耗尽 → 降级到下一个模型
    3. 遍历所有 (Key, Model) 组合，直到找到可用的或全部耗尽
    """

    def __init__(self, api_keys: list[str], chat_models: list[str]) -> None:
        if not api_keys:
            raise ValueError("至少需要一个 API Key，请设置 OPENAI_API_KEY 或 OPENAI_API_KEYS")
        if not chat_models:
            raise ValueError("至少需要一个 Chat Model，请设置 OPENAI_CHAT_MODEL 或 OPENAI_CHAT_MODELS")
        self.api_keys = api_keys
        self.chat_models = chat_models
        self._failed: set[tuple[int, str]] = set()
        self._current_key_idx = 0
        self._current_model_idx = 0

    @property
    def current_key_index(self) -> int:
        return self._current_key_idx

    @property
    def current_model(self) -> str:
        return self.chat_models[self._current_model_idx]

    def get_client(self, base_url: str) -> tuple[int, str, OpenAI]:
        """返回 (key_index, model, client)，遍历所有未失效的 (Key, Model) 组合."""
        for model_offset in range(len(self.chat_models)):
            model_idx = (self._current_model_idx + model_offset) % len(self.chat_models)
            model = self.chat_models[model_idx]
            for key_offset in range(len(self.api_keys)):
                key_idx = (self._current_key_idx + key_offset) % len(self.api_keys)
                if (key_idx, model) in self._failed:
                    continue
                try:
                    client = OpenAI(
                        api_key=self.api_keys[key_idx],
                        base_url=base_url,
                        max_retries=0,
                    )
                    self._current_key_idx = key_idx
                    self._current_model_idx = model_idx
                    return key_idx, model, client
                except Exception:
                    self._failed.add((key_idx, model))
                    continue

        raise RuntimeError("所有 API Key 与模型的组合均已耗尽，请检查配额或补充新 Key。")

    def mark_failed(self, key_index: int, model: str) -> None:
        self._failed.add((key_index, model))

    def rotate_key(self) -> None:
        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.resource_mgr = APIResourceManager(settings.openai_api_keys, settings.openai_chat_models)
        self.current_key_index: int = 0
        self.current_model: str = settings.openai_chat_models[0]
        self.client: OpenAI | None = None
        self._init_client()

    def _init_client(self) -> None:
        self.current_key_index, self.current_model, self.client = self.resource_mgr.get_client(
            self.settings.openai_base_url
        )

    @cached_property
    def embedder(self) -> SentenceTransformer:
        model_path = resolve_local_embed_model(self.settings.local_embed_model)
        return SentenceTransformer(
            model_path,
            device=self.settings.embedding_device,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        query = self._format_query_for_embedding(text)
        vector = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    @cached_property
    def reranker(self) -> CrossEncoder:
        model_path = resolve_local_hf_model(self.settings.rerank_model)
        return CrossEncoder(
            model_name_or_path=model_path,
            device=self.settings.rerank_device,
        )

    def rerank(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(question, text) for text in texts]
        scores = self.reranker.predict(
            pairs,
            batch_size=self.settings.rerank_batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """带故障转移的 Chat 接口：Key 轮换 + 模型降级."""
        last_error: Exception | None = None
        total_pairs = len(self.settings.openai_api_keys) * len(self.settings.openai_chat_models)
        attempts_per_pair = max(1, self.settings.openai_max_retries)

        for pair_attempt in range(total_pairs):
            for attempt in range(attempts_per_pair):
                try:
                    response = self._create_chat_completion(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                    )
                    return response.choices[0].message.content or ""
                except Exception as exc:
                    last_error = exc
                    if self._is_quota_error(exc):
                        break  # 配额类错误不重试同 Key，直接切换
                    if attempt < attempts_per_pair - 1:
                        sleep_seconds = self.settings.openai_retry_backoff_seconds * (2 ** attempt)
                        time.sleep(sleep_seconds)

            # 当前 (Key, Model) 耗尽，标记并切换到下一个可用组合
            self.resource_mgr.mark_failed(self.current_key_index, self.current_model)
            try:
                self._init_client()
            except RuntimeError:
                break

        if last_error:
            raise last_error
        raise RuntimeError("所有 API Key 与模型组合均已耗尽，无法完成调用。")

    def image_to_text(self, image_bytes: bytes, instruction: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = self._create_chat_completion(
            model=self.current_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def extract_graph(self, text: str) -> tuple[list[EntityNode], list[RelationEdge]]:
        raw = self.chat(GRAPH_SCHEMA_PROMPT, text[:4000])
        payload = self._safe_load_json(raw)

        entities = [
            EntityNode(name=item["name"].strip(), label=item.get("label", "Entity").strip() or "Entity")
            for item in payload.get("entities", [])
            if item.get("name")
        ]
        relations = [
            RelationEdge(
                source=item["source"].strip(),
                target=item["target"].strip(),
                relation=item["relation"].strip(),
                evidence=item.get("evidence", "").strip(),
            )
            for item in payload.get("relations", [])
            if item.get("source") and item.get("target") and item.get("relation")
        ]
        return entities, relations

    def extract_query_entities(self, question: str) -> list[str]:
        raw = self.chat(QUERY_ENTITY_PROMPT, question[:1000])
        payload = self._safe_load_json(raw)
        entities = []
        for item in payload.get("entities", []):
            if isinstance(item, str):
                name = item.strip()
                if name and name not in entities:
                    entities.append(name)
        return entities[:5]

    def build_query_variants(self, question: str) -> dict[str, str]:
        raw = self.chat(QUERY_REWRITE_PROMPT, question[:1000])
        payload = self._safe_load_json(raw)
        route_1 = self._clean_single_line(payload.get("de_colloquialized_query", ""))
        route_2 = self._clean_single_line(payload.get("synonym_keyword_query", ""))
        if not route_1:
            route_1 = self._fallback_de_colloquialized(question)
        if not route_2:
            route_2 = route_1
        return {
            "de_colloquialized_query": route_1,
            "synonym_keyword_query": route_2,
        }

    def generate_hypothetical_answer_query(self, question: str) -> str:
        text = self.chat(HYPOTHETICAL_ANSWER_PROMPT, question[:1000])
        cleaned = self._clean_single_line(text)
        return cleaned or self._fallback_de_colloquialized(question)

    def answer_question(
        self,
        question: str,
        chunk_contexts: list[dict[str, Any]],
        graph_contexts: list[dict[str, Any]],
    ) -> str:
        chunk_text = "\n\n".join(
            [
                f"[片段{i}] 文件: {item['source_path']} 页码: {item.get('page_number')}\n{item['text']}"
                for i, item in enumerate(chunk_contexts, start=1)
            ]
        )
        graph_text = "\n".join(
            [
                f"- {item['source']} --{item['relation']}--> {item['target']} (证据: {item.get('evidence', '')})"
                for item in graph_contexts
            ]
        )
        prompt = f"""问题:
{question}

检索片段:
{chunk_text or "无"}

图谱关系:
{graph_text or "无"}
"""
        return self.chat(ANSWER_PROMPT, prompt)

    @staticmethod
    def _format_query_for_embedding(text: str) -> str:
        return f"为这个句子生成表示以用于检索相关文章：{text.strip()}"

    @staticmethod
    def _clean_single_line(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned.strip("`").strip()

    @staticmethod
    def _fallback_de_colloquialized(question: str) -> str:
        text = question.strip()
        replacements = [
            ("请问", ""),
            ("一下", ""),
            ("帮我", ""),
            ("给我", ""),
            ("能不能", ""),
            ("是什么", "定义"),
            ("啥是", "定义"),
            ("什么是", "定义"),
        ]
        for src, dst in replacements:
            text = text.replace(src, dst)
        text = re.sub(r"[？?]+$", "", text).strip()
        return text or question.strip()

    def _create_chat_completion(self, model: str, messages: list[dict[str, Any]], temperature: float) -> Any:
        """底层 Completion 调用（不带重试，由上层 chat() 处理）"""
        return self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            timeout=self.settings.openai_timeout_seconds,
        )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """判断是否为配额/限流类错误（应切换资源），而非请求格式错误（不应切换）"""
        status_code = (
            getattr(exc, "status_code", None)
            or getattr(getattr(exc, "response", None), "status_code", None)
        )
        if status_code in {429, 402}:
            return True
        msg = str(exc).lower()
        quota_keywords = ["quota", "rate limit", "insufficient", "exhausted", "超出", "额度", "限制", "quota exceeded"]
        return any(kw in msg for kw in quota_keywords)

    @staticmethod
    def _safe_load_json(raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}
