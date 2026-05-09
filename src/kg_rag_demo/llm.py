from __future__ import annotations

import base64
import json
import re
import time
from functools import cached_property
from typing import Any

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


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_retries=0,
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
        response = self._create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def image_to_text(self, image_bytes: bytes, instruction: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = self._create_chat_completion(
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

    def _create_chat_completion(self, messages: list[dict[str, Any]], temperature: float) -> Any:
        last_error: Exception | None = None
        total_attempts = self.settings.openai_max_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.settings.openai_chat_model,
                    temperature=temperature,
                    messages=messages,
                    timeout=self.settings.openai_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= total_attempts:
                    raise
                sleep_seconds = self.settings.openai_retry_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("聊天接口调用失败，且未捕获到具体异常。")

    @staticmethod
    def _safe_load_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            parts = raw.split("\n", 1)
            raw = parts[1] if len(parts) > 1 else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        return {"entities": [], "relations": []}
