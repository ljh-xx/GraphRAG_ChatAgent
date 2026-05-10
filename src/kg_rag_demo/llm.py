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

输出格式:
{
  "entities": [{"name": "实体名", "label": "实体类型"}],
  "relations": [{"source": "实体A", "target": "实体B", "relation": "关系", "evidence": "证据短语"}]
}

要求:
1. 最多抽取 8 个实体和 12 条关系。
2. 实体名必须来自原文，不要凭空编造。
3. 如果内容不足，返回空数组。
4. 只输出 JSON，不要输出解释。
"""


ANSWER_PROMPT = """你是一个严谨的中文知识问答助手。
请基于给定的检索片段和图谱关系回答问题。

要求:
1. 优先使用已给出的资料，不要臆造。
2. 如果资料不足，请明确说"资料不足"。
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
2. 不要抽取"什么""如何""为什么""组成""有哪些"这类问法词。
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
请先根据问题生成一段"可能的教材式回答"，用于 HyDE 检索。

要求：
1. 回答控制在 80~160 字。
2. 包含尽可能多的关键术语与概念关系。
3. 不要声明"不确定"，直接给出一个有信息量的假设性答案文本。
"""

GRAPH_ENTITY_SYNONYM_PROMPT = """你是知识图谱查询扩展助手。
请基于输入的实体列表，为每个实体补充查询友好的同义词/别称（用于并行检索）。

严格输出 JSON，格式：
{
  "items": [
    {"entity": "实体A", "synonyms": ["同义词1", "同义词2"]}
  ]
}

要求：
1. 只输出与实体语义等价或高相关的别称，不要扩展上下位概念。
2. 每个实体同义词数量不超过指定上限。
3. 可以包含中英混合术语（如 transaction）。
4. 若没有可靠同义词，返回空数组。
5. 只输出 JSON，不要解释。
"""

INTENT_CLASSIFY_PROMPT = """你是多轮问答中的意图路由助手。你只能返回三类意图：
- unrelated: 与数据库/课程主题无关，不应调用RAG/KG检索
- already_retrieved: 与最近已检索的问题高度相关，可复用短期记忆
- need_retrieve: 需要执行新的RAG/KG检索

严格输出 JSON，格式：
{
  "intent": "unrelated|already_retrieved|need_retrieve",
  "reason": "简短原因"
}

要求：
1. 判断时可使用最近对话摘要与主题关键词。
2. 默认保守：不确定时返回 need_retrieve。
3. 只输出 JSON，不要解释。
"""

MEMORY_REUSE_ANSWER_PROMPT = """你是一个多轮问答助手。下面给出历史检索提示词（包含当时证据）和历史回答。
请优先复用历史证据回答当前问题；若历史证据不足，明确说"资料不足"，不要编造新事实。
"""

LONG_MEMORY_WRITE_INTENT_PROMPT = """你是长期记忆写入判断器。判断当前用户问题是否包含"应长期保存的用户事实"。
典型应写入的内容：
- 用户自我信息（姓名、身份、偏好、限制、长期目标）
- 用户给助手的长期人设/规则
- 未来对话需要持续记住的约束

严格输出 JSON：
{
  "should_write": true/false,
  "reason": "简短原因"
}

要求：
1. 一般知识问答内容（如课程问题）通常不写入长期记忆。
2. 不确定时返回 false。
3. 只输出 JSON。
"""

LONG_MEMORY_EXTRACT_PROMPT = """你是长期记忆提炼助手。请从本轮对话中提炼1条可长期保存的用户信息。
严格输出 JSON：
{
  "memory": "一句话的长期记忆内容"
}

要求：
1. 只保留未来对话稳定有用的信息。
2. 不要记录一次性的问答细节。
3. 若没有可提炼内容，返回空字符串。
4. 只输出 JSON。
"""

EVIDENCE_SELECTION_PROMPT = """你是证据筛选助手。请从候选检索片段中挑选最能回答问题的片段ID。
严格输出 JSON：
{
  "keep_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}

要求：
1. 优先选择信息互补、非重复的证据。
2. 如果候选中存在冲突，优先保留表述更完整、更贴近问题的片段。
3. 最多保留指定数量。
4. 只输出 JSON，不要解释。
"""


class APIResourceManager:
    """API Key + Model 资源池，支持 Key 轮换与模型降级。

    故障转移策略（优先换模型，同账号其他模型通常还有量）：
    1. 当前模型不可用 → 降级到下一个模型（同 Key）
    2. 所有模型耗尽 → 轮换到下一个 Key
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
        """返回 (key_index, model, client)，优先换模型再换 Key."""
        for key_offset in range(len(self.api_keys)):
            key_idx = (self._current_key_idx + key_offset) % len(self.api_keys)
            for model_offset in range(len(self.chat_models)):
                model_idx = (self._current_model_idx + model_offset) % len(self.chat_models)
                model = self.chat_models[model_idx]
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
            model_name=model_path,
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
                    if self._should_switch_resource(exc):
                        break
                    if attempt < attempts_per_pair - 1:
                        sleep_seconds = self.settings.openai_retry_backoff_seconds * (2 ** attempt)
                        time.sleep(sleep_seconds)

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
        memory_context: str = "",
    ) -> str:
        prompt = self.build_answer_prompt(question, chunk_contexts, graph_contexts, memory_context=memory_context)
        return self.chat(ANSWER_PROMPT, prompt)

    def answer_question_with_prompt(
        self,
        question: str,
        chunk_contexts: list[dict[str, Any]],
        graph_contexts: list[dict[str, Any]],
        memory_context: str = "",
    ) -> tuple[str, str]:
        prompt = self.build_answer_prompt(question, chunk_contexts, graph_contexts, memory_context=memory_context)
        answer = self.chat(ANSWER_PROMPT, prompt)
        return answer, prompt

    def answer_from_memory_prompt(
        self,
        question: str,
        historical_prompt: str,
        historical_answer: str,
        memory_context: str = "",
    ) -> tuple[str, str]:
        user_prompt = (
            f"会话记忆:\n{memory_context or '无'}\n\n"
            f"当前问题:\n{question.strip()}\n\n"
            f"历史回答:\n{historical_answer.strip()}\n\n"
            f"历史检索提示词:\n{historical_prompt.strip()}\n"
        )
        answer = self.chat(MEMORY_REUSE_ANSWER_PROMPT, user_prompt)
        return answer, user_prompt

    def classify_intent_with_context(
        self,
        question: str,
        recent_turns: list[dict[str, Any]],
        domain_keywords: list[str],
    ) -> dict[str, str]:
        payload = {
            "question": question.strip(),
            "domain_keywords": [item.strip() for item in domain_keywords if item.strip()],
            "recent_turns": recent_turns,
        }
        raw = self.chat(INTENT_CLASSIFY_PROMPT, json.dumps(payload, ensure_ascii=False))
        data = self._safe_load_json(raw)
        intent = self._clean_single_line(data.get("intent")).lower()
        if intent not in {"unrelated", "already_retrieved", "need_retrieve"}:
            intent = "need_retrieve"
        reason = self._clean_single_line(data.get("reason")) or "fallback"
        return {"intent": intent, "reason": reason}

    def classify_long_memory_write_intent(
        self,
        question: str,
        recent_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "question": question.strip(),
            "recent_turns": recent_turns,
        }
        raw = self.chat(LONG_MEMORY_WRITE_INTENT_PROMPT, json.dumps(payload, ensure_ascii=False))
        data = self._safe_load_json(raw)
        should_write = bool(data.get("should_write", False))
        reason = self._clean_single_line(data.get("reason")) or "fallback"
        return {"should_write": should_write, "reason": reason}

    def extract_long_term_memory(self, question: str, answer: str) -> str:
        payload = {
            "question": question.strip(),
            "answer": answer.strip(),
        }
        raw = self.chat(LONG_MEMORY_EXTRACT_PROMPT, json.dumps(payload, ensure_ascii=False))
        data = self._safe_load_json(raw)
        return self._clean_single_line(data.get("memory"))

    def build_answer_prompt(
        self,
        question: str,
        chunk_contexts: list[dict[str, Any]],
        graph_contexts: list[dict[str, Any]],
        memory_context: str = "",
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
        prompt = f"""会话记忆:
{memory_context or "无"}

问题:
{question}

检索片段:
{chunk_text or "无"}

图谱关系:
{graph_text or "无"}
"""
        return prompt

    def select_evidence_chunks(
        self,
        question: str,
        chunk_contexts: list[dict[str, Any]],
        max_chunks: int,
    ) -> list[str]:
        if not chunk_contexts:
            return []
        max_chunks = max(1, max_chunks)
        chunk_lines: list[str] = []
        for item in chunk_contexts:
            chunk_id = str(item.get("chunk_id", "")).strip()
            text = self._clean_single_line(str(item.get("text", "")))[:240]
            title = self._clean_single_line(str(item.get("title", "")))
            chunk_lines.append(f"- id={chunk_id} | title={title} | text={text}")
        prompt = (
            f"问题:\n{question}\n\n"
            f"最多保留证据数量: {max_chunks}\n\n"
            f"候选片段:\n" + "\n".join(chunk_lines)
        )
        raw = self.chat(EVIDENCE_SELECTION_PROMPT, prompt)
        payload = self._safe_load_json(raw)
        keep_ids: list[str] = []
        for item in payload.get("keep_chunk_ids", []):
            if not isinstance(item, str):
                continue
            chunk_id = item.strip()
            if chunk_id and chunk_id not in keep_ids:
                keep_ids.append(chunk_id)
            if len(keep_ids) >= max_chunks:
                break
        return keep_ids

    def expand_graph_query_entities(
        self,
        entities: list[str],
        max_synonyms_per_entity: int = 2,
    ) -> dict[str, list[str]]:
        cleaned_entities = []
        for item in entities:
            name = self._clean_single_line(item)
            if name and name not in cleaned_entities:
                cleaned_entities.append(name)
        if not cleaned_entities or max_synonyms_per_entity <= 0:
            return {name: [] for name in cleaned_entities}

        user_prompt = (
            f"实体列表: {json.dumps(cleaned_entities, ensure_ascii=False)}\n"
            f"每个实体最多同义词数: {max_synonyms_per_entity}"
        )
        raw = self.chat(GRAPH_ENTITY_SYNONYM_PROMPT, user_prompt)
        payload = self._safe_load_json(raw)
        result: dict[str, list[str]] = {name: [] for name in cleaned_entities}

        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            entity = self._clean_single_line(item.get("entity"))
            if entity not in result:
                continue
            synonyms: list[str] = []
            for syn in item.get("synonyms", []):
                term = self._clean_single_line(syn)
                if not term or term == entity or term in synonyms:
                    continue
                synonyms.append(term)
                if len(synonyms) >= max_synonyms_per_entity:
                    break
            result[entity] = synonyms
        return result

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
        """底层调用（无重试，重试逻辑由 chat() 处理）"""
        return self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            timeout=self.settings.openai_timeout_seconds,
        )

    @staticmethod
    def _should_switch_resource(exc: Exception) -> bool:
        """判断是否应切换 Key/Model（资源类错误）。"""
        status_code = (
            getattr(exc, "status_code", None)
            or getattr(getattr(exc, "response", None), "status_code", None)
        )
        if status_code in {429, 402}:
            return True
        msg = str(exc).lower()
        if any(kw in msg for kw in ["quota", "rate limit", "insufficient", "exhausted", "超出", "额度", "限制"]):
            return True
        if status_code in {404, 403} and any(
            kw in msg for kw in ["model", "endpoint", "not found", "access", "invalid", "not exist"]
        ):
            return True
        return False

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
