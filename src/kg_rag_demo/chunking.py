from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from .config import Settings
from .models import ChunkRecord, ParsedDocument

# ---------------------------------------------------------------------------
# token / 字符 测量
# ---------------------------------------------------------------------------

def _build_measure() -> Callable[[str], int]:
    """构建文本长度测量函数。

    优先使用 tiktoken (cl100k_base) 做 token 计数，这样 chunk 大小直接对应
    embedding 模型的输入窗口。如果 tiktoken 不可用，回退到字符计数。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def _token_count(text: str) -> int:
            return len(enc.encode(text))

        return _token_count
    except Exception:
        return len  # 回退到字符计数


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[ParsedDocument], settings: Settings
) -> list[ChunkRecord]:
    measure = _build_measure()

    # min_chunk_size: 过短的 chunk 会被合并到相邻 chunk，避免语义碎片
    min_chunk_size = max(30, settings.chunk_size // 20)

    chunks: list[ChunkRecord] = []

    for doc in documents:
        if doc.modality == "markdown":
            pieces = split_markdown(
                doc.text,
                settings.chunk_size,
                settings.chunk_overlap,
                measure=measure,
                min_chunk_size=min_chunk_size,
            )
        elif doc.modality == "pdf":
            pieces = split_text_recursive(
                doc.text,
                settings.chunk_size,
                settings.chunk_overlap,
                measure=measure,
                min_chunk_size=min_chunk_size,
            )
        elif doc.modality == "excel_qa":
            # QA 数据保持完整，不进行二次切分
            pieces = [doc.text]
        else:
            pieces = split_text_recursive(
                doc.text,
                settings.chunk_size,
                settings.chunk_overlap,
                measure=measure,
                min_chunk_size=min_chunk_size,
            )

        for order, piece in enumerate(pieces, start=1):
            metadata = doc.extra_meta.copy()
            if isinstance(piece, dict) and "text" in piece:
                text_content = piece["text"]
                metadata.update(piece.get("metadata", {}))
            else:
                text_content = piece

            chunk_id = _chunk_id(doc.doc_id, doc.page_number, order, text_content)

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    title=doc.title,
                    text=text_content,
                    modality=doc.modality,
                    page_number=doc.page_number,
                    order=order,
                    metadata=metadata,
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# 通用递归切分
# ---------------------------------------------------------------------------

def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
    *,
    measure: Callable[[str], int] = len,
    min_chunk_size: int = 30,
) -> list[str]:
    """递归切分文本，按分隔符优先级逐级降级，并保留 overlap。

    参数
    ----
    measure : Callable[[str], int]
        文本长度测量函数（token 计数或字符计数）。
    min_chunk_size : int
        过短的 chunk 将被延迟 flush，与后续内容合并。
    """
    if separators is None:
        separators = [
            "\n\n", "\n", "。", "？", "！", "；", ". ", "? ", "! ", "; ", " ", ""
        ]

    text = normalize_text(text)
    if measure(text) <= chunk_size:
        return [text] if text else []

    # 1. 选择分隔符 — 找到文本中实际存在的最高优先级分隔符
    final_sep = separators[-1]  # "" 兜底
    for s in separators:
        if s == "":
            final_sep = s
            break
        if s in text:
            final_sep = s
            break

    # 2. 按分隔符切分
    if final_sep:
        splits = text.split(final_sep)
    else:
        splits = list(text)

    # 3. 合并片段 + overlap
    final_chunks: list[str] = []
    current_doc: list[str] = []
    current_length = 0

    for s in splits:
        sep_cost = len(final_sep) if current_doc else 0

        if current_length + measure(s) + sep_cost > chunk_size:
            if current_doc:
                chunk_content = final_sep.join(current_doc)
                chunk_len = measure(chunk_content)

                # —— 最小 chunk 约束 ——
                # 如果当前块太短且下一个片段也不大，暂不 flush，继续累积
                if chunk_len < min_chunk_size and measure(s) < chunk_size:
                    current_doc.append(s)
                    current_length += measure(s) + (
                        len(final_sep) if len(current_doc) > 1 else 0
                    )
                    continue

                final_chunks.append(chunk_content)

                # 计算 overlap：从尾部往前取，不超过 chunk_overlap
                overlap_doc: list[str] = []
                overlap_length = 0
                for part in reversed(current_doc):
                    part_len = measure(part)
                    sep = len(final_sep) if overlap_doc else 0
                    if overlap_length + part_len + sep <= chunk_overlap:
                        overlap_doc.insert(0, part)
                        overlap_length += part_len + (
                            len(final_sep) if len(overlap_doc) > 1 else 0
                        )
                    else:
                        break
                current_doc = overlap_doc
                current_length = overlap_length

            # 单个片段巨大 → 递归切分
            if measure(s) > chunk_size:
                remaining_seps = (
                    separators[separators.index(final_sep) + 1:]
                    if final_sep in separators
                    else separators
                )
                if not remaining_seps:
                    remaining_seps = [""]

                sub_splits = split_text_recursive(
                    s,
                    chunk_size,
                    chunk_overlap,
                    remaining_seps,
                    measure=measure,
                    min_chunk_size=min_chunk_size,
                )
                if sub_splits:
                    final_chunks.extend(sub_splits[:-1])
                    # 追加到 current_doc（保留之前计算的 overlap 上下文）
                    current_doc.append(sub_splits[-1])
                    current_length += measure(sub_splits[-1]) + (
                        len(final_sep) if len(current_doc) > 1 else 0
                    )
                continue

        # 累积到当前块
        current_doc.append(s)
        current_length += measure(s) + (
            len(final_sep) if len(current_doc) > 1 else 0
        )

    # 收尾：最后一个块如果太短，尝试合并到前一个块
    if current_doc:
        tail = final_sep.join(current_doc)
        if measure(tail) < min_chunk_size and final_chunks:
            final_chunks[-1] = final_chunks[-1] + final_sep + tail
        else:
            final_chunks.append(tail)

    return final_chunks


# ---------------------------------------------------------------------------
# Markdown 结构化切分
# ---------------------------------------------------------------------------

# 一级标题或水平分割线，视为"章节大边界"
_MAJOR_BOUNDARY = re.compile(r"^(#{1,2}\s|--|-{3,}|\*{3,}|_{3,})")


def split_markdown(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    measure: Callable[[str], int] = len,
    min_chunk_size: int = 30,
) -> list[dict[str, Any]]:
    """按 Markdown 标题层级切分，合并小章节，并在章节间保留 overlap。"""
    text = normalize_text(text, preserve_trailing_spaces=True)
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] = {"title": "Root", "level": 0, "content": []}
    stack = [current_section]

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        m = header_pattern.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            new_section = {"title": title, "level": level, "content": [line]}
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack.append(new_section)
            sections.append(new_section)
        else:
            for sec in reversed(stack):
                sec["content"].append(line)
                break

    if not sections:
        return [
            {"text": p, "metadata": {}}
            for p in split_text_recursive(
                text, chunk_size, chunk_overlap, measure=measure,
                min_chunk_size=min_chunk_size,
            )
        ]

    # 第一个标题之前的文本作为 preamble
    if current_section["content"]:
        preamble_text = "\n".join(current_section["content"]).strip()
        if preamble_text:
            sections.insert(
                0, {"title": "Preamble", "level": 0, "content": current_section["content"]}
            )

    # —— 合并相邻小章节 ——
    merged: list[tuple[list[str], dict[str, Any]]] = []
    buffer_content: list[str] = []
    buffer_meta: dict[str, Any] | None = None
    buffer_len = 0

    for sec in sections:
        sec_text = "\n".join(sec["content"]).strip()
        if not sec_text:
            continue

        if buffer_meta is None:
            buffer_content = sec["content"]
            buffer_meta = {"section": sec["title"], "level": sec["level"]}
            buffer_len = measure(sec_text)
        elif buffer_len + measure(sec_text) < chunk_size:
            buffer_content.extend(sec["content"])
            buffer_len = measure("\n".join(buffer_content).strip())
            buffer_meta = {
                "section": f"{buffer_meta['section']} → {sec['title']}",
                "level": buffer_meta["level"],
            }
        else:
            merged.append((buffer_content, buffer_meta))
            buffer_content = sec["content"]
            buffer_meta = {"section": sec["title"], "level": sec["level"]}
            buffer_len = measure(sec_text)

    if buffer_content:
        merged.append((buffer_content, buffer_meta))

    # —— 生成最终 chunk（带章节间 overlap） ——
    final_chunks: list[dict[str, Any]] = []
    prev_tail = ""

    for content_lines, meta in merged:
        section_text = "\n".join(content_lines).strip()

        # 检查是否遇到大边界（一级/二级标题），重置 prev_tail
        first_line = content_lines[0] if content_lines else ""
        if _MAJOR_BOUNDARY.match(first_line):
            prev_tail = ""

        # 章节间 overlap：将前一个块的尾部拼接到当前块开头
        if prev_tail and section_text:
            section_text = prev_tail + "\n" + section_text

        if measure(section_text) > chunk_size:
            # 计算传给递归切分的有效 overlap：
            # section_text 已包含 prev_tail，所以需要扣减，避免两级 overlap 叠加
            effective_overlap = max(
                0,
                chunk_overlap - measure(prev_tail) if prev_tail else chunk_overlap,
            )
            sub_pieces = split_text_recursive(
                section_text,
                chunk_size,
                effective_overlap,
                measure=measure,
                min_chunk_size=min_chunk_size,
            )
            for p in sub_pieces:
                final_chunks.append({"text": p, "metadata": dict(meta)})
            if sub_pieces:
                last = sub_pieces[-1]
                prev_tail = last[-chunk_overlap:] if measure(last) > chunk_overlap else last
        else:
            final_chunks.append({"text": section_text, "metadata": dict(meta)})
            if measure(section_text) > chunk_overlap:
                # 取尾部约 chunk_overlap 长度（沿换行边界截断）
                tail_start = max(0, len(section_text) - chunk_overlap * 2)
                tail_candidate = section_text[tail_start:]
                # 从第一个换行后开始，避免在行中间硬截断
                nl = tail_candidate.find("\n")
                prev_tail = tail_candidate[nl + 1:] if nl != -1 and nl < len(tail_candidate) // 2 else tail_candidate
            else:
                prev_tail = section_text

    return final_chunks


# ---------------------------------------------------------------------------
# 兼容旧接口
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return split_text_recursive(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# 文本标准化
# ---------------------------------------------------------------------------

def normalize_text(text: str, *, preserve_trailing_spaces: bool = False) -> str:
    """标准化文本。

    - 压缩 3 个及以上连续换行为 2 个（保留段落间距）
    - 去除行尾多余空格（除非 preserve_trailing_spaces=True，用于 Markdown 强制换行）
    """
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    if preserve_trailing_spaces:
        # Markdown: 行尾 2 空格 = <br>，只去除超过 2 个的空格
        lines = []
        for line in text.splitlines():
            stripped = line.rstrip(" ")
            # 如果原来末尾恰好是 2 空格，保留
            if line.endswith("  ") and not line.endswith("   "):
                stripped = stripped + "  "
            lines.append(stripped)
        return "\n".join(lines).strip()
    else:
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _chunk_id(doc_id: str, page_number: int | None, order: int, text: str) -> str:
    digest = hashlib.md5(
        f"{doc_id}:{page_number}:{order}:{text[:100]}".encode("utf-8")
    ).hexdigest()
    return f"chunk_{digest}"
