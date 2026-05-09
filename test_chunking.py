import os
import sys
from pathlib import Path

# 确保使用本地 src 下的最新代码，而非已安装的旧版本
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kg_rag_demo.parsers import DocumentParser
from kg_rag_demo.chunking import chunk_documents
from kg_rag_demo.config import Settings

# 强制 UTF-8 输出，避免 Windows GBK 编码报错
sys.stdout.reconfigure(encoding='utf-8')

def test_file_chunking(file_path: str, settings: Settings):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    print(f"\n{'='*20} Testing: {path.name} {'='*20}")
    
    # 1. Parse
    parser = DocumentParser(progress_callback=lambda x: print(f"  [Parser] {x}"))
    docs = parser.parse_path(path)
    print(f"  Parsed into {len(docs)} document objects.")

    # 2. Chunk
    chunks = chunk_documents(docs, settings)
    print(f"  Chunked into {len(chunks)} chunks.")

    # 3. Analyze Results
    for i, chunk in enumerate(chunks, start=1):
        # 仅显示前 2 个和最后 1 个，避免输出过多
        if i > 2 and i < len(chunks):
            continue
            
        print(f"\n  --- Chunk {i} ---")
        print(f"  [Modality]: {chunk.modality}")
        print(f"  [Metadata]: {chunk.metadata}")
        snippet = chunk.text[:150].replace('\n', ' ')
        print(f"  [Text Snippet]: {snippet}...")
        print(f"  [Length]: {len(chunk.text)} characters")
        
        # 检查重叠：显示结尾和下一个开头的对比（如果是连续块）
        if i < len(chunks) and i <= 2:
            next_chunk = chunks[i]
            overlap_end = chunk.text[-50:].replace('\n', ' ')
            overlap_start = next_chunk.text[:50].replace('\n', ' ')
            print(f"  [Overlap Check]: End: '...{overlap_end}'")
            print(f"  [Overlap Check]: Next Start: '{overlap_start}...'")

def main():
    settings = Settings()
    # 根据用户要求修改参数：chunk_size=1200, overlap=15% (1200 * 0.15 = 180)
    settings.chunk_size = 1200 
    settings.chunk_overlap = 180

    data_dir = Path("data")
    test_files = [
        data_dir / "BERT.pdf",
        data_dir / "AI_QA_Dataset.xlsx",
        data_dir / "ICLR 2026 _ LLM×Graph论文总结【LLM4Graph与Graph4LLM】.md"
    ]

    for file in test_files:
        test_file_chunking(str(file), settings)

if __name__ == "__main__":
    main()
