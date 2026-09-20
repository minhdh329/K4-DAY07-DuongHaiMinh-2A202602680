from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker, FixedSizeChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# 1. Cập nhật đường dẫn tới 2 file tài liệu Shopee của bạn
SAMPLE_FILES = [
    "data/ecommerce/shopee-return-refund-policy.md",
    "data/ecommerce/shopee-warranty-policy.md",
]


def load_and_chunk_documents(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths and split them into chunks."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    
    # 2. Tích hợp chiến lược Chunking (Bạn có thể đổi sang FixedSizeChunker ở đây để test)
    chunker = RecursiveChunker(chunk_size=400)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        
        # Cắt file thành các đoạn nhỏ (chunks)
        chunks = chunker.chunk(content)
        
        # Lưu từng chunk thành một Document riêng biệt với metadata
        for i, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}_chunk_{i}",
                    content=chunk_text,
                    metadata={"source": str(path), "extension": path.suffix.lower()},
                )
            )

    return documents


def real_llm(prompt: str) -> str:
    """Gọi Gemini API thật để trả lời câu hỏi dựa trên ngữ cảnh."""
    import os
    try:
        from google import genai
        # Khởi tạo client tự động lấy GEMINI_API_KEY từ biến môi trường (file .env)
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
        )
        return response.text
    except ImportError:
        return "Lỗi: Chưa cài đặt thư viện. Hãy chạy 'pip install google-genai'"
    except Exception as e:
        return f"[LỖI LLM]: {e}"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    # --- THÊM 2 DÒNG NÀY ĐỂ BỎ QUA FILE .ENV ---
    os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6Ja9x40Wj3O8DskAsZ7FLQBp-HnR7NMzGGWOERkGI8caw"
    os.environ["EMBEDDING_PROVIDER"] = "gemini"
    # -------------------------------------------
    
    files = sample_files or SAMPLE_FILES
    
    # 3. Đặt câu hỏi mặc định là 1 trong 5 câu benchmark
    query = question or "Người mua có tối đa bao nhiêu ngày để yêu cầu Trả hàng/Hoàn tiền?"

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    # Gọi hàm load_and_chunk_documents thay vì hàm cũ
    docs = load_and_chunk_documents(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded and chunked into {len(docs)} documents (chunks)")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')} id={result['id']}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=real_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())