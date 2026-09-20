import os
import yaml
from pathlib import Path

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker, FixedSizeChunker, SentenceChunker
from src.embeddings import GeminiEmbedder
from src.models import Document
from src.store import EmbeddingStore

# 1. THIẾT LẬP API KEY
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6L6ccRIvD5KLaMxJMZbp2p0AlSmPGYUKxV_qNWLBtvhLw"
os.environ["EMBEDDING_PROVIDER"] = "gemini"

def load_and_chunk_documents(folder_path: str) -> list[Document]:
    """Tự động đọc toàn bộ file .md và .txt trong thư mục ecommerce."""
    documents = []
    # Thay đổi chiến lược Chunking tại đây để so sánh
    chunker = RecursiveChunker(chunk_size=400) 
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Không tìm thấy thư mục {folder_path}")
        return []

    for path in folder.glob("*.*"):
        if path.suffix.lower() not in [".md", ".txt"]:
            continue
            
        content = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(content)
        
        # Trích xuất metadata cơ bản (để test metadata_filter của Lazada)
        # Trong thực tế, bạn có thể parse cục YAML ở đầu mỗi file .md
        doc_metadata = {"source": path.name}
        if "lazada" in path.name.lower():
            doc_metadata.update({"audience": "seller", "platform": "Lazada"})
            
        for i, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}_chunk_{i}",
                    content=chunk_text,
                    metadata=doc_metadata
                )
            )
    return documents

def real_llm(prompt: str) -> str:
    """Gọi Gemini API để sinh câu trả lời."""
    try:
        from google import genai
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"[LỖI LLM]: {e}"

def main():
    # 2. ĐƯỜNG DẪN TỚI FILE YAML CỦA BẠN
    yaml_path = "benchmark_queries.yaml" 
    if not os.path.exists(yaml_path):
        yaml_path = "data/ecommerce/benchmark_queries.yaml"
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    queries = data.get("queries", [])

    print("Đang nạp toàn bộ tài liệu trong data/ecommerce/ và tính toán vector...")
    
    # Nạp toàn bộ tài liệu từ thư mục
    docs = load_and_chunk_documents("data/ecommerce")
    
    store = EmbeddingStore(embedding_fn=GeminiEmbedder())
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=real_llm)

    print(f"Đã lưu {store.get_collection_size()} chunks. Bắt đầu đánh giá {len(queries)} câu hỏi...\n" + "="*60)
    
    passed_retrieval = 0
    for q in queries:
        print(f"\n[Câu {q['id']} - {q.get('platform', 'Chung')}]: {q['query']}")
        
        # 3. XỬ LÝ METADATA FILTER CHO CÂU Q2
        if "metadata_filter" in q:
            print(f"   *Sử dụng bộ lọc:* {q['metadata_filter']}")
            results = store.search_with_filter(q['query'], top_k=3, metadata_filter=q['metadata_filter'])
        else:
            results = store.search(q['query'], top_k=3)
            
        retrieved_texts = [res['content'].lower() for res in results]
        
        # Kiểm tra nội dung kỳ vọng (gold_contains)
        gold_text = q['gold_contains'].lower()
        found = any(gold_text in text for text in retrieved_texts)
        
        if found:
            print("✅ ĐẠT: Truy xuất đúng đoạn văn bản chứa đáp án.")
            passed_retrieval += 1
        else:
            print("❌ TRƯỢT: Không tìm thấy đoạn văn bản chứa đáp án chuẩn.")
            print(f"   (Kỳ vọng: '{q['gold_contains']}')")
            
        print("-> Agent Trả lời:")
        print(agent.answer(q['query'], top_k=3))
        print("-" * 60)

    print(f"\n=== TỔNG KẾT CHIẾN LƯỢC ===")
    print(f"Tỷ lệ truy xuất đúng ngữ cảnh: {passed_retrieval}/{len(queries)}")

if __name__ == "__main__":
    main()