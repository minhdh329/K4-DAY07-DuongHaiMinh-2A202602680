from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Truy xuất các đoạn văn bản (chunks) liên quan nhất từ vector store
        results = self.store.search(question, top_k=top_k)
        
        # 2. Ghép nội dung các chunks lại để tạo thành ngữ cảnh (context)
        context_parts = []
        for res in results:
            context_parts.append(res["content"])
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        # 3. Xây dựng prompt kết hợp ngữ cảnh và câu hỏi của người dùng
        prompt = (
            f"Dựa vào các thông tin ngữ cảnh sau đây, hãy trả lời câu hỏi.\n"
            f"Nếu không có thông tin trong ngữ cảnh, hãy trả lời 'Tôi không biết'.\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Trả lời:"
        )
        
        # 4. Gọi mô hình ngôn ngữ (LLM) để sinh câu trả lời
        return self.llm_fn(prompt)