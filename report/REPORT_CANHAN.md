# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Dương Hải Minh
**Nhóm:** G11
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:*
Độ tương tự cosine cao (kết quả tiến gần về 1) nghĩa là góc giữa hai vector nhúng rất nhỏ, thể hiện chúng hướng về cùng một phía trong không gian đa chiều. Về mặt văn bản, điều này chứng tỏ hai câu mang ý nghĩa cốt lõi rất giống nhau hoặc cùng nói về một chủ đề, bất kể độ dài hay số lượng từ vựng có khác biệt.

**Ví dụ có độ tương tự CAO:**
- Câu A:"Shopee sẽ tiến hành hoàn tiền cho khách hàng trong vòng tối đa 15 ngày."
- Câu B:"Người mua có thể nhận lại tiền từ hệ thống trong 15 ngày."
- Tại sao tương đồng: Dù sử dụng cấu trúc câu và từ đồng nghĩa khác biệt (hoàn tiền/nhận lại tiền, khách hàng/người mua), mô hình AI vẫn nắm bắt được hai câu này đang truyền tải chung một thông tin quy định nên các vector sẽ nằm sát nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A:"Thời hạn Người mua yêu cầu Trả hàng/Hoàn tiền là 15 ngày."
- Câu B:"Cách nướng bánh mì bơ tỏi giòn rụm bằng nồi chiên không dầu."
- Tại sao khác: Hai câu thuộc hai miền tri thức hoàn toàn không liên quan (chính sách thương mại điện tử so với công thức ẩm thực), do đó các vector đại diện sẽ chĩa về hai hướng hoàn toàn khác xa nhau trong không gian (cosine gần bằng 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:*
Độ tương tự cosine ưu việt hơn vì nó chỉ xét đến hướng của vector (đại diện cho ngữ nghĩa) mà bỏ qua độ lớn của vector (đại diện cho độ dài văn bản), giúp hệ thống dễ dàng ghép khớp một câu hỏi tìm kiếm rất ngắn với một đoạn tài liệu rất dài. Trong khi đó, khoảng cách Euclid đo lường khoảng cách hình học tuyệt đối, khiến hai văn bản dù có cùng ý nghĩa nhưng chênh lệch số lượng từ ngữ vẫn bị đánh giá là nằm xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
Bước nhảy (step) khi cửa sổ trượt giữa các chunk là: $500 - 50 = 450$ ký tự.
Áp dụng công thức tính tổng số chunk: $\lceil \frac{\text{Tổng chiều dài} - \text{Overlap}}{\text{Bước nhảy}} \rceil$
Thay số: $\lceil \frac{10000 - 50}{450} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$.
(Cách tính đếm bước: Chunk đầu tiên chiếm 500 ký tự, còn lại 9500 ký tự cần bao phủ. Với mỗi bước nhảy 450 ký tự, cần thêm $\lceil \frac{9500}{450} \rceil = 22$ chunk nữa).
> *Đáp án:*

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:*
Số lượng chunk sẽ tăng lên (cụ thể là tăng thành 25 chunks do bước nhảy bị thu hẹp xuống còn 400). Việc tăng độ chồng chéo giúp đảm bảo các câu văn, từ khóa, hoặc ý tưởng quan trọng không bị cắt đứt gãy mất nghĩa tại ranh giới chia cắt, giúp hệ thống RAG giữ được trọn vẹn ngữ cảnh khi truy xuất.
---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*
Tôi sử dụng biểu thức chính quy (regex) nhận diện dấu kết thúc câu, ví dụ như (?<=[.!?])\s+, để tách văn bản tại các vị trí dấu chấm, dấu hỏi hoặc dấu chấm than có kèm khoảng trắng. Để tránh lỗi, thuật toán xử lý các trường hợp ngoại lệ (edge cases) như chuỗi rỗng, khoảng trắng thừa hoặc các ký tự xuống dòng liên tiếp bằng cách dùng hàm strip() dọn dẹp và bỏ qua các câu không có nội dung thực tế trước khi gom chúng lại thành chunk.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*
Thuật toán hoạt động bằng cách ưu tiên cắt văn bản theo các dấu phân cách lớn (như hai dấu xuống dòng \n\n cho đoạn văn), nếu phần tử được cắt ra vẫn lớn hơn chunk_size, nó tiếp tục gọi đệ quy hàm _split để cắt bằng dấu phân cách nhỏ hơn (như dấu \n hoặc khoảng trắng). Trường hợp cơ sở (base case) xảy ra khi độ dài của đoạn văn bản đang xét nhỏ hơn hoặc bằng chunk_size (hoặc không còn dấu phân cách nào để cắt), lúc này đệ quy dừng lại và trả về chính đoạn văn bản đó.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*
Đối với hệ thống in-memory, các tài liệu được lưu trữ trong một cấu trúc danh sách (list) gồm các bản ghi (dictionary), trong đó mỗi bản ghi chứa id, nội dung văn bản, metadata và vector nhúng (embedding) của nó. Khi thực hiện search, câu hỏi của người dùng được chuyển thành vector, sau đó hệ thống quét qua toàn bộ kho lưu trữ để tính toán điểm số bằng hàm tính độ tương tự cosine (hoặc tích vô hướng dot product) và sắp xếp kết quả giảm dần để lấy top K.
**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*
Quá trình lọc (filter) metadata được thực hiện trước khi tính độ tương tự; hệ thống sẽ duyệt qua kho lưu trữ, chỉ giữ lại các tài liệu có metadata khớp với điều kiện, sau đó mới tính toán vector để tối ưu hiệu suất. Hàm delete_document hoạt động bằng cách quét qua danh sách dữ liệu hiện có và trực tiếp loại bỏ (remove/lọc bỏ) các bản ghi có doc_id trùng khớp với danh sách ID yêu cầu xóa.
### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*
Cấu trúc prompt được thiết kế theo ba phần rõ ràng: chỉ thị hệ thống (yêu cầu AI chỉ trả lời dựa trên thông tin được cung cấp hoặc báo không biết), phần ngữ cảnh, và phần câu hỏi của người dùng. Ngữ cảnh (context) được đưa vào bằng cách truy xuất top K tài liệu có điểm tương đồng cao nhất từ vector store, trích xuất nội dung văn bản của chúng, nối lại với nhau bằng các ký tự phân cách (như \n\n---\n\n), và chèn trực tiếp vào chuỗi prompt thông qua định dạng f-string trước khi gửi tới mô hình ngôn ngữ (LLM).
---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
```
42 passed in 0.12s

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Khách hàng có thể trả lại hàng trong vòng 15 ngày. | Người mua được phép hoàn trả sản phẩm trong nửa tháng. | cao | ~ 0.85 | Có |
| 2 | Shopee hoàn tiền cho các đơn hàng bị giao sai. | Cách nấu món phở bò truyền thống ngon tại nhà. | thấp | ~ 0.12 | Có |
| 3 | Sản phẩm này được áp dụng bảo hành miễn phí. | Sản phẩm này KHÔNG được áp dụng bảo hành miễn phí. | thấp | ~ 0.88 | Không |
| 4 | Điện thoại iPhone mới mua bị xước màn hình. | Màn hình của thiết bị di động Apple bị trầy xước. | cao | ~ 0.82 | Có |
| 5 | Trẻ em thường rất thích ăn kẹo bông gòn. | Bác sĩ dùng bông gòn y tế để lau vết thương. | thấp | ~ 0.35 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*
Cặp câu số 3 mang lại kết quả bất ngờ nhất vì dù mang ý nghĩa logic hoàn toàn trái ngược nhau (khẳng định và phủ định), điểm tương tự thực tế lại rất cao. Điều này cho thấy các mô hình embeddings đôi khi vẫn bị phụ thuộc quá nhiều vào sự phân bố và trùng lặp từ vựng ("sản phẩm", "áp dụng", "bảo hành", "miễn phí") trong không gian vector, dẫn đến việc chưa thực sự thấu hiểu được sức mạnh của một từ khóa phủ định làm đảo ngược hoàn toàn ngữ nghĩa của cả câu.
---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn yêu cầu Trả hàng/Hoàn tiền trên Shopee và thời gian gửi trả lại hàng? | Người mua có thể yêu cầu Trả hàng/Hoàn tiền trong vòng 15 ngày... gửi trả hàng lại trong vòng 6 ngày. | ~ 0.85 | Có| 15 ngày để yêu cầu và 6 ngày để gửi trả hàng lại cho người bán. |
| 2 | Lazada: NBH có chịu phí đồng kiểm không, khiếu nại kiện hàng ở đâu? | NBH sẽ không chi trả thêm bất kỳ chi phí nào... liên hệ ngay với Bộ Phận Hỗ trợ NBH PSC. | ~ 0.82 | Có | Không phát sinh chi phí, liên hệ bộ phận PSC để khiếu nại hư hỏng/thiếu hàng. |
| 3 | HACOM: Đổi mới 100% trong bao lâu và ai được bảo hành tại nơi sử dụng? | Áp dụng đổi mới 100% trong 15 ngày đầu... áp dụng cho KH doanh nghiệp Thẻ bảo hành vàng. | ~ 0.79 | Có | Đổi mới trong 15 ngày đầu; áp dụng cho KH doanh nghiệp thẻ vàng cách dưới 20km. |
| 4 | DMX: Thời hạn trả hàng và bằng chứng khi không đồng kiểm? | Gửi yêu cầu trong vòng 15 ngày kể từ khi giao thành công... cung cấp video quay rõ ràng lúc mở. | ~ 0.81 | Có | Trả hàng trong 15 ngày; cần cung cấp video mở hộp rõ ràng, không cắt ghép. |
| 5 | AVAKids: Đồ chơi lỗi kỹ thuật đổi trả trong bao lâu, có bảo hành không? | Đồ chơi được đổi một-một trong 30 ngày kể từ ngày mua... không áp dụng bảo hành. | ~ 0.80 | Có | Được đổi 1-1 trong 30 ngày, nhóm đồ chơi không áp dụng chính sách bảo hành. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*
Điều tôi thấy ấn tượng nhất là sức mạnh của việc kết hợp metadata_filter vào quá trình truy xuất (như ở câu hỏi số 2 về Lazada). Khi gộp chung nhiều tài liệu chính sách của các sàn TMĐT khác nhau, mô hình rất dễ bị nhầm lẫn giữa quy định của sàn này với sàn khác do có chung từ khóa (như "hoàn tiền", "trả hàng"), nhưng nhờ gắn thẻ metadata (platform, audience), hệ thống đã lọc chính xác không gian tìm kiếm, loại bỏ hoàn toàn hiện tượng "râu ông nọ cắm cằm bà kia".
---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | ** 60 / 60** |
