# TÀI LIỆU THUYẾT MINH KỸ THUẬT & BÁO CÁO TOÀN DIỆN DỰ ÁN
# R2AI STAGE 2 — FINANCIAL TABLE RETRIEVAL & TEXT-TO-PANDAS
> **Nhóm thực hiện:** yuiyl  
> **Kho mã nguồn:** [https://github.com/Nostagi/R2AI-Stage-2.git](https://github.com/Nostagi/R2AI-Stage-2.git)

---

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

### 1.1. Bối cảnh & Mục tiêu bài toán
Trong lĩnh vực tài chính, việc trích xuất và suy luận số liệu từ các Báo cáo tài chính (BCTC) kiểm toán của các doanh nghiệp niêm yết là một thách thức cực lớn do:
- Tài liệu dài từ 40 đến 120 trang với định dạng văn bản OCR phi cấu trúc hoặc nửa cấu trúc (Semi-structured HTML inline tables).
- Mỗi tài liệu chứa hàng chục đến hàng trăm bảng biểu phức tạp (Bảng Cân đối kế toán, Báo cáo Kết quả hoạt động kinh doanh, Báo cáo Lưu chuyển tiền tệ và hàng loạt bảng Thuyết minh chi tiết).
- Câu hỏi tài chính tiếng Việt đòi hỏi tính toán đa bước (*multi-step reasoning*), tính chỉ số phái sinh (*ROE, ROA, D/E, Vòng quay hàng tồn kho, Khả năng thanh toán lãi vay*), hoặc so sánh nhóm đa công ty qua nhiều năm tài chính.

**Dự án R2AI Stage 2** được xây dựng như một hệ thống toàn diện giải quyết trọn vẹn bài toán **Financial Table Retrieval & Text-to-Pandas** với 4 trụ cột cốt lõi:
1. **Truy hồi Bảng biểu Chính xác (Hybrid Table Retrieval):** Tìm kiếm và xác định đúng bảng chứa số liệu từ kho dữ liệu gồm hơn **119,000 bảng BCTC**.
2. **Sinh mã Suy luận Tự động (Text-to-Pandas via LLM):** Chuyển đổi câu hỏi ngôn ngữ tự nhiên thành mã lệnh thao tác dữ liệu Python/Pandas thực thi được.
3. **Thực thi An toàn trong AST Sandbox & Tự sửa lỗi (Self-Repair Loop):** Cô lập môi trường thực thi, kiểm tra cú pháp AST tĩnh và tự động phản hồi ngữ cảnh lỗi để LLM tự khắc phục tối đa 3 lượt.
4. **Chuẩn hóa Đơn vị & Đóng gói Bài nộp Tự động:** Tự động điều chỉnh hệ số tiền tệ (*Tỷ đồng, Triệu đồng, Nghìn đồng, USD, %*) và đóng gói file ZIP đạt 100% quy chuẩn Ban Tổ Chức (BTC).

---

## 2. NGUỒN DỮ LIỆU & QUY TRÌNH TIỀN XỬ LÝ (DATASET & PREPROCESSING)

### 2.1. Thông tin Nguồn dữ liệu thô (Raw Data)
- **Nguồn gốc:** Tập dữ liệu BCTC OCR tiếng Việt (thuộc kho dữ liệu ViFinQA) bao gồm **1,973 tài liệu BCTC** của các công ty niêm yết trên sàn chứng khoán Việt Nam giai đoạn 2015 – 2024.
- **Đặc điểm tệp dữ liệu thô:** Các tệp `.txt` chứa văn bản trích xuất từ OCR. Trong đó, các bảng số liệu được mã hóa dưới dạng HTML một dòng (`<table><tr><td>...</td></tr></table>`) kèm các điểm phân trang (`===== PAGE N =====`).

### 2.2. Các vấn đề kỹ thuật của Dữ liệu OCR thô
1. **Số dính liền do thuộc tính `rowspan` (Glued Numbers):** Khi một ô dữ liệu gộp 3 dòng, OCR nối chuỗi các số lại với nhau mà không có dấu cách (ví dụ: `<td rowspan="3">963.717.122.052237.314.356.418726.402.765.634</td>`).
2. **Bảng bị bẻ gãy qua trang (Fractured Tables):** Một bảng BCTC kéo dài từ trang $N$ sang trang $N+1$ bị tách thành 2 thẻ `<table>` rời rạc.
3. **Dính chữ và mất dấu do lỗi OCR:** Tiêu đề cột hoặc tên chỉ tiêu bị dính ký tự (ví dụ: `TÀISẢN`, `Vốncổphần`) hoặc biến dạng thanh điệu.
4. **Cấu trúc bảng ngang (Wide Format):** Mỗi năm là một cột riêng biệt, gây khó khăn cho việc viết query lọc năm tổng quát.

### 2.3. Quy trình Trích xuất & Chuẩn hóa (Corpus Pipeline)
Hệ thống đã triển khai module trích xuất `src/extraction/` và chuẩn hóa `src/normalization/`:
- **Tách số dính (Smart Number Splitting):** Áp dụng regex phân tích nhóm số kế toán Việt Nam `\d{1,3}(?:\.\d{3})*` để giải phóng các số bị dính trong `rowspan` và phân phối chính xác về từng dòng.
- **Chuẩn hóa cấu trúc dọc (Long Table Reshaping):** Toàn bộ bảng biểu được chuyển đổi về định dạng **Long Format** đồng nhất gồm 4 cột bắt buộc:
  * `item` (string): Tên chỉ tiêu tài chính đã làm sạch.
  * `year` (integer): Năm tài chính của số liệu.
  * `period` (string): Kỳ báo cáo (`annual`, `closing`, `opening`, `beginning_period`...).
  * `value` (float): Giá trị số thực tế (đã loại bỏ dấu chấm phân tách hàng nghìn).
- **Trích xuất Thẻ mô tả Bảng (Table Card Generation):** Với mỗi bảng, tạo một thẻ tóm tắt (Card) chứa: Mã ticker, Năm, Loại báo cáo (*separate / consolidated*), Phần báo cáo (*CĐKT / KQKD / LCTT / Thuyết minh*), Đơn vị tính và danh sách 10-15 chỉ tiêu cốt lõi.

### 2.4. Kho Dữ liệu đã Xử lý Hoàn thiện
- **Tổng số bảng sau xử lý:** **119,045 bảng CSV** tại `data/processed/`.
- **Tệp chỉ mục:** `data/index/manifest.jsonl`, `data/index/bm25.pkl`, `data/index/faiss.index`.
- **Liên kết tải Dataset đã xử lý sẵn (Kaggle Datasets):**
  * Dataset dạng nén `.zip.bin`: [https://www.kaggle.com/datasets/anhtu25/s2-backup](https://www.kaggle.com/datasets/anhtu25/s2-backup)
  * Dataset thư mục đã giải nén sẵn: [https://www.kaggle.com/datasets/anhtu25/r2ai-s2-backup](https://www.kaggle.com/datasets/anhtu25/r2ai-s2-backup)

---

## 3. KIẾN TRÚC MÔ HÌNH & SUY LUẬN (MODELS & INFERENCE)

```
                       [ Câu hỏi tài chính tiếng Việt ]
                                      │
                                      ▼
                        [ 01. Query Analyzer ]
                    (Trích xuất Tickers, Years, Metrics, Unit)
                                      │
                                      ▼
                   [ 02. Hard Candidate Filtering ]
               (Lọc Metadata: Ticker, Năm, BCTC Mẹ/HN)
                                      │
                                      ▼
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       [ 03A. BM25 Search ]                         [ 03B. Dense Search ]
      (Từ khóa chính xác trên Card)               (BAAI/bge-m3 + FAISS FlatIP)
               └──────────────────────┬──────────────────────┘
                                      ▼
                       [ 04. Reciprocal Rank Fusion ]
                           (Hợp nhất thứ hạng k=60)
                                      │
                                      ▼
                      [ 05. Adaptive Table Selector ]
                    (Cấp ngân sách đa công ty / đa năm)
                                      │
                                      ▼
                     [ 06. Context Assembly & Prompt ]
                    (Schema Anchoring + Few-Shot CoT)
                                      │
                                      ▼
                    [ 07. LLM: Qwen2.5-Coder-7B (4-bit) ]
                              (Sinh mã Pandas)
                                      │
                                      ▼
                     [ 08. Python AST Sandbox Exec ]
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼ (Lỗi thực thi)                                  ▼ (Thành công)
   [ Self-Repair Loop (max 3) ]                     [ 09. Financial Unit Scaler ]
   (Traceback + Item Anchors)                       (Cân bằng Tỷ/Triệu/Nghìn/%)
                                                               │
                                                               ▼
                                                    [ 10. Submission Builder ]
```

### 3.1. Mô hình Nhúng & Cơ chế Truy hồi Lai (Hybrid Retrieval)
- **Mô hình nhúng:** `BAAI/bge-m3` (Multilingual, hỗ trợ độ dài ngữ cảnh 512 tokens, 1024 chiều).
- **Lý do lựa chọn:** BGE-M3 là một trong những mô hình nhúng đa ngữ mạnh nhất cho tiếng Việt hiện nay, có khả năng nắm bắt ngữ nghĩa chuyên sâu của các thuật ngữ kế toán tài chính Việt Nam (VAS).
- **Cơ chế Hợp nhất (RRF - Reciprocal Rank Fusion):**
  $$\text{Score}_{\text{RRF}}(d) = \frac{0.5}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{0.5}{60 + \text{rank}_{\text{Dense}}(d)}$$
  Giúp cân bằng hoàn hảo giữa việc bắt chính xác từ khóa định danh (Ticker, Năm, Mã số) và ngữ nghĩa tương đồng của chỉ tiêu tài chính.

### 3.2. Mô hình Ngôn ngữ Lớn (LLM for Text-to-Pandas)
- **Mô hình suy luận chính:** `Qwen/Qwen2.5-Coder-7B-Instruct` (phiên bản lượng tử hóa **4-bit BitsAndBytes** hoặc **AWQ**).
- **Mô hình dự phòng:** `Qwen/Qwen2.5-14B-Instruct-AWQ` (hoặc API Endpoint vLLM).
- **Lý do lựa chọn:** Dòng Qwen 2.5 Coder được tối ưu hóa đặc biệt cho tác vụ sinh mã lập trình và hiểu cấu trúc dữ liệu. Bản 4-bit chỉ chiếm khoảng **5.5 GB VRAM**, giúp toàn bộ pipeline (LLM + Embedding + FAISS + Pandas) vận hành ổn định trên môi trường thử nghiệm **GPU Tesla T4 16GB** của Kaggle / Google Colab mà không bị tràn bộ nhớ (Out-of-Memory).

---

## 4. MÃ NGUỒN & CẤU TRÚC HẠ TẦNG (CODEBASE & INFRASTRUCTURE)

### 4.1. Thông tin Kho mã nguồn
- **GitHub URL:** [https://github.com/Nostagi/R2AI-Stage-2.git](https://github.com/Nostagi/R2AI-Stage-2.git)
- **Nhánh chính:** `main`
- **Môi trường hoạt động:** Windows 10/11, Ubuntu 20.04/22.04, Google Colab, Kaggle Environment.
- **Phiên bản Python:** Python 3.10 – 3.12.

### 4.2. Danh mục Thư viện Phụ thuộc (`requirements.txt`)
- `torch>=2.2.0`, `transformers>=4.40.0`, `accelerate>=0.28.0`, `bitsandbytes>=0.43.0`
- `sentence-transformers>=2.6.0`, `faiss-cpu>=1.8.0` (hoặc `faiss-gpu`)
- `rank-bm25>=0.2.2`, `pandas>=2.1.0`, `numpy>=1.24.0`
- `pyyaml>=6.0`, `pydantic>=2.5.0`, `pytest>=8.0.0`

### 4.3. Các tệp Cấu hình Cốt lõi
- `configs/config.yaml`: Cấu hình thư mục dữ liệu, checkpointing, LLM backend, và timeout sandbox.
- `configs/retrieval.yaml`: Cấu hình tham số BM25, FAISS, trọng số RRF, và ngân sách `max_tables: 16`.
- `configs/prompts/pandas_gen.txt`: Prompt mẫu sinh code Pandas với kỹ thuật Chain-of-Thought và ví dụ mẫu Few-Shot đa bảng.
- `configs/prompts/self_repair.txt`: Prompt sửa lỗi vòng lặp phản hồi runtime error.

---

## 5. HƯỚNG DẪN VẬN HÀNH TRÊN CÁC NỀN TẢNG (OPERATIONAL GUIDE)

### 5.1. Vận hành trên Kaggle Notebook
1. Tải file Notebook Mở file [notebooks/pipeline_runner_kaggle.ipynb](file:R2AI-Stage-2/notebooks/pipeline_runner_kaggle.ipynb) lên Kaggle, cấu hình **GPU T4 x2**.
2. Chạy `Mục 1` để clone dự án vào môi trường làm việc và cấu hình môi trường.
3. Nạp dữ liệu từ Kaggle Dataset (`/kaggle/input/r2ai-s2-backup` hoặc `/kaggle/input/s2-backup`) và chạy `Mục 2.A` để nạp vào dataset vào dự án hoặc chạy `Mục 2.B` để xử lý dataset lại (nếu cần).
4. Chạy `Mục 3` để xử lý giai đoạn suy luận, có thể tải file `questions_pred.json` chưa hoàn chỉnh lên `/kaggle/input` để chạy resume (nếu cần).
5. Chạy `Mục 4` để đóng gói kết quả theo chuẩn yêu cầu và Chạy `Mục 5` để đánh giá trên bộ đánh giá file [labels/gold.json](file:R2AI-Stage-2/labels/gold.json) được xây dựng để kiểm thử cá nhân (có thể sai sót so với bộ đánh giá của BTC).
6. Nhận kết quả tại suy luận và kết quả đóng gói tại `outputs/predictions/questions_pred.json` và `outputs/submissions/submission.zip` hoặc bản đã sao lưu ngay tại `/kaggle/working/`.
  

### 5.2. Vận hành trên Google Colab
1. Mở file [notebooks/pipeline_runner_colab.ipynb](file:R2AI-Stage-2/notebooks/pipeline_runner_colab.ipynb), chọn cấu hình **GPU T4**. kết nối Google Drive và thực thi tuần tự các cell từ Mục 1 đến Mục 5.
2. Nạp dữ liệu từ Kaggle Dataset (`r2ai-s2-backup` hoặc `s2-backup`) (**Lưu ý:** Cần phiên bản file `zip`) vào thư mục **backup** trên Google Drive và chạy `Mục 2.A` để nạp vào dataset vào dự án hoặc chạy `Mục 2.B` để xử lý dataset lại (nếu cần).
4. Chạy `Mục 3` để xử lý giai đoạn suy luận, có thể tải file `questions_pred.json` chưa hoàn chỉnh lên thư mục **backup** trên Google Drive để chạy resume (nếu cần).
5. Chạy `Mục 4` để đóng gói kết quả theo chuẩn yêu cầu và Chạy `Mục 5` để đánh giá trên bộ đánh giá file [labels/gold.json](file:R2AI-Stage-2/labels/gold.json) được xây dựng để kiểm thử cá nhân (có thể sai sót so với bộ đánh giá của BTC).
6. Nhận kết quả tại suy luận và kết quả đóng gói tại `outputs/predictions/questions_pred.json` và `outputs/submissions/submission.zip` hoặc bản đã sao lưu tại thư mục **backup** trên Google Drive.
---

## 6. ĐÁNH GIÁ KỸ THUẬT CHUYÊN SÂU & PHÂN TÍCH KẾT QUẢ THỰC TẾ 1012 CÂU (TECHNICAL BENCHMARK & FAILURE ANALYSIS)

### 6.1. Kết quả Đánh giá Thực tế từ Hệ thống Chấm điểm của BTC
Trên tập dữ liệu đánh giá 1012 câu hỏi thực tế, hệ thống BTC trả về kết quả như sau:

| Chỉ Số Đánh Giá (Metric) | Điểm Đạt Được | Đánh Giá Sơ Bộ |
| :--- | :---: | :--- |
| **`DOCS_F2MACRO`** | **`0.6688`** | **Khá tốt:** Retrieval ở cấp độ tài liệu (báo cáo) đạt độ phủ và độ chính xác cao. |
| **`DOCS_RECALL`** | **`0.7395`** | Bắt đúng 74% tài liệu BCTC liên quan. |
| **`DOCS_PRECISION`** | **`0.5219`** | Độ chính xác tài liệu ở mức trung bình khá. |
| **`DOCS_MRR5`** | **`0.7094`** | Tài liệu đúng xuất hiện ở vị trí Top 1 - Top 2. |
| **`TABLES_F2MACRO`** | **`0.0000`** | **Lỗi nghiêm trọng:** 0 điểm tuyệt đối do lệch định dạng vị trí bảng. |
| **`TABLES_PRECISION` / `RECALL`** | **`0.0000`** | Không có bảng nào được tính là trùng khớp với nhãn BTC. |
| **`ANSWER_ACCURACY`** | **`0.0988`** | ~9.88% câu hỏi cho ra đáp số chính xác. |
| **`EXECUTION_ACCURACY`** | **`0.0593`** | ~5.93% câu hỏi thực thi mã độc lập thành công trên môi trường của BTC. |

---

### 6.2. Phân tích Nguyên nhân Gốc rễ (Root Cause Analysis)

#### Vấn đề 1: Tại sao `TABLES_F2MACRO = 0.0`?
1. **Lệch quy ước định danh vị trí bảng:**
   - Trong quy chế bài nộp của BTC: `relevant_tables` có định dạng `<id_báo_cáo>|<vị trí bảng trong báo cáo>` (Ví dụ: `AAA_financial_statements_2015_consolidated|350`). Con số `350` này là **chỉ số dòng bắt đầu (line index)** của khối bảng trong file văn bản OCR gốc `.txt`.
   - Trong code tiền xử lý ban đầu (`src/extraction/html_table.py`), hệ thống lại sử dụng **biến đếm thứ tự tuần tự** (`position = 1, 2, 3... 50`).
   - Việc ghép bảng (`stitch_consecutive_tables`) làm gộp 2-3 bảng rời rạc vào bảng đầu tiên, càng làm sai lệch hoàn toàn số thứ tự bảng so với tập ground truth của BTC.
   - Do hệ thống BTC chấm điểm theo so khớp chuỗi chính xác (*Exact Match*), toàn bộ 1012 câu đều bị chấm 0 điểm `TABLES_F2`.
2. **Tại sao bắt buộc phải Rebuild Dataset từ đầu thay vì chạy Script Patch?**
   - Bộ dữ liệu `data/processed/` chứa tới 119,045 file CSV. Việc viết script đổi tên và tách ngược lại các file đã bị `stitch` (gộp cứng) sẽ gặp nghẽn I/O nghiêm trọng và có rủi ro làm mất mát dòng/cột tiêu đề gốc.
   - Rebuild sạch từ 1,973 file raw `.txt` đảm bảo tính toàn vẹn, đúng từng dòng `line_no` và chuẩn hóa cơ chế liên kết bảng kế thừa metadata.

#### Vấn đề 2: Tại sao `EXECUTION_ACCURACY` tụt sâu xuống `0.0593` và `ANSWER_ACCURACY` chỉ đạt `0.0988`?
1. **Lệch pha quy trình Chuẩn hóa Đơn vị giữa Hệ thống nội bộ và Bộ chấm BTC (Bất đối xứng Hậu xử lý):**
   - **Thực tế:** Hệ thống BTC đánh giá `EXECUTION_ACCURACY` bằng cách lấy danh sách file CSV trong `evidence`, nạp thành `df1, df2...`, sau đó chạy trực tiếp `exec(pandas_query)` và so sánh kết quả trả về của code với trường `answer` trong file JSON.
   - **Điểm nghẽn của bản V1:** Trước đây, logic chia đơn vị (*Tỷ đồng, Triệu đồng*) được thực hiện bằng Python logic ở **tầng ngoài** (`AnswerNormalizer.apply()`) **sau khi** chạy xong mã Pandas. Khi đó:
     * `pandas_query` sinh ra trích xuất số thô trong bảng: `result = 208253201298.0` (VND).
     * `answer` trong JSON được chia đơn vị ra: `208253.2` (triệu đồng).
     * Khi BTC chạy lại `exec(pandas_query)`, kết quả trả về là `208253201298.0` $\neq$ `208253.2` $\rightarrow$ **Đánh trượt `EXECUTION_ACCURACY` ngay lập tức!**
   - **Yêu cầu cốt lõi:** Toàn bộ phép tính chuyển đổi đơn vị và làm tròn `round(..., 2)` **BẮT BUỘC PHẢI NẰM NGAY TRONG MÃ PANDAS** do LLM sinh ra.
2. **Cơ chế Ghép bảng cứng (Stitched Tables) từ khâu Corpus gây sai lệch cấu trúc dữ liệu:**
   - Ở phiên bản V1, module trích xuất tự ý gộp các bảng nối tiếp thành 1 file CSV duy nhất (`stitch_consecutive_tables`). Điều này làm cấu trúc bảng nội bộ có thể không khớp với cấu trúc bảng của BTC (có thể làm rời rạc).
   - Khi truy vấn và thực thi mã trên các bảng bị ngắt trang, mã Pandas không tự chứa lệnh ghép bảng tường minh, dẫn đến thiếu dòng dữ liệu hoặc lỗi truy cập trên môi trường kiểm thử của BTC.
3. **Thiếu hỗ trợ cho các loại câu hỏi Phi tiền tệ & trường hợp `unit = null`:**
   - Trong 1012 câu hỏi, có nhiều nhóm câu hỏi đặc thù mà bộ phân tích từ khóa trả về `unit = null`:
     * **Câu hỏi về Số lượng cổ phiếu / Cổ tức (Shares / Quantity / Counts):** Cần đếm số lượng bản ghi hoặc lấy khối lượng cổ phần lưu hành (nghìn/triệu/tỷ cổ phiếu, không chia nhầm hệ số tiền tệ).
     * **Câu hỏi về Tỷ lệ % / Số lần (Percentages, Ratios, ROE, ROS, Tỷ trọng, Tốc độ tăng trưởng):** Cần nhân $100.0$ và làm tròn 2 chữ số thập phân, tránh chia nhầm $10^6, 10^9$.
     * **Câu hỏi về Năm (Year Index: "Vào năm nào...", "Năm có lợi nhuận cao nhất"):** Cần trích xuất giá trị cột `year` trả về số thực (ví dụ `2019.0`), tránh làm tròn dạng tiền tệ.
     * **Câu hỏi So sánh / Trung vị nhóm:** Cần tính `np.median()`, tính tỷ trọng nhóm và đếm điều kiện.
   - **Lỗi nhầm lẫn mục tiêu `result` và điều kiện lọc:** LLM chưa phân biệt được rõ ràng giữa **"chỉ tiêu điều kiện để lọc/so sánh"** (ví dụ: tìm năm có D/E cao nhất) và **"chỉ tiêu mục tiêu cần gán cho result"** (ví dụ: lấy khả năng thanh toán lãi vay của năm đó), dẫn đến việc gán nhầm giá trị điều kiện vào kết quả cuối cùng.
   - Khi bảng có `unit = null` hoặc câu hỏi không nhận diện được đơn vị, LLM bị mất phương hướng trong việc xác định đơn vị gốc và làm tròn số liệu.
   - LLM cần được trang bị bộ hướng dẫn Prompting tường minh và các ví dụ Few-Shot bao phủ toàn bộ các dạng câu hỏi này.

---

## 7. THỐNG NHẤT PHƯƠNG PHÁP & THIẾT KẾ KIẾN TRÚC CẢI TIẾN TIẾP THEO

Để giải quyết triệt để toàn bộ các nguyên nhân gốc rễ trên, hệ thống thống nhất triển khai 5 trụ cột cải tiến:

### 7.1. Rebuild Dataset Chuẩn Hóa & Kế Thừa Thông Tin
- **Tính đúng `line_index`:** Xác định vị trí bảng bằng số dòng bắt đầu của thẻ `<table>` trong file OCR gốc: `line_no = text[:m.start()].count('\n') + 1`. Định danh bảng chuẩn: `<doc_id>|<line_no>`.
- **Không ghép cứng từ đầu:** Mỗi thẻ `<table>` lưu thành 1 file CSV độc lập dạng `data/{doc_id}_table_L{line_no}.csv`.
- **Gắn cờ liên kết & Kế thừa tự động:** Gắn các trường `is_continuation`, `parent_table_ref`, `group_id` trong `manifest.jsonl`. Các bảng nối tiếp tự động kế thừa `section`, `unit`, `title`, và tên cột từ bảng mốc.
### 7.2. Linked Retrieval & Nâng Ngân Sách `max_tables: 24`
- **Linked Tables Retrieval:** Khi câu hỏi khớp với một bảng bất kỳ trong chuỗi, bộ chọn bảng tự động kéo toàn bộ các bảng trong cùng `group_id` vào danh sách `relevant_tables` nộp bài.
- **Nâng `max_tables: 24`:** Phân bổ ngân sách động để bao phủ đầy đủ các câu hỏi nhóm 5–7 công ty (mỗi công ty có thể có 2 bảng liên kết) và chuỗi 5 năm mà không bị cắt tỉa bảng.
### 7.3. In-Query Explicit Concat trong `pandas_query`
- Trong prompt, hệ thống gắn cờ rõ ràng để LLM nhận diện các bảng cần gộp và **TỰ VIẾT LỆNH GHÉP TƯỜNG MINH** trong mã Pandas:
  ```python
  df = pd.concat([df1, df2], ignore_index=True)
  sub = df[df['item'].str.contains('...', case=False, na=False)]
  result = round(float(sub['value'].iloc[0]) / 1e6, 2)
  ```
- Đảm bảo khi nạp `df1, df2` từ `evidence` và chạy `exec(pandas_query)`, bảng sẽ được ghép đầy đủ ngay trên môi trường của BTC.
### 7.4. In-Query Unit Scaling & Rounding Toàn Diện
- **Toàn bộ logic chuyển đổi đơn vị và làm tròn `round(..., 2)` bắt buộc nằm trong `pandas_query`:**
  * **Tiền tệ (Triệu/Tỷ/Nghìn đồng/VND):** Tự chia $10^6, 10^9, 10^3$ tương ứng và `round(..., 2)`.
  * **Cổ phiếu / Cổ tức (Nghìn/Triệu/Tỷ cổ phiếu):** Chia hệ số cổ phiếu tương ứng hoặc giữ nguyên nếu hỏi số lượng cổ phiếu thô.
  * **Tỷ lệ % / Tăng trưởng / Tỷ suất:** Nhân $100.0$ và làm tròn `round(..., 2)` (Với các câu tính toán ra, giữ nguyên giá trị với các câu trích xuất từ bảng).
  * **Năm (Year Index):** Trả về số thực của năm (ví dụ: `result = float(max_year)` $\rightarrow 2019.0$).
  * **Xử lý `unit = null`:** Prompt hướng dẫn LLM kiểm tra ngữ cảnh câu hỏi, độ lớn số liệu và phân biệt rõ ràng giữa "chỉ tiêu điều kiện lọc" với "chỉ tiêu mục tiêu của `result`".
### 7.5. Quy trình Suy Luận 2 Pha Tách Rời (Decoupled Two-Phase Pipeline)
Để tối ưu hóa thời gian chạy, kiểm soát chi phí GPU và phân lập lỗi chính xác, hệ thống đề xuất kiến trúc **Tách rời 2 Giai đoạn Độc lập**:

```
[ data/questions/questions.jsonl ]
        │
        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ GIAI ĐOẠN 1: RETRIEVAL & TABLE SELECTION (Chạy độc lập)               ║
║ 1. Query Analyzer (Regex + NER tài chính nhẹ)                         ║
║ 2. Hybrid Retriever (BM25 + BGE-M3 Dense)                             ║
║ 3. Adaptive Table Selector (Ngân sách đa công ty / đa năm)            ║
║ 4. Xuất file trung gian: outputs/retrieval/retrieval_results.json     ║
╚═══════════════════════════════════════════════════════════════════════╝
        │
        ├──► Tạo retrieval_submission.zip nộp lên để [ ĐÁNH GIÁ ĐỘC LẬP: DOCS_F2 & TABLES_F2]
        │
        ▼
╔═════════════════════════════════════════════════════════════════════════════════════════════╗
║ GIAI ĐOẠN 2: GENERATION, EXECUTION & PACKAGING (Chạy LLM)                                   ║
║ 1. Đọc inputs từ outputs/retrieval/retrieval_results.json                                   ║
║ 2. Context Assembly: Nạp DataFrames và Schema Anchoring + Gợi ý liên kết bảng cần ghép      ║
║ 3. In-Query Unit & Rounding Prompt: LLM tự tính chuyển đổi đơn vị và round(..., 2)          ║
║ 4. LLM Sinh mã Pandas (Text-to-Pandas)                                                      ║
║ 4. Thực thi Sandbox AST + Self-Repair Loop (max 3 lượt)                                     ║
║ 5. Đồng bộ giá trị answer trong JSON trùng khớp exec(pandas_query)                          ║
║ 6. Xuất final_results.json và đóng gói: outputs/submissions/final_submission.zip            ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
```
- **Pha 1 — Retrieval:**
  - Lệnh: `python main.py retrieve --questions data/questions/questions.jsonl`
  - Đóng gói kiểm thử sớm: `python main.py package --pred outputs/retrieval/retrieval_results.json --out outputs/submissions/retrieval_submission.zip`
- **Pha 2 — Generation & Execution:**
  - Lệnh: `python main.py generate --retrieval outputs/retrieval/retrieval_results.json --out outputs/predictions/final_results.json`
  - Hỗ trợ khôi phục / nạp dữ liệu từ `/MyDrive/backup/` (Colab) và `/kaggle/input/<dataset>/` (Kaggle).
  - Đóng gói hoàn chỉnh: `python main.py package --pred outputs/predictions/final_results.json --out outputs/submissions/final_submission.zip`

---

## 8. LỘ TRÌNH THỰC THI (ACTIONABLE ROADMAP)

```
[ BƯỚC 1: REBUILD DATASET ] ──► [ BƯỚC 2: BUILD INDEX & LINKING ] ──► [ BƯỚC 3: DECOUPLED PIPELINE ]
  - Trích xuất line_index         - Build BM25 & BGE-M3 Dense           - CLI: retrieve & generate
  - Gắn cờ is_continuation        - Linked Table Chaining               - In-Query Unit & Concat
  - Tạo 119k CSV không gộp        - max_tables: 24                      - Đóng gói kiểm thử 2 pha
```
1. **Bước 1 — Rebuild Dataset Chuẩn Hóa:**
   - Cập nhật `src/extraction/html_table.py` tính chỉ số dòng `line_no = text[:m.start()].count('\n') + 1`.
   - Không gộp file cứng; mỗi thẻ `<table>` lưu thành 1 file CSV riêng dạng `data/{doc_id}_table_L{line_no}.csv`.
   - Gắn cờ `is_continuation`, `parent_table_ref`, `group_id` trong `manifest.jsonl` và tự động kế thừa metadata.
2. **Bước 2 — Build lại Chỉ mục & Linked Retrieval:**
   - Chạy lại `python main.py index` để tạo BM25 và FAISS Vector BGE-M3.
   - Nâng `max_tables: 24` trong `configs/retrieval.yaml`.
   - Tích hợp logic Linked Retrieval: Tự động kéo đủ toàn bộ các bảng trong cùng `group_id` khi câu hỏi khớp với một bảng trong nhóm.
3. **Bước 3 — Triển khai Decoupled Pipeline & Prompting Đa Dạng:**
   - Tách hai lệnh độc lập `main.py retrieve` và `main.py generate`.
   - Cập nhật `pandas_gen.txt` với các Few-Shot mẫu: In-Query Concat, In-Query Scaling (Tỷ, Triệu, Nghìn, Cổ phiếu, Năm, Tỷ lệ %).
4. **Bước 4 — Kiểm thử & Đánh giá Toàn Diện:**
   - Đánh giá Giai đoạn 1: Nộp `retrieval_submission.zip` để xác nhận `DOCS_F2` và `TABLES_F2`.
   - Đánh giá Giai đoạn 2: Chạy LLM suy luận, kiểm tra độ đồng bộ `answer == exec(pandas_query)` và nộp `final_submission.zip` hoàn chỉnh để tối đa hóa `ANSWER_ACCURACY` và `EXECUTION_ACCURACY`
