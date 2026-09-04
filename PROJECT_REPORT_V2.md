# TÀI LIỆU THUYẾT MINH KỸ THUẬT & BÁO CÁO TOÀN DIỆN DỰ ÁN
# R2AI STAGE 2 — FINANCIAL TABLE RETRIEVAL & TEXT-TO-PANDAS (PHIÊN BẢN V2.0)
> **Nhóm thực hiện:** yuiyl  
> **Kho mã nguồn:** [https://github.com/Noone9725/R2AI-Stage-2.git](https://github.com/Noone9725/R2AI-Stage-2.git)

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
3. **Thực thi An toàn trong AST Sandbox & Tự sửa lỗi (Self-Repair Loop):** Cô lập môi trường thực thi, kiểm tra cú pháp AST tĩnh và tự động phản hồi ngữ cảnh lỗi để LLM tự khắc phục tối đa 3–5 lượt.
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
  [https://www.kaggle.com/datasets/anhtu25/r2ai-s2-dataset](https://www.kaggle.com/datasets/anhtu25/r2ai-s2-dataset)

---

## 3. KIẾN TRÚC MÔ HÌNH & SUY LUẬN (MODELS & INFERENCE V2)

Ở phiên bản V2, hệ thống đã được tái cấu trúc toàn diện sang kiến trúc **Quy trình 2 Pha Tách Rời (Decoupled Two-Phase Pipeline)**: Phân lập hoàn toàn giữa Pha Truy hồi (Retrieval) và Pha Sinh mã Suy luận (Generation). Điều này cho phép kiểm thử, tối ưu hóa độc lập từng tầng và ngăn chặn lãng phí tài nguyên tính toán GPU.

```
                            [ Câu hỏi tài chính tiếng Việt ]
                                           │
                                           ▼
  ╔═════════════════════════════════════════════════════════════════════════════╗
  ║ PHA 1: RETRIEVAL & LINKED TABLE SELECTION (CLI: main.py retrieve)           ║
  ║  1. Query Analyzer (Regex + NER tài chính: Tickers, Years, Units, Metrics)   ║
  ║  2. Hard Candidate Filtering (Lọc Metadata: Ticker, Năm, BCTC Mẹ/HN)        ║
  ║  3. Hybrid Retrieval: BM25 (Thưa) + BAAI/bge-m3 FAISS (Dày) + RRF (k=60)     ║
  ║  4. Adaptive Table Selector (Ngân sách đa công ty / đa năm: max_tables=24)   ║
  ║  5. Linked Table Expansion (Tự động kéo đủ chuỗi bảng cùng group_id)       ║
  ║  ► Xuất file: outputs/retrieval/retrieval_results.json                      ║
  ║  ► Đóng gói nộp sớm: outputs/submissions/retrieval_submission.zip           ║
  ╚═════════════════════════════════════════════════════════════════════════════╝
                                           │
                                           ▼
  ╔═════════════════════════════════════════════════════════════════════════════╗
  ║ PHA 2: GENERATION, SANDBOX & PACKAGING (CLI: main.py generate)              ║
  ║  1. Đọc inputs từ outputs/retrieval/retrieval_results.json                 ║
  ║  2. Context Assembly: Nạp DataFrames, Schema Anchoring & Gợi ý Linked Tables║
  ║  3. In-Query Explicit Concat Instruction: Prompt yêu cầu ghép bảng tường    ║
  ║     minh trong Pandas code: df = pd.concat([df1, df2], ignore_index=True)   ║
  ║  4. In-Query Unit Scaling & Rounding: LLM tự tính chuyển đổi đơn vị và      ║
  ║     round(..., 2) trực tiếp trong mã (Triệu/Tỷ/Nghìn/Cổ phiếu/%/Năm)        ║
  ║  5. LLM Text-to-Pandas: Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4)          ║
  ║  6. Python AST Sandbox Exec & Strict Validation                             ║
  ║     ┌──────────────────────────┴──────────────────────────┐                 ║
  ║     ▼ (Lỗi runtime hoặc code cụt "result = ")             ▼ (Thành công)    ║
  ║   [ Self-Repair Loop (max 3-5 lượt) ]                 [ Đồng bộ kết quả ]   ║
  ║   (Feedback Traceback + Schema Anchors)               (answer = exec(query))║
  ║  ► Xuất kết quả hoàn chỉnh: outputs/predictions/questions_pred.json         ║
  ║  ► Đóng gói bài nộp: outputs/submissions/final_submission.zip               ║
  ╚═════════════════════════════════════════════════════════════════════════════╝
```

### 3.1. Mô hình Nhúng & Cơ chế Truy hồi Lai (Hybrid Retrieval V2)
- **Mô hình nhúng:** `BAAI/bge-m3` (Multilingual, 1024 chiều biểu diễn ngữ nghĩa dày đặc).
- **Chỉ số định danh bảng chuẩn (`table_ref`):** `<doc_id>|<line_no>` với `line_no` là số dòng bắt đầu của thẻ `<table>` trong file OCR gốc.
- **Cơ chế Hợp nhất RRF (Reciprocal Rank Fusion):**
  $$\text{Score}_{\text{RRF}}(d) = \frac{0.5}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{0.5}{60 + \text{rank}_{\text{Dense}}(d)}$$
- **Linked Retrieval Expansion:** Khi một bảng được chọn trúng, hệ thống tự động kiểm tra trường `group_id` trong `manifest.jsonl` và kéo thêm các bảng liên đới bị ngắt trang kế tiếp vào tập `relevant_tables` (với ngân sách mở rộng `max_tables: 24`).

### 3.2. Mô hình Ngôn ngữ Lớn & Kỹ thuật Prompting (LLM & Execution V2)
- **Mô hình suy luận chính:** `Qwen/Qwen2.5-Coder-7B-Instruct` (lượng tử hóa 4-bit qua `bitsandbytes`, chỉ tiêu thụ ~5.5 GB VRAM trên Tesla T4).
- **In-Query Explicit Concat:** LLM tự viết lệnh `pd.concat([df1, df2], ignore_index=True)` ngay trong câu lệnh Pandas.
- **In-Query Unit Scaling & Rounding:** LLM tự thực hiện phép chia đơn vị (`/ 1e6`, `/ 1e9`, `* 100.0`) và làm tròn `round(..., 2)` trực tiếp trong code, đảm bảo giá trị `result` do code sinh ra đồng nhất với giá trị `answer` trong JSON.
- **AST Sandbox Execution:** Môi trường thực thi cô lập có kiểm tra cú pháp trừu tượng AST nhằm phát hiện các lệnh nguy hiểm hoặc lỗi cú pháp trước khi chạy.

---

## 4. MÃ NGUỒN & CẤU TRÚC HẠ TẦNG (CODEBASE & INFRASTRUCTURE)

### 4.1. Thông tin Kho mã nguồn
- **GitHub URL:** [https://github.com/Noone9725/R2AI-Stage-2.git](https://github.com/Noone9725/R2AI-Stage-2.git)
- **Nhánh chính:** `main`
- **Môi trường hoạt động:** Windows 10/11, Ubuntu 20.04/22.04, Google Colab, Kaggle Environment.
- **Phiên bản Python:** Python 3.10 – 3.12.

### 4.2. Danh mục Thư viện Phụ thuộc (`requirements.txt`)
- `torch>=2.2.0`, `transformers>=4.40.0`, `accelerate>=0.28.0`, `bitsandbytes>=0.43.0`
- `sentence-transformers>=2.6.0`, `faiss-cpu>=1.8.0` (hoặc `faiss-gpu`)
- `rank-bm25>=0.2.2`, `pandas>=2.1.0`, `numpy>=1.24.0`
- `pyyaml>=6.0`, `pydantic>=2.5.0`, `pytest>=8.0.0`

### 4.3. Các tệp Cấu hình & Scripts Cốt lõi
- `configs/config.yaml`: Cấu hình đường dẫn, checkpointing, LLM backend, và timeout sandbox.
- `configs/retrieval.yaml`: Cấu hình tham số BM25, FAISS, trọng số RRF, và ngân sách `max_tables: 24`.
- `configs/prompts/pandas_gen.txt`: Prompt mẫu sinh code Pandas tích hợp hướng dẫn ghép bảng tường minh và chuẩn hóa đơn vị in-query.
- `configs/prompts/self_repair.txt`: Prompt sửa lỗi vòng lặp phản hồi runtime traceback.
- `scripts/03_run_inference.py`: Script điều phối suy luận hỗ trợ các cờ `--retrieval`, `--generate`, `--all` và khôi phục tiến trình tự động (`--resume`).

---

## 5. HƯỚNG DẪN VẬN HÀNH TRÊN CÁC NỀN TẢNG (OPERATIONAL GUIDE)

### 5.1. Vận hành trên Kaggle Notebook
1. Tải file Notebook Mở file [notebooks/pipeline_runner_kaggle.ipynb](file:///c:/Users/Admin/R2AI-Stage-2/notebooks/pipeline_runner_kaggle.ipynb) lên Kaggle, cấu hình **GPU T4 x2**.
2. Chạy `Mục 1` để clone dự án vào môi trường làm việc và cấu hình môi trường.
3. Nạp dữ liệu từ Kaggle Dataset (`/kaggle/input/r2ai-s2-backup` hoặc `/kaggle/input/s2-backup`) và chạy `Mục 2.A` để nạp vào dataset vào dự án hoặc chạy `Mục 2.B` để xử lý dataset lại (nếu cần).
4. Chạy `Mục 3` để xử lý giai đoạn suy luận, có thể tải file `questions_pred.json` chưa hoàn chỉnh lên `/kaggle/input` để chạy resume (nếu cần).
5. Chạy `Mục 4` để đóng gói kết quả theo chuẩn yêu cầu và Chạy `Mục 5` để đánh giá trên bộ đánh giá file [labels/gold.json](file:///c:/Users/Admin/R2AI-Stage-2/labels/gold.json) được xây dựng để kiểm thử cá nhân (có thể sai sót so với bộ đánh giá của BTC).
6. Nhận kết quả tại suy luận và kết quả đóng gói tại `outputs/predictions/questions_pred.json` và `outputs/submissions/submission.zip` hoặc bản đã sao lưu ngay tại `/kaggle/working/`.

### 5.2. Vận hành trên Google Colab
1. Mở file [notebooks/pipeline_runner_colab.ipynb](file:///c:/Users/Admin/R2AI-Stage-2/notebooks/pipeline_runner_colab.ipynb), chọn cấu hình **GPU T4**, kết nối Google Drive và thực thi tuần tự các cell từ Mục 1 đến Mục 5.
2. Nạp dữ liệu từ Kaggle Dataset (`r2ai-s2-backup` hoặc `s2-backup`) (**Lưu ý:** Cần phiên bản file `zip`) vào thư mục **backup** trên Google Drive và chạy `Mục 2.A` để nạp vào dataset vào dự án hoặc chạy `Mục 2.B` để xử lý dataset lại (nếu cần).
3. Chạy `Mục 3` để xử lý giai đoạn suy luận, có thể tải file `questions_pred.json` chưa hoàn chỉnh lên thư mục **backup** trên Google Drive để chạy resume (nếu cần).
4. Chạy `Mục 4` để đóng gói kết quả theo chuẩn yêu cầu và Chạy `Mục 5` để đánh giá trên bộ đánh giá file [labels/gold.json](file:///c:/Users/Admin/R2AI-Stage-2/labels/gold.json) được xây dựng để kiểm thử cá nhân (có thể sai sót so với bộ đánh giá của BTC).
5. Nhận kết quả tại suy luận và kết quả đóng gói tại `outputs/predictions/questions_pred.json` và `outputs/submissions/submission.zip` hoặc bản đã sao lưu tại thư mục **backup** trên Google Drive.

---

## 6. ĐÁNH GIÁ KỸ THUẬT CHUYÊN SÂU & PHÂN TÍCH KẾT QUẢ THỰC TẾ 1012 CÂU (TECHNICAL BENCHMARK & FAILURE ANALYSIS V2)

### 6.1. Bảng So Sánh Kết Quả Thực Tế V1 vs V2 trên Hệ Thống Chấm Điểm BTC
Dưới đây là bảng đối sánh chi tiết toàn bộ các chỉ số giữa phiên bản V1 ban đầu và phiên bản V2 hiện tại:

| Nhóm Chỉ Số | Tên Chỉ Số (Metric) | Điểm Phiên Bản V1 | Điểm Phiên Bản V2 (`scores.txt`) | Chênh Lệch ($\Delta$) | Trạng Thái Đánh Giá |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Bảng Biểu** | **`TABLES_F2MACRO`** | `0.0000` | **`0.2811`** | **`+0.2811`** | **Đã thoát điểm 0:** Định danh đúng vị trí bảng theo quy chuẩn BTC, tuy nhiên mức điểm còn thấp. |
| *(Table Level)* | `TABLES_PRECISION` | `0.0000` | `0.2145` | `+0.2145` | Tỷ lệ bảng trích xuất trúng đạt ~21.5%. |
| | `TABLES_RECALL` | `0.0000` | `0.3217` | `+0.3217` | Độ phủ bảng đạt ~32.2% (còn bỏ sót gần 68% bảng đúng). |
| | `TABLES_MRR5` | `0.0000` | `0.3548` | `+0.3548` | Bảng liên quan xuất hiện trong Top 5 đạt ~35.5%. |
| **Tài Liệu** | **`DOCS_F2MACRO`** | `0.6688` | **`0.6865`** | **`+0.0177`** | **Tăng nhẹ không đáng kể:** Chưa thể xem là bước cải tiến đột phá. |
| *(Doc Level)* | `DOCS_PRECISION` | `0.5219` | `0.5599` | `+0.0380` | Độ chính xác tài liệu tăng nhẹ gần 4%. |
| | `DOCS_RECALL` | `0.7395` | `0.7490` | `+0.0095` | Độ phủ duy trì ổn định ở ngưỡng ~75%. |
| | `DOCS_MRR5` | `0.7094` | `0.7509` | `+0.0415` | Thứ hạng tài liệu đúng trong Top 5 tăng nhẹ. |
| **Độ Chính Xác** | **`ANSWER_ACCURACY`** | `0.0988` | **`0.0791`** | **`-0.0197`** | **GIẢM NHẸ (CẢNH BÁO XẤU):** Tỷ lệ đáp án đúng tụt từ ~9.9% xuống ~7.9%. |
| *(QA Accuracy)* | **`EXECUTION_ACCURACY`** | `0.0593` | **`0.0514`** | **`-0.0079`** | **GIẢM NHẸ (CẢNH BÁO XẤU):** Tỷ lệ thực thi mã độc lập tụt từ ~5.9% xuống ~5.1%. |

---

### 6.2. Đánh Giá Những Điểm Đã Cải Tiến Thành Công
1. **Giải quyết triệt để lỗi 0 điểm `TABLES_F2MACRO`:**
   - Phiên bản V2 đã chuyển đổi toàn bộ cơ chế đánh số vị trí bảng từ biến đếm tuần tự (`1, 2, 3...`) sang **vị trí dòng bắt đầu thực tế** (`line_index = text[:m.start()].count('\n') + 1`) trong file văn bản OCR `.txt` gốc.
   - Định dạng bảng `<doc_id>|<line_no>` đã tương thích với tập nhãn ground truth của BTC, giúp hệ thống ghi nhận điểm số `TABLES_F2MACRO = 0.2811` thay vì 0 điểm tuyệt đối như ở V1.
2. **Kế thừa metadata và chuẩn hóa dữ liệu bảng độc lập:**
   - Hệ sinh thái bảng đã được chuyển đổi thành các file CSV độc lập (không gộp cứng từ đầu). Bảng ngắt trang được gán cờ liên kết tự động (`is_continuation`, `group_id`, `parent_table_ref`) và kế thừa trọn vẹn thông tin tiêu đề, đơn vị tính từ bảng mốc.
3. **Phân lập quy trình với Kiến trúc 2 Pha Độc lập:**
   - Cơ chế chạy tách biệt giữa `main.py retrieve` và `main.py generate` cho phép nộp bài kiểm thử nhanh Phase 1 để lấy điểm `DOCS_F2` và `TABLES_F2` sớm mà không cần đợi chạy LLM mất quá nhiều thời gian.

---

### 6.3. Phân Tích Chuyên Sâu Các Vấn Đề Tồn Đọng & Nguyên Nhân Gốc Rễ

Mặc dù đã giải quyết được lỗi định dạng bảng của V1, kết quả thực nghiệm V2 bộc lộ những vấn đề kỹ thuật nghiêm trọng cần được đánh giá thẳng thắn:

#### Vấn đề 1: Điểm DOCS Tăng Không Đáng Kể, Điểm TABLES Còn Rất Thấp (Recall ~0.32, Precision ~0.21)
- **Nhận định:** Điểm `DOCS_F2` tăng từ `0.6688` lên `0.6865` (+0.0177) là mức biến thiên rất nhỏ, không thể coi là một bước cải tiến thực chất. Trong khi đó, điểm `TABLES_F2 = 0.2811` với Precision ~21.5% và Recall ~32.2% chứng tỏ bộ máy truy hồi (Retrieval) ở cấp độ bảng biểu vẫn còn hoạt động rất kém. Hệ thống đang bỏ sót tới **gần 68% số bảng cần thiết** để trả lời câu hỏi.
- **Nguyên nhân cốt lõi:**
  1. **Siêu tham số truy hồi chưa được chuẩn hóa tối ưu:** Các trọng số kết hợp RRF ($k=60$), tham số BM25 ($k_1, b$), ngưỡng lọc ứng viên (`candidate_filter`) và cơ chế phân bổ ngân sách chọn bảng (`max_tables`) chưa được tinh chỉnh một cách có hệ thống qua Grid Search / Validation.
  2. **Hạn chế từ Dữ liệu Đầu vào & Metadata (Metadata & Table Titles):** Nếu sau khi tinh chỉnh thông số mà điểm số vẫn không cải thiện, nguyên nhân sâu xa rất có thể xuất phát từ khâu trích xuất siêu dữ liệu OCR:
  Phần `title` của các bảng được sinh ra còn quá sơ sài hoặc bị rỗng, không đủ thông tin ngữ nghĩa để phân biệt giữa các bảng trong cùng một tài liệu (ví dụ: các bảng cùng có tiêu đề chung chung như *"Thuyết minh báo cáo tài chính"* hoặc các bảng thuyết minh con cùng thuộc mục Tài sản dở dang nhưng khác mã số).

#### Vấn đề 2: Điểm `ANSWER_ACCURACY` và `EXECUTION_ACCURACY` Còn rất thấp mà còn giảm nhẹ (Vấn Đề Trọng Yếu)
Điểm `ANSWER_ACCURACY` giảm từ `0.0988` xuống `0.0791` (-0.0197) và `EXECUTION_ACCURACY` giảm từ `0.0593` xuống `0.0514` (-0.0079). Dù ở mức điểm rất thấp, việc còn tiếp tục sụt giảm là một cảnh báo lớn về mặt kiến trúc suy luận:

1. **Hệ quả của Nút thắt Truy hồi ("Garbage In - Garbage Out"):**
   - Do Recall của bảng chỉ đạt ~32%, đối với các câu hỏi phức tạp (câu hỏi đa chỉ tiêu, chuỗi nhiều năm, hoặc so sánh nhóm doanh nghiệp), LLM không nhận được đủ các bảng chứa dữ liệu cần thiết trong context. Việc thiếu dữ kiện đầu vào khiến LLM bắt buộc phải "ảo giác" (`hallucination`), sinh ra các phép truy xuất dòng/cột không tồn tại dẫn đến lỗi runtime.
2. **Mô hình LLM Bị Quá Tải Nhận Thức (Cognitive Overload & Prompt Bloat):**
   - Ở phiên bản V2, ta đã dồn quá nhiều trách nhiệm phức tạp trực tiếp vào Prompt yêu cầu LLM xử lý:
     * Tự phát hiện và viết lệnh ghép bảng tường minh `df = pd.concat([df1, df2], ignore_index=True)`.
     * Tự quy đổi và làm tròn đơn vị tiền tệ (/1e6, /1e9, /1e3) và tỷ lệ phần trăm (* 100.0) ngay trong mã Pandas.
     * Tự phân biệt chỉ tiêu điều kiện lọc so với chỉ tiêu mục tiêu gán cho `result`.
   - Đối với mô hình kích thước 7B (`Qwen2.5-Coder-7B`), việc xử lý đồng thời ngữ cảnh dài (schema của nhiều bảng) cùng hàng loạt chỉ thị ràng buộc phức tạp trong System Prompt đã gây ra hiện tượng **ô nhiễm ngữ cảnh (context pollution)** và quá tải suy luận.
   - Hệ quả là: Mô hình **không thể tạo ra mã gần đúng ngay từ đầu** (tỷ lệ *First-pass Accuracy* cực thấp). Hệ thống phải liên tục kích hoạt Self-Repair Loop, nhưng sau các lượt sửa thì mã sinh ra vẫn mắc lỗi cú pháp hoặc logic.
3. **Các Dạng Lỗi Phổ Biến Được Ghi Nhận Từ Nhật Ký Thực Thi (Execution Logs):**
   - **Lỗi không truy cập dữ liệu bảng thực tế:** Mã sinh ra không thao tác trên dữ liệu từ các DataFrame nạp vào hay không truy cập các bảng mà lại cố tình giả định biến khác hoặc thao tác trên cấu trúc cột/bảng không tồn tại.
   - **Lỗi không thực hiện nối bảng:** Mặc dù ngữ cảnh có `df1, df2` kế tiếp nhau, LLM vẫn bỏ qua không viết `pd.concat`, dẫn đến việc chỉ lọc trên `df1` và trả về tập dữ liệu rỗng.
   - **Lỗi truy xuất thiếu bảng và lệch chỉ số:** Phát sinh hàng loạt lỗi `KeyError`, `IndexError` (`iloc[0] on empty DataFrame`) do truy xuất chỉ tiêu tài chính bị lệch ký tự hoặc không tồn tại trong bảng trích xuất sai.
4. **LỖI NGHIÊM TRỌNG: Mã Cụt Đúng Một Dòng `"result = "` Bị Bỏ Qua Khỏi Cơ Chế Sửa Lỗi:**
   - Qua theo dõi nhật ký thực tế, phát hiện một lỗi đặc biệt nghiêm trọng: Trong nhiều câu hỏi, mô hình LLM chỉ sinh ra đúng một dòng mã cộc lốc duy nhất:
     ```python
     result = 
     ```
     (hoặc `result = ` kết thúc bằng ký tự ngắt dòng mà không có biểu thức hay giá trị tính toán).
   - Tuy nhiên, bộ parser/sandbox hiện tại đã **xử lý sơ hở**: Không kích hoạt ngoại lệ một cách thích đáng hoặc coi đây là giá trị rỗng/None hợp lệ, từ đó ghi nhận kết quả này vào file nộp bài mà **hoàn toàn không kích hoạt vòng lặp tự sửa lỗi (Self-Repair Loop)** để ép LLM phải viết lại!
5. **Lệch Pha Giữa `result` Sau Mã Pandas và `answer` (Xem Xét Lại Cơ Chế Làm Tròn Hậu Xử Lý):**
   - Log ghi nhận một số câu hỏi có kết quả `result` trả về từ việc thực thi mã Pandas bị khác biệt so với giá trị `answer` được lưu trữ trong JSON.
   - Nguyên nhân là cơ chế làm tròn/chia đơn vị tầng ngoài (hậu xử lý) vẫn đang chạy song song hoặc can thiệp đè lên kết quả.
   - Đáng chú ý: Qua đối chiếu số liệu mẫu, giá trị sau khi làm tròn ở `answer` (do bộ quy tắc hậu xử lý tính) trong nhiều trường hợp **lại có vẻ chính xác và logic hơn** so với giá trị `result` do mã Pandas của LLM tự chia/tính toán. Điều này chứng minh LLM 7B chưa thực sự ổn định trong việc tự làm chủ các phép toán chia hệ số phức tạp trong code, và việc giao toàn quyền cho LLM tự chia đơn vị đang tạo ra sai số lớn hơn.

---

## 7. ĐỀ XUẤT CẢI TIẾN & PHƯƠNG ÁN TỐI ƯU HÓA HỆ THỐNG (V3 ROADMAP)

Để đưa điểm số bứt phá ở cả 4 thước đo (`TABLES_F2`, `DOCS_F2`, `ANSWER_ACCURACY`, `EXECUTION_ACCURACY`), hệ thống đề xuất chiến lược tối ưu hóa toàn diện theo 5 giải pháp trọng tâm:

### 7.1. Tối Ưu Hóa Tầng Retrieval & Chuẩn Hóa Siêu Dữ Liệu Bảng (Tạo Mức Sàn Tránh "Garbage In")
- **Chuẩn hóa thông số tìm kiếm (Hyperparameter Optimization):**
  * Thực hiện thử nghiệm lưới (Grid-Search) các bộ tham số của BM25 ($k_1 \in [1.2, 1.8], b \in [0.6, 0.85]$) và điều chỉnh tỷ lệ trọng số RRF giữa BM25 và BGE-M3 Dense (thử nghiệm tỷ lệ $0.6 : 0.4$ hoặc $0.7 : 0.3$ ưu tiên từ khóa chính xác).
  * Đánh giá lại độ rộng `max_tables` theo từng nhóm câu hỏi cụ thể để cân bằng giữa Recall và Precision, tránh đưa quá nhiều bảng rác vào ngữ cảnh gây loãng thông tin.
- **Nâng cấp Chất lượng Metadata & Table Title Extraction:**
  * Cải tiến thuật toán quét ngược và quét xuôi văn bản OCR để trích xuất chính xác tiêu đề bảng (Table Title/Caption) thay vì để rỗng.
  * Phân loại bảng rõ ràng vào Metadata: Đánh dấu phân hệ báo cáo (*Bảng Cân đối kế toán, Báo cáo KQKD, Báo cáo LCTT, Thuyết minh tiền tệ, Thuyết minh tài sản cố định...*).

### 7.2. Tái Cấu Hình & Tối Ưu Hóa Chiến Lược Suy Luận LLM (Prompt Engineering)
- **Thử nghiệm Đa dạng System Prompt (A/B Prompt Testing):**
  * Thiết kế các System Prompt chuyên biệt, cô đọng: Giảm bớt các câu văn giải thích dài dòng gây quá tải context; tập trung vào các quy tắc lập trình cốt lõi và định dạng chuẩn của DataFrame Long-Format (`item`, `year`, `value`).
  * Mục tiêu tiên quyết: **Nâng cao độ chính xác mã ngay từ lượt sinh đầu tiên (*First-pass Accuracy*)**, hạn chế việc phải dựa dẫm vào vòng lặp sửa lỗi.
- **Phân luồng Prompt theo Dạng Câu Hỏi:**
  * *Dạng 1 (Đơn bảng/Trích xuất trực tiếp):* Prompt tối giản, hướng dẫn lọc chính xác `item` và `year`.
  * *Dạng 2 (Đa bảng / Cần nối chuỗi):* Nhấn mạnh cú pháp `pd.concat([df1, df2], ignore_index=True)` và kiểm tra `dropna()`.
  * *Dạng 3 (Chỉ số tài chính phái sinh & So sánh nhiều công ty):* Cung cấp mẫu tính toán cụ thể cho ROE, ROA, D/E, Khả năng thanh toán lãi vay, Tỷ lệ tăng trưởng.

### 7.3. Siết Chặt Quy Tắc Bắt Lỗi AST & Tăng Cường Vòng Lặp Self-Repair
- **Chặn đứng và Phạt nặng Mã Cụt / Mã Rỗng (`result = `):**
  * Thiết lập bộ kiểm tra cú pháp nghiêm ngặt trước khi chạy sandbox: Nếu mã chỉ chứa dòng `result = ` hoặc không tìm thấy phép gán biến `result` có giá trị, hệ thống **bắt buộc coi đây là lỗi cú pháp nghiêm trọng (Critical Syntax Error)**.
  * Lập tức kích hoạt Self-Repair Loop với phản hồi tường minh: *"Lỗi: Mã chưa gán giá trị cho biến result! Bạn phải tính toán và gán kết quả cuối cùng vào biến result."*
- **Tăng Số Lượt Tự Sửa Lỗi (Nâng từ 3 lên 5 lượt):**
  * Khi yêu cầu LLM thực hiện nhiều đầu việc phức tạp, 3 lượt sửa lỗi có thể là chưa đủ để mô hình hội tụ. Nâng lên tối đa 5 lượt kèm phản hồi ngữ cảnh chi tiết (gợi ý danh sách các `item` tương tự có trong bảng khi bị lỗi `KeyError` hoặc `IndexError`).

### 7.4. Đánh Giá & Tinh Chỉnh Cơ Chế Làm Tròn & Đồng Bộ `result` vs `answer`
- Rà soát lại toàn bộ cơ chế làm tròn và chuyển đổi đơn vị. Nếu LLM tự tính toán dễ sinh lỗi hoặc làm tròn sai số, hệ thống sẽ chuẩn hóa theo cơ chế:
  * LLM tập trung trích xuất giá trị số thực tế từ bảng trong mã Pandas.
  * Khâu tính toán chia đơn vị được hỗ trợ bởi các hàm an toàn có sẵn trong namespace của Sandbox (hoặc prompt được chuẩn hóa dạng công thức mẫu bất biến: `result = round(float(...) / scale, 2)`).
  * Bảo đảm tính nhất quán: Giá trị `answer` nộp lên BTC phải chính xác là giá trị sinh ra từ `exec(pandas_query)`.

### 7.5. Thử Nghiệm Benchmark Các Mô Hình Ngôn Ngữ Text-to-Pandas Khác
- Mô hình hiện tại (`Qwen2.5-Coder-7B-Instruct`) có thể chưa đạt độ ổn định cao nhất khi xử lý các truy vấn bảng biểu tài chính tiếng Việt dài và phức tạp.
- Đề xuất thử nghiệm thực nghiệm trên các mô hình khác để tìm kiếm lựa chọn tối ưu:
  * **`Qwen/Qwen2.5-Coder-14B-Instruct` (hoặc bản lượng tử hóa AWQ/GGUF):** Năng lực suy luận và lập trình cao hơn rõ rệt so với bản 7B, khả năng hiểu ràng buộc đa bước tốt hơn mà vẫn có thể triển khai được trên phần cứng GPU hạn chế như Tesla T4 nếu tối ưu VRAM.
  * **`DeepSeek-Coder-V2-Lite-Instruct`:** Mô hình Mixture-of-Experts chuyên dụng cho sinh mã với chi phí tính toán thấp và khả năng tuân thủ cấu trúc tốt.
  * **Đánh giá qua API / vLLM Endpoint:** Thử nghiệm benchmark nhanh một số mẫu trên các mô hình lớn hơn (`Qwen2.5-14B-Instruct`) để xác định trần năng lực (*performance ceiling*) của phương pháp Text-to-Pandas trên bài toán ViFinQA.

---

## 8. LỘ TRÌNH TRIỂN KHAI PHIÊN BẢN V3 (ACTIONABLE NEXT STEPS)

```
[ GIAI ĐOẠN 1: RETRIEVAL & METADATA ] ──► [ GIAI ĐOẠN 2: PROMPT & VALIDATION ] ──► [ GIAI ĐOẠN 3: REPAIR & UNIT SYNC ] ──► [ GIAI ĐOẠN 4: BENCHMARKING ]
  - Tuning BM25, Dense & RRF               - Thử nghiệm A/B System Prompts          - Nâng Self-Repair lên 5 lượt           - Thử nghiệm Qwen 14B / DeepSeek
  - Trích xuất Table Title & BCTC Mẹ/HN    - Bắt lỗi nghiêm ngặt "result = "        - Đồng bộ triệt để result vs answer     - Đánh giá toàn diện 1012 câu
```

1. **Giai đoạn 1 — Tối ưu hóa Tầng Retrieval & Metadata Bảng:**
   - Hoàn thiện module trích xuất tiêu đề bảng và phân loại BCTC Mẹ / Hợp nhất vào `manifest.jsonl`.
   - Chạy grid-search tham số BM25 và trọng số RRF; đo đạc trực tiếp trên tập kiểm thử để đưa `DOCS_F2 > 0.75` và `TABLES_F2 > 0.50`.
2. **Giai đoạn 2 — Tái Cấu Trúc Prompt & Siết Chặt AST Validation:**
   - Xây dựng bộ System Prompts tinh gọn, bổ sung ví dụ Few-Shot đa dạng cho từng nhóm câu hỏi.
   - Thêm bộ lọc cú pháp bắt lỗi mã cụt `result = ` hoặc mã không gán kết quả, bắt buộc kích hoạt Self-Repair ngay lập tức.
3. **Giai đoạn 3 — Nâng Cấp Vòng Lặp Sửa Lỗi & Chuẩn Hóa Đơn Vị:**
   - Mở rộng số lượt Self-Repair lên 5 lượt, làm giàu thông tin traceback và danh sách chỉ tiêu gợi ý.
   - Hoàn thiện cơ chế đồng bộ đảm bảo `answer` trong JSON trùng khớp với kết quả thực thi độc lập của `exec(pandas_query)`.
4. **Giai đoạn 4 — Thử Nghiệm Mô Hình Mới & Đánh Giá Tổng Thể:**
   - Triển khai thử nghiệm mô hình `Qwen2.5-Coder-14B` (hoặc các mô hình lập trình chuyên biệt khác) trên tập 1012 câu hỏi của BTC.
   - Đóng gói bài nộp hoàn thiện và cập nhật báo cáo kỹ thuật phiên bản tiếp theo.
