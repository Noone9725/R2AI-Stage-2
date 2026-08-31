"""Chon bao nhieu bang de dua vao prompt / nop trong relevant_tables.

F2 = 5PR / (4P + R): recall nang gap 4 lan precision. Voi 1 bang dung:
 - tra 1 bang dung        -> P=1.00 R=1.00 F2=1.00
 - tra 3 bang, 1 dung     -> P=0.33 R=1.00 F2=0.71
 - tra 1 bang, sai        -> P=0    R=0    F2=0
Nen bang thu 2-3 gan nhu luon dang gia: mat ~0.29 F2 khi thua, cuu ca
diem khi bang dau sai. Selector vi vay uu tien BAO PHU hon la sac net.
"""

from __future__ import annotations

from ..config import get_settings
from ..schemas import Question, RetrievalResult, RetrievedTable
from ..utils.logging import get_logger

log = get_logger(__name__)


class TableSelector:
    def __init__(
        self,
        strategy: str | None = None,
        min_tables: int | None = None,
        max_tables: int | None = None,
        score_ratio_threshold: float | None = None,
    ):
        cfg = get_settings().retrieval.get("selector", {})
        self.strategy = strategy or cfg.get("strategy", "adaptive")
        self.min_tables = min_tables if min_tables is not None else int(cfg.get("min_tables", 2))
        self.max_tables = max_tables if max_tables is not None else int(cfg.get("max_tables", 15))
        self.ratio = (
            score_ratio_threshold
            if score_ratio_threshold is not None
            else float(cfg.get("score_ratio_threshold", 0.45))
        )

    def select(
        self, question: Question, tables: list[RetrievedTable]
    ) -> RetrievalResult:
        if not tables:
            return RetrievalResult(question_id=question.id, tables=[])

        # Rerank bang theo do khop noi dung cot item/column_label thuc te trong CSV
        reranked = self._rerank_by_content(question, tables)

        if self.strategy == "topk":
            chosen = reranked[: self.max_tables]
        else:
            chosen = self._adaptive(question, reranked)

        log.debug(
            "Q%d: chon %d/%d bang (%s)",
            question.id, len(chosen), len(tables), self.strategy,
        )
        return RetrievalResult(question_id=question.id, tables=chosen)

    # ── content-aware reranking ───────────────────────────

    def _rerank_by_content(
        self, question: Question, tables: list[RetrievedTable], max_candidates: int = 35
    ) -> list[RetrievedTable]:
        """Quet nhanh noi dung cot `item` va `column_label` trong CSV cua top candidate de tai xep hang.
        Bang chua dung cum tu khoa (phrase) hoac chi tieu cua cau hoi se duoc dua len dau.
        """
        import re
        from pathlib import Path
        import pandas as pd

        if not tables or not question.question:
            return tables

        stopwords = {
            "bao", "nhiêu", "tổng", "công", "ty", "mẹ", "tập", "đoàn", "ngân", "hàng",
            "tmcp", "ctcp", "tnhh", "đồng", "triệu", "nghìn", "tỷ", "vnd", "ngày",
            "tháng", "năm", "trong", "theo", "được", "người", "nhóm", "của", "cho",
            "vào", "đến", "các", "những", "nào", "mấy", "là", "và", "khoản", "mục",
            "tại", "kỳ", "đầu", "cuối", "số", "dư", "giá", "trị", "mức", "tính",
            "percent", "phần", "trăm", "%"
        }
        q_raw = question.question.lower()
        q_words = [
            w for w in re.sub(r"[^\w\s]", " ", q_raw).split()
            if len(w) >= 2 and w not in stopwords
        ]
        q_tokens = [w for w in q_raw.split() if w not in stopwords]

        # Bigrams va trigrams tu cau hoi (vi du: "hoat dong kinh doanh", "thuong mai", "quy khen thuong")
        phrases: list[str] = []
        for i in range(len(q_tokens) - 1):
            bg = f"{q_tokens[i]} {q_tokens[i+1]}"
            if len(bg) >= 5:
                phrases.append(bg)
        for i in range(len(q_tokens) - 2):
            tg = f"{q_tokens[i]} {q_tokens[i+1]} {q_tokens[i+2]}"
            if len(tg) >= 8:
                phrases.append(tg)

        candidates = tables[:max_candidates]
        tail = tables[max_candidates:]

        settings = get_settings()
        proc_dir = settings.paths.processed

        scored_candidates: list[tuple[float, float, RetrievedTable]] = []
        for rank, table in enumerate(candidates):
            content_score = 0.0
            items_list: list[str] = []
            csv_file = None
            raw_path = table.csv_path or f"{table.doc_id}_table_{table.position}.csv"
            filename = raw_path.replace("\\", "/").split("/")[-1]

            for candidate_path in (
                proc_dir / filename,
                proc_dir.parent / filename,
                Path("data/processed") / filename,
                Path("data") / filename,
                Path(raw_path),
            ):
                if candidate_path and candidate_path.exists():
                    csv_file = candidate_path
                    break

            if csv_file and csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, nrows=120)
                    if "item" in df.columns:
                        items_list.extend(df["item"].dropna().astype(str).tolist())
                    if "column_label" in df.columns:
                        items_list.extend(df["column_label"].dropna().astype(str).tolist())
                except Exception:
                    pass
            
            # Fallback: neu chua doc duoc CSV tren disk, doc tu Table Card trong RAM
            if not items_list and table.card:
                items_list.append(table.card.lower())

            if items_list:
                full_text = " ".join(items_list).lower()
                for phr in phrases:
                    if phr in full_text:
                        content_score += 6.0  # Tang diem manh cho cum tu dac thu (vd: "thuong mai", "quy khen thuong")
                for kw in set(q_words):
                    if kw in full_text:
                        content_score += 2.0

                # Phan biet thong minh giua So du (CDKT) va Dong tien (LCTT)
                is_balance_query = any(k in q_raw for k in ("số dư", "cuối năm", "đầu năm", "tại ngày", "quỹ", "ngành"))
                is_cashflow_query = any(k in q_raw for k in ("lưu chuyển tiền", "dòng tiền", "tiền thuần từ"))

                is_cashflow_table = "lưu chuyển tiền" in full_text or (table.section == "cash_flow")
                if is_balance_query and not is_cashflow_query and is_cashflow_table:
                    content_score -= 10.0  # Tru diem nang de tuyet doi tranh chon LCTT cho cau hoi so du / nganh
                elif is_cashflow_query and is_cashflow_table:
                    content_score += 6.0  # Uu tien bang LCTT cho cau hoi dong tien

            combined = content_score * 10.0 + table.score
            scored_candidates.append((combined, table.score, table))

        scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        reranked = [item[2] for item in scored_candidates]
        return reranked + tail

    # ── adaptive ──────────────────────────────────────────

    def _adaptive(
        self, question: Question, tables: list[RetrievedTable]
    ) -> list[RetrievedTable]:
        budget = self._budget(question)
        top = tables[0].score
        cutoff = top * self.ratio if top > 0 else float("-inf")

        chosen = [t for t in tables[:budget] if t.score >= cutoff]

        # Dam bao du min_tables
        if len(chosen) < self.min_tables:
            chosen = tables[: min(len(tables), self.min_tables)]

        return self._ensure_coverage(question, chosen, tables, budget)

    def _budget(self, question: Question) -> int:
        """Cau hoi nhieu ticker/nam can nhieu bang hon, cau don can it hon."""
        n_tickers = len(question.tickers)
        n_years = len(question.years)

        if n_tickers >= 2:
            # Multi-company comparison: moi cong ty can 1-2 bang (CDKT / KQKD)
            need = max(n_tickers, n_tickers * min(2, max(1, n_years)))
            return min(self.max_tables, need)
        elif n_years >= 2:
            # Multi-year single company: 2 bang moi nam
            need = max(self.min_tables, n_years * 2)
            return min(self.max_tables, need)
        else:
            # Single company, 1 year -> 2-3 bang (KQKD + CDKT + Thuyet minh)
            need = 3 if question.needs_derived else 2
            return max(self.min_tables, need)

    def _ensure_coverage(
        self,
        question: Question,
        chosen: list[RetrievedTable],
        pool: list[RetrievedTable],
        budget: int,
    ) -> list[RetrievedTable]:
        """Bo sung bang cho ticker/nam chua duoc bao phu."""
        picked = {t.table_ref for t in chosen}

        # 1. Coverage theo ticker (cho cau hoi nhom / so sanh da cong ty)
        if len(question.tickers) >= 2:
            have_tickers: dict[str, int] = {}
            for t in chosen:
                tk = (t.doc_id.split("_")[0] if "_" in t.doc_id else "").upper()
                have_tickers[tk] = have_tickers.get(tk, 0) + 1

            missing_tickers = [tk for tk in question.tickers if tk.upper() not in have_tickers]
            for tk in missing_tickers:
                cand_found = None
                for cand in pool:
                    if cand.table_ref in picked:
                        continue
                    cand_tk = (cand.doc_id.split("_")[0] if "_" in cand.doc_id else "").upper()
                    if cand_tk == tk.upper() or cand.doc_id.upper().startswith(f"{tk.upper()}_"):
                        cand_found = cand
                        break
                if cand_found:
                    if len(chosen) < self.max_tables:
                        chosen.append(cand_found)
                        picked.add(cand_found.table_ref)
                        have_tickers[tk.upper()] = 1
                    else:
                        # Thay the bang thu hai cua ticker da co > 1 bang
                        redundant_idx = -1
                        for idx in reversed(range(len(chosen))):
                            t_tk = (chosen[idx].doc_id.split("_")[0] if "_" in chosen[idx].doc_id else "").upper()
                            if have_tickers.get(t_tk, 0) > 1:
                                redundant_idx = idx
                                have_tickers[t_tk] -= 1
                                break
                        if redundant_idx >= 0:
                            picked.remove(chosen[redundant_idx].table_ref)
                            chosen[redundant_idx] = cand_found
                            picked.add(cand_found.table_ref)
                            have_tickers[tk.upper()] = 1

        # 2. Coverage theo nam
        if question.years and len(chosen) < self.max_tables:
            missing_years = [
                y for y in question.years
                if not any(f"_{y}_" in t.doc_id or str(y) in (t.card or "") for t in chosen)
            ]
            for year in missing_years:
                for cand in pool:
                    if cand.table_ref in picked:
                        continue
                    if f"_{year}_" in cand.doc_id or str(year) in (cand.card or ""):
                        chosen.append(cand)
                        picked.add(cand.table_ref)
                        break
                if len(chosen) >= self.max_tables:
                    break

        # 3. Coverage da bao cao (Multi-Section) cho chi so phai sinh
        # ROE/ROA/ROS/DE can ca Bang can doi (balance_sheet) va Ket qua kinh doanh (income_statement)
        if question.needs_derived and len(chosen) < self.max_tables:
            derived_needs_both = any(m in ("roe", "roa", "debt_to_equity", "current_ratio", "quick_ratio") for m in question.metrics)
            if derived_needs_both:
                have_sections = {t.section for t in chosen if t.section}
                for req_sec in ("income_statement", "balance_sheet"):
                    if req_sec not in have_sections and len(chosen) < self.max_tables:
                        for cand in pool:
                            if cand.table_ref in picked:
                                continue
                            if cand.section == req_sec:
                                chosen.append(cand)
                                picked.add(cand.table_ref)
        # 4. Linked Retrieval: Tu dong keo tron bo chuoi bang noi tiep (cung group_id)
        chosen = self._link_continuation_tables(chosen, pool)

        return chosen[: self.max_tables]

    def _link_continuation_tables(
        self, chosen: list[RetrievedTable], pool: list[RetrievedTable]
    ) -> list[RetrievedTable]:
        """Linked Retrieval: Tu dong keo tron bo chuoi bang noi tiep (group_id)
        cua cac bang da duoc chon vao tap ung vien nop bai.
        """
        picked = {t.table_ref for t in chosen}
        linked_chosen = list(chosen)

        for t in chosen:
            if not t.group_id and not t.parent_table_ref and not t.next_table_ref:
                continue

            for cand in pool:
                if cand.table_ref in picked:
                    continue
                is_match = False
                if t.group_id and cand.group_id and t.group_id == cand.group_id:
                    is_match = True
                elif t.parent_table_ref and cand.table_ref == t.parent_table_ref:
                    is_match = True
                elif cand.parent_table_ref and cand.parent_table_ref == t.table_ref:
                    is_match = True
                elif t.next_table_ref and cand.table_ref == t.next_table_ref:
                    is_match = True
                elif cand.next_table_ref and cand.next_table_ref == t.table_ref:
                    is_match = True

                if is_match and len(linked_chosen) < self.max_tables:
                    linked_chosen.append(cand)
                    picked.add(cand.table_ref)

        return linked_chosen
