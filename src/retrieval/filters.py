"""Hard filter tren metadata truoc khi rank.

Tach rieng khoi MetadataStore de logic "noi long dan" nam mot cho:
loc chat -> rong -> noi long -> van rong -> bo loc. F2 nghieng ve recall,
tra ve rong la mat diem tuyet doi, con thua bang thi chi mat mot phan precision.
"""

from __future__ import annotations

from ..schemas import Question
from ..utils.logging import get_logger
from ..vectordb.metadata_store import MetadataStore

log = get_logger(__name__)


class CandidateFilter:
    def __init__(self, store: MetadataStore, year_tolerance: int = 1):
        self.store = store
        self.year_tolerance = year_tolerance

    def build(self, question: Question) -> set[str] | None:
        """Tap table_ref duoc phep xet. None = khong loc gi ca."""
        tickers = question.tickers or None
        years = question.years or None
        rep_types = [question.report_type] if question.report_type else None

        # 1. loc ca ticker + nam + report_type (neu co chi dinh ro)
        if rep_types:
            refs = self.store.filter_refs(
                tickers=tickers, years=years, year_tolerance=self.year_tolerance, report_types=rep_types
            )
            if refs:
                return refs

            if tickers and years:
                refs = self.store.filter_refs(
                    tickers=tickers, years=years, year_tolerance=self.year_tolerance + 2, report_types=rep_types
                )
                if refs:
                    log.debug("Q%d: noi long year_tolerance voi report_type", question.id)
                    return refs

        # 2. loc ca ticker + nam (bao gom ca separate lan consolidated de khong bo sot bang o cac ngan hang/doanh nghiep)
        refs = self.store.filter_refs(
            tickers=tickers, years=years, year_tolerance=self.year_tolerance
        )
        if refs:
            return refs

        # 3. noi long nam (BCTC nam N chua so lieu N-1, N-2)
        if tickers and years:
            refs = self.store.filter_refs(
                tickers=tickers, years=years, year_tolerance=self.year_tolerance + 2
            )
            if refs:
                log.debug("Q%d: noi long year_tolerance", question.id)
                return refs

        # 3. chi loc ticker
        if tickers:
            refs = self.store.filter_refs(tickers=tickers)
            if refs:
                log.debug("Q%d: chi loc theo ticker", question.id)
                return refs

        # 4. chi loc nam
        if years:
            refs = self.store.filter_refs(years=years, year_tolerance=self.year_tolerance)
            if refs:
                log.debug("Q%d: chi loc theo nam", question.id)
                return refs

        log.debug("Q%d: khong loc duoc gi — search toan corpus", question.id)
        return None
