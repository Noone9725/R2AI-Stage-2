import pytest
from src.schemas import Question, RetrievedTable
from src.retrieval.selector import TableSelector

def test_multi_section_coverage_for_derived_metrics():
    # Question asks for ROE which requires both balance_sheet (equity) and income_statement (profit_after_tax)
    q = Question(
        id=1,
        question="ROE năm 2022 của VNM là bao nhiêu %?",
        tickers=["VNM"],
        years=[2022],
        metrics=["roe"],
        needs_derived=True
    )
    
    # Pool has multiple income_statement tables and balance_sheet tables
    pool = [
        RetrievedTable(table_ref="VNM_2022|1", doc_id="VNM_2022", position=1, score=10.0, section="income_statement"),
        RetrievedTable(table_ref="VNM_2022|2", doc_id="VNM_2022", position=2, score=9.5, section="income_statement"),
        RetrievedTable(table_ref="VNM_2022|3", doc_id="VNM_2022", position=3, score=8.0, section="balance_sheet"),
        RetrievedTable(table_ref="VNM_2022|4", doc_id="VNM_2022", position=4, score=7.0, section="cash_flow"),
    ]
    
    selector = TableSelector(strategy="adaptive", max_tables=3)
    res = selector.select(q, pool)
    
    # Selected tables must include at least one balance_sheet and one income_statement
    sections = {t.section for t in res.tables}
    assert "income_statement" in sections
    assert "balance_sheet" in sections


def test_multi_ticker_coverage_for_group_comparison():
    # Question compares multiple companies: GVR, DPM, DCM, PRT
    q = Question(
        id=2,
        question="So sánh biên lợi nhuận của GVR, DPM, DCM, PRT năm 2021?",
        tickers=["GVR", "DPM", "DCM", "PRT"],
        years=[2021],
        metrics=["gross_margin"],
        needs_derived=True,
    )

    # Pool has top tables dominated by GVR and DPM
    pool = [
        RetrievedTable(table_ref="GVR_2021|1", doc_id="GVR_2021", position=1, score=15.0, card="ticker:GVR year:2021"),
        RetrievedTable(table_ref="GVR_2021|2", doc_id="GVR_2021", position=2, score=14.0, card="ticker:GVR year:2021"),
        RetrievedTable(table_ref="DPM_2021|1", doc_id="DPM_2021", position=3, score=13.0, card="ticker:DPM year:2021"),
        RetrievedTable(table_ref="DCM_2021|1", doc_id="DCM_2021", position=4, score=8.0, card="ticker:DCM year:2021"),
        RetrievedTable(table_ref="PRT_2021|1", doc_id="PRT_2021", position=5, score=6.0, card="ticker:PRT year:2021"),
    ]

    selector = TableSelector(strategy="adaptive", max_tables=4)
    res = selector.select(q, pool)

    tickers_covered = {t.doc_id.split("_")[0] for t in res.tables}
    assert "GVR" in tickers_covered
    assert "DPM" in tickers_covered
    assert "DCM" in tickers_covered
    assert "PRT" in tickers_covered
