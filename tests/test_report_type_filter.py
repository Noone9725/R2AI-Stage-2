import pytest
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.filters import CandidateFilter
from src.vectordb.metadata_store import MetadataStore

def test_extract_report_type():
    qa = QueryAnalyzer()
    
    q1 = qa.analyze(1, "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu?")
    assert q1.report_type == "separate"
    
    q2 = qa.analyze(2, "Doanh thu thuần năm 2022 riêng lẻ của Hòa Phát (HPG) là bao nhiêu?")
    assert q2.report_type == "separate"
    
    q3 = qa.analyze(3, "Lợi nhuận sau thuế hợp nhất năm 2023 của Vinamilk (VNM) là bao nhiêu?")
    assert q3.report_type == "consolidated"
    
    q4 = qa.analyze(4, "Tổng tài sản năm 2020 của VCB là bao nhiêu tỷ đồng?")
    # Default without explicit mention is None (or consolidated preference)
    assert q4.report_type is None
