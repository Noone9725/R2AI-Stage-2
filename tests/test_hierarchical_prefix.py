import pytest
import pandas as pd
from src.schemas import ExtractedTable
from src.normalization.schema_std import SchemaStandardizer

def test_hierarchical_prefixing():
    # Construct a sample table with section header and sub-items
    rows = [
        ["Chi tiêu", "2023", "2022"],
        ["I. TÀI SẢN NGẮN HẠN", "", ""],
        ["- Tiền và các khoản tương đương tiền", "100.000.000", "80.000.000"],
        ["- Hàng tồn kho", "200.000.000", "150.000.000"],
        ["II. TÀI SẢN DÀI HẠN", "", ""],
        ["- Tài sản cố định hữu hình", "", ""],
        ["  + Nguyên giá", "500.000.000", "400.000.000"],
        ["  + Giá trị hao mòn lũy kế", "(100.000.000)", "(80.000.000)"],
    ]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    table = ExtractedTable(doc_id="TEST_2023", position=1, rows=rows, df=df)
    
    std = SchemaStandardizer()
    res = std.standardize(table, ticker="TEST", fallback_year=2023)
    
    assert not res.empty
    items = res["item"].tolist()
    
    # Verify that sub-items contain parent section prefix
    assert any("TÀI SẢN NGẮN HẠN - - Tiền và các khoản tương đương tiền" in it or "TÀI SẢN NGẮN HẠN" in it for it in items)
    assert any("TÀI SẢN DÀI HẠN" in it for it in items)
