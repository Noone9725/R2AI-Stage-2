import pytest
import pandas as pd
from src.execution.sandbox import PandasSandbox, UnsafeCodeError
from src.execution.validator import diagnose
from src.schemas import ExecutionResult

def test_sandbox_valid_execution():
    df1 = pd.DataFrame({
        "item": ["Doanh thu thuần", "Giá vốn hàng bán"],
        "year": [2022, 2022],
        "value": [1000.0, 600.0]
    })
    sandbox = PandasSandbox()
    code = """
sub = df1[df1['item'].str.contains('Doanh thu', case=False, na=False)]
result = round(float(sub['value'].iloc[0]), 2)
"""
    res = sandbox.run(code, {"df1": df1})
    assert res.success
    assert res.value == 1000.0

def test_sandbox_series_coercion():
    df1 = pd.DataFrame({"value": [123.45]})
    sandbox = PandasSandbox()
    code = "result = df1['value']"
    res = sandbox.run(code, {"df1": df1})
    assert res.success
    assert res.value == 123.45

def test_sandbox_runtime_error_captured():
    df1 = pd.DataFrame({"item": ["A"], "value": [10.0]})
    sandbox = PandasSandbox()
    code = "result = df1['non_existent_column'].iloc[0]"
    res = sandbox.run(code, {"df1": df1})
    assert not res.success
    assert res.error_type == "KeyError"
    
    # Check that diagnose produces actionable hints
    hints = diagnose(res, {"df1": df1})
    assert "non_existent_column" in hints
    assert "item" in hints or "value" in hints

def test_sandbox_blocks_unsafe_code():
    df1 = pd.DataFrame({"value": [10.0]})
    sandbox = PandasSandbox()
    
    # Import forbidden
    code_import = "import os\nresult = 10"
    res_import = sandbox.run(code_import, {"df1": df1})
    assert not res_import.success
    assert "Import" in res_import.error or "import" in res_import.error
    
    # Builtin open forbidden
    code_open = "f = open('test.txt')\nresult = 10"
    res_open = sandbox.run(code_open, {"df1": df1})
    assert not res_open.success


def test_extract_code_filters_natural_language():
    from src.execution.sandbox import extract_code

    # Case 1: Pure Vietnamese natural language -> should return ""
    raw_vn = "Đề xuất sửa chữa:\nĐể sửa lỗi, bạn cần loại bỏ các cột không cần thiết."
    assert extract_code(raw_vn) == ""

    # Case 2: Code embedded with explanations
    raw_mixed = """
Để giải quyết bài toán, ta lọc dữ liệu như sau:
```python
sub = df1[(df1['year'] == 2022) & (df1['item'].str.contains('Doanh thu', case=False, na=False))]
result = float(sub['value'].iloc[0]) if not sub.empty else 0.0
```
Hy vọng đoạn code trên giúp ích cho bạn.
"""
    code = extract_code(raw_mixed)
    assert "sub = df1" in code
    assert "result = float" in code
    assert "Để giải quyết" not in code
    assert "Hy vọng" not in code
