from langchain.tools import Tool, tool
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

from ..dto.data import Database

class QueryExecutor:
    """
    Thực thi code Pandas trong sandbox.
    Giữ reference tới các DataFrame gốc.
    """
    
    def __init__(self, database: Database):
        self.database = database
    
    @tool
    def query_data(self, code: str) -> str:
        """
        Thực thi code Python để query dữ liệu.
        Các DataFrame có sẵn trong biến: {table_names}
        """
        # Chuẩn bị namespace
        namespace = {}
        for name, table in self.database.tables.items():
            namespace[name] = table.dataframe
        namespace["pd"] = pd
        
        # Thực thi trong sandbox (demo đơn giản)
        try:
            exec(code, namespace)
            # Lấy biến cuối cùng được gán
            result_var = [k for k in namespace.keys() 
                         if k not in list(self.database.tables.keys()) + ["pd"]]
            if result_var:
                result = namespace[result_var[-1]]
                if isinstance(result, pd.DataFrame):
                    return f"SUCCESS\nShape: {result.shape}\n{result.head(10).to_string()}"
                return f"SUCCESS\nResult: {result}"
            return "SUCCESS (no output variable)"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    @tool
    def generate_plot(self, code: str) -> str:
        """
        Tạo biểu đồ từ code Python (dùng matplotlib).
        Trả về base64 của ảnh.
        """
        namespace = {}
        for name, table in self.database.tables.items():
            namespace[name] = table.dataframe
        namespace["pd"] = pd
        namespace["plt"] = plt
        
        try:
            plt.clf()
            exec(code, namespace)
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()
            return f"PLOT_SUCCESS:data:image/png;base64,{img_base64}"
        except Exception as e:
            plt.close()
            return f"PLOT_ERROR: {str(e)}"