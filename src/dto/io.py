from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────
# INPUT DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class UserQuery:
    """Input từ người dùng."""
    query_id: str                               # UUID định danh câu hỏi
    prompt: str                                 # Câu hỏi ngôn ngữ tự nhiên
    raw_dataframes: Dict[str, pd.DataFrame]     # DF gốc: {"table_name": df}
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Lịch sử chat

@dataclass
class UserQuery:
    """Câu hỏi từ người dùng sau khi đã parse."""
    raw_prompt: str                              # Prompt gốc
    intent: Optional[str]                                  # Ý định trích xuất (vd: "doanh thu theo tháng")
    entities: List[str] = field(default_factory=list)  # Các thực thể liên quan (tên bảng, cột)


# ──────────────────────────────────────────────
# OUTPUT DATA MODELS
# ──────────────────────────────────────────────

class OutputCellType(Enum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    ERROR = "error"


@dataclass
class OutputCell:
    """Một cell output đơn lẻ."""
    type: OutputCellType
    content: str                                 # Với text: nội dung; code: code string; image: base64 hoặc path
    metadata: dict[str, Any] = field(default_factory=dict)
    # Ví dụ metadata: {"language": "python", "image_path": "/tmp/plot.png", "columns": [...]}

@dataclass
class Output:
    """Output tổng hợp trả về cho người dùng."""
    cells: list[OutputCell] = field(default_factory=list)
    
    @property
    def final_answer(self) -> str:
        """Lấy câu trả lời text cuối cùng."""
        return "\n\n".join(
            [cell.content for cell in self.cells if cell.type == OutputCellType.TEXT]
            )
    
    def to_dict(self) -> dict:
        """Chuyển đổi sang dict để trả về API."""
        result = {"cells": []}
        for cell in self.cells:
            result["cells"].append({
                "type": cell.type.value,
                "content": cell.content,
                "metadata": cell.metadata
            })
        return result

