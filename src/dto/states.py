from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd


# ──────────────────────────────────────────────
# ENUMS
# ──────────────────────────────────────────────

class ActionType(str, Enum):
    """Loại hành động Agent có thể thực hiện."""
    GENERATE_CODE = "generate_code"         # Sinh & chạy Pandas code
    GENERATE_CHART = "generate_chart"       # Sinh & chạy code vẽ biểu đồ
    DIRECT_ANSWER = "direct_answer"         # Trả lời trực tiếp bằng ngôn ngữ tự nhiên


# ──────────────────────────────────────────────
# PLANNING & EXECUTION DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class Action:
    """Một hành động đơn lẻ trong kế hoạch."""
    action_id: str                              # UUID
    action_type: ActionType
    reasoning: str                              # Lý do chọn hành động này
    code: str = ""                              # Code Python (nếu là GENERATE_CODE/CHART)
    input_tables: List[str] = field(default_factory=list)  # Bảng liên quan
    dependencies: List[str] = field(default_factory=list)  # action_id của actions phụ thuộc


@dataclass
class ActionPlan:
    """Kế hoạch tổng thể để trả lời câu hỏi."""
    user_intent: str                            # Diễn giải ý định người dùng
    required_tables: List[str]                  # Các bảng cần dùng
    actions: List[Action]                       # Danh sách hành động
    fallback_answer: str = ""                   # Nếu không làm được thì trả lời gì


@dataclass
class ExecutionResult:
    """Kết quả thực thi một Action."""
    action_id: str
    success: bool
    result_dataframe: Optional[pd.DataFrame] = None  # DF kết quả nếu là query
    result_image_path: Optional[str] = None          # Path ảnh nếu là chart
    error_message: str = ""                          # Thông báo lỗi nếu có
    executed_code: str = ""                          # Code đã chạy thực tế
    execution_time_ms: float = 0.0                   # Thời gian chạy



# ──────────────────────────────────────────────
# AGENT STATE (Dùng trong LangChain)
# ──────────────────────────────────────────────

@dataclass
class AgentState:

    from .io import UserQuery, Output
    from .data import Database

    """State xuyên suốt quá trình xử lý của LangGraph."""
    # Input
    user_query: Optional[UserQuery] = None
    
    # Schema
    database: Optional[Database] = None
    
    # Planning
    action_plan: Optional[ActionPlan] = None
    
    # Execution results (map action_id -> ExecutionResult)
    execution_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    
    # Retry counter để tránh loop vô hạn
    retry_count: int = 0
    max_retries: int = 3
    
    # Final
    final_output: Optional[Output] = None
    
    # Messaging
    messages: List[Any] = field(default_factory=list)  # LangChain messages