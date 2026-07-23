from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..dto.data import Database
from ..dto.io import UserQuery, Output
from ..dto.states import ActionPlan, ExecutionResult


# ──────────────────────────────────────────────
# INTERFACE: Output Builder
# ──────────────────────────────────────────────

class IOutputBuilder(ABC):
    """
    Gom tất cả ExecutionResult → Output cuối cùng.
    Format lại câu trả lời, tạo OutputCell cho text/code/table/image.
    """
    
    @abstractmethod
    def build(self,
              user_query: UserQuery,
              action_plan: ActionPlan,
              execution_results: Dict[str, ExecutionResult],
              database: Database) -> Output:
        """Xây dựng Output hoàn chỉnh."""
        pass
    
    @abstractmethod
    def summarize(self, question: str, results: Dict[str, ExecutionResult]) -> str:
        """Tạo câu trả lời tổng quan bằng ngôn ngữ tự nhiên."""
        pass


# ──────────────────────────────────────────────
# INTERFACE: LangChain Agent Factory
# ──────────────────────────────────────────────

class IAgentFactory(ABC):
    """
    Factory để tạo LangChain StateChain.
    Kết nối tất cả components thành một pipeline hoàn chỉnh.
    """
    
    @abstractmethod
    def create_chain(self):
        """Tạo và trả về compiled LangChain StateChain."""
        pass
    
    @abstractmethod
    def run(self, user_query: UserQuery) -> Output:
        """Chạy toàn bộ pipeline, input → output."""
        pass