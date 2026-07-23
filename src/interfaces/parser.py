from abc import ABC, abstractmethod

from ..dto.io import UserQuery

class IIntentParser(ABC):
    """Interface cho Intent Parser."""
    
    @abstractmethod
    def parse(self, raw_prompt: str, schema_context: str) -> UserQuery:
        """
        Phân tích prompt người dùng thành UserQuery có cấu trúc.
        Dùng LLM hoặc rule-based.
        """
        pass