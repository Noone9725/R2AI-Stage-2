from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd

from ..dto.data import Database, Table, Relationship


class ISchemaProfiler(ABC):
    """Interface cho Data Profiler."""
    
    @abstractmethod
    def profile_dataframe(self, df: pd.DataFrame, table_name: str) -> Table:
        """Nhận DataFrame thô, trả về Table với đầy đủ Column metadata."""
        pass


class IRelationshipInferrer(ABC):
    """Interface cho Relationship Inference."""
    
    @abstractmethod
    def infer_relationships(self, tables: dict[str, Table]) -> list[Relationship]:
        """Từ các bảng đã profile, suy luận quan hệ PK-FK."""
        pass


class SchemaManager:
    """
    Quản lý toàn bộ schema.
    Gọi 1 lần khi load dữ liệu, sau đó dùng đi dùng lại.
    """
    
    def __init__(
        self,
        profiler: ISchemaProfiler,
        inferrer: IRelationshipInferrer
    ):
        self.profiler = profiler
        self.inferrer = inferrer
        self.database: Optional[Database] = None
    
    def build(self, raw_data: dict[str, pd.DataFrame]) -> Database:
        """
        Input: {"sales": df1, "customers": df2}
        Output: Database đã profile + inferred relationships.
        """
        # Step 1: Profile từng bảng
        tables = {}
        for name, df in raw_data.items():
            tables[name] = self.profiler.profile_dataframe(df, name)
        
        # Step 2: Suy luận quan hệ
        relationships = self.inferrer.infer_relationships(tables)
        
        self.database = Database(tables=tables, relationships=relationships)
        return self.database
    
    def get_schema_context(self) -> str:
        """Lấy text context cho Prompt."""
        if not self.database:
            raise ValueError("Database chưa được build. Gọi .build() trước.")
        return self.database.get_schema_context()