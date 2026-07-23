from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict
import pandas as pd


# ──────────────────────────────────────────────
# 2. Database & Schema
# ──────────────────────────────────────────────

@dataclass
class Column:
    """Mô tả một cột trong bảng."""
    name: str
    dtype: str                                   # int64, float64, object, datetime64...
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_to: Dict[str] = None       # Nếu là FK, trỏ tới bảng nào (table: ref_column)
    null_count: int = 0
    sample_values: list[Any] = field(default_factory=list)  # 3-5 giá trị mẫu
    description: str = ""                        # Mô tả tự sinh hoặc từ metadata


@dataclass
class Table:
    """Mô tả một bảng dữ liệu."""
    name: str
    columns: list[Column]
    row_count: int
    dataframe: pd.DataFrame                      # Dữ liệu thật (giữ trong memory)
    
    @property
    def schema_text(self) -> str:
        """Tạo text mô tả schema của bảng này để nhét vào Prompt."""
        lines = [f"Table: {self.name} ({self.row_count} rows)"]
        for col in self.columns:
            pk_fk = ""
            if col.is_primary_key:
                pk_fk += " [PK]"
            if col.is_foreign_key:
                pk_fk += f" [FK → {col.referenced_table}.{col.referenced_column}]"
            lines.append(
                f"  - {col.name} ({col.dtype}){pk_fk}: "
                f"unique={col.unique_count}, nulls={col.null_count}, "
                f"samples={col.sample_values[:3]}"
            )
        return "\n".join(lines)


@dataclass
class Relationship:
    """Quan hệ giữa 2 bảng."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float = 1.0                      # Độ tin cậy của mối quan hệ (0-1)


@dataclass
class Database:
    """Toàn bộ database trong phiên làm việc."""
    tables: dict[str, Table]                     # key = table name
    relationships: list[Relationship] = field(default_factory=list)
    
    def get_schema_context(self) -> str:
        """Tạo toàn bộ schema context để đưa vào System Prompt."""
        parts = []
        for table in self.tables.values():
            parts.append(table.schema_text)
        if self.relationships:
            parts.append("\n--- Relationships ---")
            for rel in self.relationships:
                parts.append(
                    f"  {rel.from_table}.{rel.from_column} → "
                    f"{rel.to_table}.{rel.to_column} "
                    f"(confidence: {rel.confidence:.2f})"
                )
        return "\n\n".join(parts)
