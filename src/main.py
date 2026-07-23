from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.agents import create_react_agent, AgentExecutor

import pandas as pd
from typing import Optional, List

from .interfaces.profiler import SchemaManager
from .interfaces.parser import IIntentParser
from .implements.sandbox import QueryExecutor
from .dto.io import Output, OutputCell, OutputCellType


class FinancialAssistant:
    """
    Lớp chính điều phối toàn bộ hệ thống.
    Đã sửa để tương thích LangChain >= 1.0.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema_manager: SchemaManager,      # SchemaManager
        intent_parser: IIntentParser
    ):
        self.llm = llm
        self.schema_manager = schema_manager
        self.intent_parser = intent_parser

        self.query_executor: Optional[QueryExecutor] = None
        self.agent_executor: Optional[AgentExecutor] = None
        
        self.chat_history: List[BaseMessage] = []  

    def load_data(self, raw_data: dict[str, pd.DataFrame]):
        """
        Load dữ liệu vào hệ thống. Gọi 1 lần khi bắt đầu phiên.
        """
        # Build schema
        self.schema_manager.build(raw_data)

        # Khởi tạo QueryExecutor
        self.query_executor = QueryExecutor(self.schema_manager.database)

        # Tạo tools
        tools = [
            self.query_executor.query_data,
            self.query_executor.generate_plot,
        ]

        # Tạo Agent
        schema_context = self.schema_manager.get_schema_context()

        # ============================================================
        # SỬA: PromptTemplate → ChatPromptTemplate
        # Dùng MessagesPlaceholder cho chat_history và agent_scratchpad
        # ============================================================
        
        system_prompt = f"""You are a financial data analyst AI assistant.

            DATABASE SCHEMA:
            {schema_context}

            You have access to the following tools:

            - query_data: Execute Python code to query the DataFrames. 
            The DataFrames are pre-loaded with the table names shown above.
            Input: a string of Python code.
            Output: the result of executing that code.

            - generate_plot: Execute Python code to generate a matplotlib chart.
            Input: a string of Python code that creates a plot using matplotlib.
            Output: base64 encoded PNG image string prefixed with PLOT_SUCCESS:.

            RULES:
            1. When writing code, use the EXACT table and column names from the schema above.
            2. For multi-table queries, use pd.merge() with the relationships described.
            3. If query_data returns an error, read the error, fix your code, and try again.
            4. After getting data, explain the result to the user in natural language.
            5. For plotting, generate the chart and tell the user it's ready.
            6. Always use the tools to get data, never make up answers.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # SỬA: create_react_agent giờ trả về Runnable, nhận ChatPromptTemplate
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # SỬA: AgentExecutor giờ chỉ cần agent (Runnable) + tools
        # verbose=True để debug, max_iterations giới hạn số vòng lặp
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def run(self, raw_prompt: str) -> Output:
        """
        Nhận prompt người dùng, chạy Agent, trả về Output.
        """
        # Step 1: Parse intent (giữ nguyên)
        schema_context = self.schema_manager.get_schema_context()
        user_query = self.intent_parser.parse(raw_prompt, schema_context)

        # Step 2: Run Agent
        result = self.agent_executor.invoke({
            "input": raw_prompt,
            "chat_history": self.chat_history
        })

        agent_output = result.get("output", "")

        # SỬA: Cập nhật chat_history thủ công
        self.chat_history.append(HumanMessage(content=raw_prompt))
        self.chat_history.append(AIMessage(content=agent_output))

        # Step 3: Parse Agent output thành Output cells
        cells = self._parse_agent_output(agent_output)

        return Output(cells=cells)

    def _parse_agent_output(self, raw_output: str) -> list[OutputCell]:
        """
        Parse raw output từ Agent thành các OutputCell.
        Phát hiện code blocks, base64 images, tables.
        """
        cells = []

        # Phát hiện PLOT_SUCCESS: (ảnh base64)
        if "PLOT_SUCCESS:" in raw_output:
            parts = raw_output.split("PLOT_SUCCESS:", 1)
            # Phần text trước ảnh
            text_before = parts[0].strip()
            if text_before:
                cells.append(OutputCell(
                    type=OutputCellType.TEXT,
                    content=text_before
                ))
            
            # Phần ảnh
            img_data = parts[1].strip()
            # Nếu sau ảnh còn text thì tách ra
            if "\n" in img_data:
                img_lines = img_data.split("\n", 1)
                img_data = img_lines[0].strip()
                text_after = img_lines[1].strip() if len(img_lines) > 1 else ""
            else:
                text_after = ""
            
            cells.append(OutputCell(
                type=OutputCellType.IMAGE,
                content=img_data,
                metadata={"format": "base64_png"}
            ))
            
            if text_after:
                cells.append(OutputCell(
                    type=OutputCellType.TEXT,
                    content=text_after
                ))
        else:
            # Không có ảnh, chỉ có text
            cells.append(OutputCell(
                type=OutputCellType.TEXT,
                content=raw_output
            ))

        # Phát hiện code blocks ```python ... ```
        # (Có thể mở rộng sau)
        
        return cells

    def reset_chat(self):
        """Reset lịch sử chat."""
        self.chat_history = []