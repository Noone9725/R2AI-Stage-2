# Coding Note

## Folder structure

```markdown

financial_ai_assistant/
├── models/
│   ├── __init__.py
│   └── dataclasses.py          # Tất cả dataclass ở trên
│
├── interfaces/
│   ├── __init__.py
│   ├── schema_builder.py       # ISchemaBuilder
│   ├── intent_planner.py       # IIntentPlanner
│   ├── code_generator.py       # ICodeGenerator
│   ├── code_executor.py        # ICodeExecutor
│   ├── output_builder.py       # IOutputBuilder
│   └── agent_factory.py        # IAgentFactory
│
├── implementations/
│   ├── __init__.py
│   ├── schema_builder_impl.py       # Dùng ydata-profiling
│   ├── intent_planner_impl.py       # Dùng LLM
│   ├── code_generator_impl.py       # Dùng LLM
│   ├── code_executor_impl.py        # Sandbox Python
│   ├── output_builder_impl.py       # Format output
│   └── agent_graph.py               # LangGraph state machine
│
├── prompts/
│   ├── schema_prompts.py
│   ├── planner_prompts.py
│   ├── code_gen_prompts.py
│   └── output_prompts.py
│
├── main.py                      # Entry point
└── requirements.txt

```
