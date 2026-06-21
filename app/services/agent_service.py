import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.services.llm_service import OLLAMA_BASE_URL
from app.services.redis_memory import (
    load_conversation,
    save_conversation_turn,
)
from app.tools.agent_tools import book_interview_tool, search_document_chunks


AGENT_TOOLS = [search_document_chunks, book_interview_tool]


class AgentState(MessagesState):
    top_k: int
    embedding_model: str


@dataclass
class AgentResult:
    answer: str
    retrieved_context_count: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    booking: dict[str, Any] | None = None


class AgentExecutionError(Exception):
    pass


def _system_prompt(top_k: int, embedding_model: str) -> str:
    return f"""
You are a document RAG and interview-booking assistant.

For questions about uploaded documents, always call search_document_chunks
before answering. Use only retrieved document content, and say the uploaded
documents do not contain enough information when retrieval has no answer.
Use top_k={top_k} and embedding_model={embedding_model} for retrieval.

For interview booking, collect and confirm full_name, email, interview_date,
and interview_time. Ask a concise follow-up for any missing required value.
Do not ask whether the candidate should receive an email; backend settings
control that behavior. Call book_interview_tool only after all required
values are known. Never claim a booking succeeded unless the tool confirms it.

Keep responses clear and concise.
""".strip()


@lru_cache(maxsize=4)
def _build_agent_graph(model_name: str):
    model = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        client_kwargs={"timeout": 120},
    ).bind_tools(AGENT_TOOLS)

    def call_model(state: AgentState):
        prompt = SystemMessage(
            content=_system_prompt(
                top_k=state["top_k"],
                embedding_model=state["embedding_model"],
            )
        )
        response = model.invoke([prompt, *state["messages"]])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(AGENT_TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    return str(message.content).strip()


def _extract_tool_data(messages: list[BaseMessage]) -> AgentResult:
    result = AgentResult(answer="")

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        try:
            data = json.loads(_message_text(message))
        except (TypeError, json.JSONDecodeError):
            continue

        if message.name == "search_document_chunks":
            search_results = data.get("results", [])
            result.retrieved_context_count += len(search_results)
            result.sources.extend(
                {
                    "original_filename": item.get("original_filename"),
                    "chunk_index": item.get("chunk_index"),
                    "score": item.get("score"),
                    "chunking_method": item.get("chunking_method"),
                    "embedding_model": item.get("embedding_model"),
                }
                for item in search_results
            )
        elif message.name == "book_interview_tool" and data.get("booking"):
            result.booking = data["booking"]

    return result


def run_agent(
    query: str,
    session_id: str,
    top_k: int,
    embedding_model: str,
    llm_model: str,
) -> AgentResult:
    history = load_conversation(session_id)
    input_messages = [*history, HumanMessage(content=query)]
    graph = _build_agent_graph(llm_model)

    try:
        graph_result = graph.invoke(
            {
                "messages": input_messages,
                "top_k": top_k,
                "embedding_model": embedding_model,
            },
            config={"recursion_limit": 8},
        )
    except Exception as exc:
        raise AgentExecutionError(
            "The agent could not complete the request. Make sure Ollama, "
            "Qdrant, and Redis are running."
        ) from exc

    output_messages = graph_result["messages"]
    final_message = next(
        (
            message
            for message in reversed(output_messages)
            if isinstance(message, AIMessage) and not message.tool_calls
        ),
        None,
    )
    if final_message is None:
        raise AgentExecutionError("The agent did not return a final answer.")

    answer = _message_text(final_message)
    new_messages = output_messages[len(input_messages) :]
    result = _extract_tool_data(new_messages)
    result.answer = answer

    save_conversation_turn(
        session_id=session_id,
        user_message=query,
        assistant_message=answer,
    )
    return result
