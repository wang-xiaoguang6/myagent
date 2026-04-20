from dotenv import load_dotenv
load_dotenv()  # 必须在其他导入之前加载环境变量

from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (
    rag_summarize, get_weather, get_user_location, get_user_id,
    get_current_month, fetch_external_data, fill_context_for_report
)
from utils.logger_handler import logger


_agent_context = {"report": False}


def set_report_flag(value: bool = True):
    _agent_context["report"] = value


def get_agent_context():
    return _agent_context


tools = [
    rag_summarize, get_weather, get_user_location, get_user_id,
    get_current_month, fetch_external_data, fill_context_for_report
]

tools_by_name = {t.name: t for t in tools}


class ReactAgent:
    def __init__(self):
        self.model = chat_model
        self.tools = tools
        self.agent = self._build_agent()

    def _build_agent(self):

        def call_model(state: MessagesState):
            messages = state["messages"]

            # 确保系统消息在第一位
            if not any(isinstance(m, SystemMessage) for m in messages):
                if get_agent_context().get("report", False):
                    system_prompt = load_report_prompts()
                else:
                    system_prompt = load_system_prompts()
                messages = [SystemMessage(content=system_prompt)] + messages

            logger.info(f"[call_model]即将调用模型，带有{len(messages)}条消息。")

            model_with_tools = self.model.bind_tools(self.tools)
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        def call_tools(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return state

            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                logger.info(f"[tool monitor]执行工具：{tool_name}")
                logger.info(f"[tool monitor]传入参数：{tool_args}")

                tool_func = tools_by_name.get(tool_name)

                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                        logger.info(f"[tool monitor]工具{tool_name}调用成功")

                        if tool_name == "fill_context_for_report":
                            set_report_flag(True)

                        tool_messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call_id)
                        )
                    except Exception as e:
                        logger.error(f"工具{tool_name}调用失败，原因：{str(e)}")
                        tool_messages.append(
                            ToolMessage(content=f"错误：{str(e)}", tool_call_id=tool_call_id)
                        )

            return {"messages": tool_messages}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def execute_stream(self, query: str):
        set_report_flag(False)

        input_message = HumanMessage(content=query)
        config = {"configurable": {"thread_id": "default"}}

        for event in self.agent.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="values"
        ):
            messages = event.get("messages", [])
            if not messages:
                continue

            latest_message = messages[-1]

            if isinstance(latest_message, AIMessage):
                if latest_message.content and not latest_message.tool_calls:
                    yield latest_message.content.strip()
