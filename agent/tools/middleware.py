"""
中间件系统 - LangChain 0.3.x 适配版本
使用 LangGraph 的节点和边来实现中间件功能
"""
from typing import Callable, Any
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


# 全局上下文存储（用于报告生成标记）
_agent_context = {"report": False}


def get_agent_context():
    """获取 Agent 上下文"""
    return _agent_context


def set_report_flag(value: bool = True):
    """设置报告生成标记"""
    _agent_context["report"] = value


import functools


def monitor_tool_decorator(func: Callable) -> Callable:
    """工具调用监控装饰器（替代 wrap_tool_call）"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        logger.info(f"[tool monitor]执行工具：{tool_name}")
        logger.info(f"[tool monitor]传入参数：{kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.info(f"[tool monitor]工具{tool_name}调用成功")

            # 如果是 fill_context_for_report，设置报告标记
            if tool_name == "fill_context_for_report":
                set_report_flag(True)

            return result
        except Exception as e:
            logger.error(f"工具{tool_name}调用失败，原因：{str(e)}")
            raise e

    return wrapper


def log_before_model_node(state: dict) -> dict:
    """在模型调用前输出日志（替代 before_model）"""
    messages = state.get("messages", [])
    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息。")

    if messages:
        last_message = messages[-1]
        content = getattr(last_message, 'content', str(last_message))
        logger.debug(f"[log_before_model]{type(last_message).__name__} | {content}")

    return state


def get_system_prompt() -> str:
    """获取系统提示词（根据上下文动态切换）"""
    if _agent_context.get("report", False):
        return load_report_prompts()
    return load_system_prompts()


# 导出兼容的接口
monitor_tool = monitor_tool_decorator
log_before_model = log_before_model_node
report_prompt_switch = get_system_prompt
