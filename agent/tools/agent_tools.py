import os
from utils.logger_handler import logger
from langchain_core.tools import StructuredTool

from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


def _rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料"""
    return rag.rag_summarize(query)


rag_summarize = StructuredTool.from_function(
    func=_rag_summarize,
    name="rag_summarize",
    description="从向量存储中检索参考资料"
)


def _get_weather(city: str) -> str:
    """获取指定城市的天气，以消息字符串的形式返回"""
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


get_weather = StructuredTool.from_function(
    func=_get_weather,
    name="get_weather",
    description="获取指定城市的天气，以消息字符串的形式返回"
)


def _get_user_location() -> str:
    """获取用户所在城市的名称，以纯字符串形式返回"""
    return random.choice(["深圳", "合肥", "杭州", "北京"])


get_user_location = StructuredTool.from_function(
    func=_get_user_location,
    name="get_user_location",
    description="获取用户所在城市的名称，以纯字符串形式返回"
)


def _get_user_id() -> str:
    """获取用户的ID，以纯字符串形式返回"""
    return random.choice(user_ids)


get_user_id = StructuredTool.from_function(
    func=_get_user_id,
    name="get_user_id",
    description="获取用户的ID，以纯字符串形式返回"
)


def _get_current_month() -> str:
    """获取当前月份，以纯字符串形式返回"""
    return random.choice(month_arr)


get_current_month = StructuredTool.from_function(
    func=_get_current_month,
    name="get_current_month",
    description="获取当前月份，以纯字符串形式返回"
)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


def _fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串"""
    generate_external_data()

    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


fetch_external_data = StructuredTool.from_function(
    func=_fetch_external_data,
    name="fetch_external_data",
    description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串"
)


def _fill_context_for_report() -> str:
    """调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"""
    return "fill_context_for_report已调用"


fill_context_for_report = StructuredTool.from_function(
    func=_fill_context_for_report,
    name="fill_context_for_report",
    description="调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"
)
