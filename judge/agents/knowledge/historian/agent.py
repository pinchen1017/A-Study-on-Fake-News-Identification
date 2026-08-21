from typing import List
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types
import time
import logging
def delayed_callback(callback_context):
    # 從 context 中提取資訊 (如果需要的話)
    
    ctx = callback_context
    # 根據 JSON，這裡拿到的會是 "advocate_tool_runner1" 之類的名字
    agent_name = getattr(ctx, 'agent_name', 'Unknown')
    delay_seconds = 20 
    print(f"--- [系統訊息] {agent_name} 執行完畢，等待 {delay_seconds} 秒 ---")
    
    # 執行延遲
    time.sleep(delay_seconds)
    
    # 重要：回呼函式通常需要回傳 None 或特定的修改內容，
    # 在延遲需求中，回傳 None 即可讓工作流繼續。
    return None

class TimelineEvent(BaseModel):
    date: str = Field(description="事件發生日期（ISO 8601 或文字描述）")
    description: str = Field(description="事件概述")


class PromotionPattern(BaseModel):
    pattern: str = Field(description="宣傳或推廣模式")
    comparison: str = Field(description="與時間軸事件的比對或證據")


class HistorianOutput(BaseModel):
    timeline: List[TimelineEvent]
    promotion_patterns: List[PromotionPattern]


historian_llm_agent = LlmAgent(
    name="historian_schema_agent",
    model="gemini-2.5-flash",
    instruction=(
        "你是『歷史學者（Historian）』。\n"
        "以下提供 Curator 的整理結果 JSON：{curation}\n"
        "1) 根據資料建立重要事件時間軸。\n"
        "2) 分析是否存在宣傳或推廣模式，並給出比對結果。\n"
        "請僅輸出符合 HistorianOutput schema 的 JSON，不要額外文字。"
    ),
    output_schema=HistorianOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="history",
    after_agent_callback=delayed_callback,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)


historian_agent = SequentialAgent(
    name="historian",
    sub_agents=[historian_llm_agent],
)

