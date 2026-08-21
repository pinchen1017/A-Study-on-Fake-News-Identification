from typing import List, Optional
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types
from google.adk.tools.google_search_tool import GoogleSearchTool
from judge.tools.evidence import Evidence
import time
import logging


class CuratorInput(BaseModel):
    query: str = Field(description="搜尋查詢關鍵字或問題")
    top_k: int = Field(default=5, description="回傳前幾筆結果（1~10 建議）")
    site: Optional[str] = Field(
        default=None,
        description="可選的站點過濾，如 'site:reuters.com' 或 'site:gov.tw'",
    )


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

    def to_evidence(
        self,
        claim: str,
        warrant: str,
        method: Optional[str] = None,
        risk: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> Evidence:
        return Evidence(
            source=self.url,
            claim=claim,
            warrant=warrant,
            method=method,
            risk=risk,
            confidence=confidence,
        )


class CuratorOutput(BaseModel):
    query: str
    results: List[SearchResult]
import time
import logging
def delayed_callback(callback_context):
    # 從 context 中提取資訊 (如果需要的話)
    
    ctx = callback_context
    # 根據 JSON，這裡拿到的會是 "advocate_tool_runner1" 之類的名字
    agent_name = getattr(ctx, 'agent_name', 'Unknown')
    delay_seconds = 40 
    print(f"--- [系統訊息] {agent_name} 執行完畢，等待 {delay_seconds} 秒 ---")
    
    # 執行延遲
    time.sleep(delay_seconds)
    
    # 重要：回呼函式通常需要回傳 None 或特定的修改內容，
    # 在延遲需求中，回傳 None 即可讓工作流繼續。
    return None



curator_tool_agent = LlmAgent(
    name="curator_tool_runner",
    model="gemini-2.5-flash",
    instruction=(
        "你是 Curator 的工具執行者：使用 GoogleSearchTool 來取得原始搜尋結果，輸出的內容請精簡，200字以內。\n"
        "請把原始結果（未经 schema 驗證的 JSON）存入 state['curation_raw']。"
    ),
    tools=[],
    output_key="curation_raw",
)


curator_schema_agent = LlmAgent(
    name="curator_schema_validator",
    model="gemini-2.5-flash",
    instruction=(
        "你負責把 state['curation_raw'] 轉為符合 CuratorOutput schema 的 JSON，"
        "僅輸出最終的 JSON（不要多餘文字）。"
    ),
    input_schema=CuratorInput,
    output_schema=CuratorOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="curation",
    after_agent_callback=delayed_callback,
    generate_content_config=types.GenerateContentConfig(temperature=0.4),
)


curator_agent = SequentialAgent(
    name="curator",
    sub_agents=[curator_tool_agent, curator_schema_agent],
    
)

