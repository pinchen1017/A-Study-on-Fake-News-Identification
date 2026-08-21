from typing import Optional
from google.adk.agents import LlmAgent
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

def create_influencer_agent(index: Optional[int] = None, output_key: Optional[str] = None) -> LlmAgent:
    name = f"influencer_{index}" if index else "influencer"
    out_key = output_key or ("influencer" if not index else f"influencer_{index}")
    return LlmAgent(
        name=name,
        model="gemini-2.5-flash",
        instruction="你是 Influencer，根據 Echo Chamber 的反應放大或扭轉訊息，請以精簡的方式輸出內容，減少無謂詞語輸出。",
        output_key=out_key,
        before_agent_callback=delayed_callback,
    )


__all__ = ["create_influencer_agent"]

