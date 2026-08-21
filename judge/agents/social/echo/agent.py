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

def echo_agent() -> LlmAgent:
    return LlmAgent(
        name="echo_chamber",
        model="gemini-2.5-flash",
        instruction="你是 Echo Chamber，模擬多個社群群組對當前議題的即時反應，請提供摘要，並且精簡輸出，僅保留重點。",
        output_key="echo_chamber",
        before_agent_callback=delayed_callback,
    )


__all__ = ["echo_agent"]

