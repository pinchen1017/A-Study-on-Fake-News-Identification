from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent

from judge.agents.social.base import create_social_agent
import time
import logging
def delayed_callback(callback_context):
    # 從 context 中提取資訊 (如果需要的話)
    
    ctx = callback_context
    # 根據 JSON，這裡拿到的會是 "advocate_tool_runner1" 之類的名字
    agent_name = getattr(ctx, 'agent_name', 'Unknown')
    delay_seconds = 10 
    print(f"--- [系統訊息] {agent_name} 執行完畢，等待 {delay_seconds} 秒 ---")
    
    # 執行延遲
    time.sleep(delay_seconds)
    
    # 重要：回呼函式通常需要回傳 None 或特定的修改內容，
    # 在延遲需求中，回傳 None 即可讓工作流繼續。
    return None

INFLUENCER_COUNT = 2


# ==== 社群噪音紀錄 Schema ====
class NoiseLog(BaseModel):
    """社群噪音紀錄"""
    echo_chamber: str = Field(description="各同溫層的反應摘要")
    influencers: list[str] = Field(description="各意見領袖的放大或扭轉訊息")
    disrupter: str = Field(description="干擾者投放的訊息與系統反應")


# ==== 建立平行角色流程 ====
_social_noise_parallel = create_social_agent(
    influencer_count=INFLUENCER_COUNT, include_noise=True
)

# 動態產生 Influencer 輸出段落
_influencer_lines = "\n".join(
    f"- Influencer {i}: {{influencer_{i}}}" for i in range(1, INFLUENCER_COUNT + 1)
)

# 聚合社群噪音輸出為 NoiseLog JSON
_noise_aggregator = LlmAgent(
    name="noise_aggregator",
    model="gemini-2.5-flash",
    instruction=(
        "你是社群噪音紀錄者，請依序讀取以下輸出並統整成 JSON。\n"
        "- Echo Chamber: {echo_chamber}\n"
        f"{_influencer_lines}\n"
        "- Disrupter: {disrupter}\n"
        "僅輸出符合 NoiseLog schema 的 JSON。"
    ),
    output_schema=NoiseLog,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="social_noise",
    before_agent_callback=delayed_callback,
)

# 公開的 social_noise_agent，先平行模擬，再聚合結果
social_noise_agent = SequentialAgent(
    name="social_noise",
    sub_agents=[_social_noise_parallel, _noise_aggregator],
)

