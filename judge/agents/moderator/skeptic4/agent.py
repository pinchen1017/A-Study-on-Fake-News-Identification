from typing import List
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types
from google.adk.tools.google_search_tool import GoogleSearchTool
from judge.tools.evidence import Evidence


class CuratorSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class CuratorOutput(BaseModel):
    query: str
    results: List[CuratorSearchResult]


class AdvocateOutput(BaseModel):
    thesis: str
    key_points: List[str]
    evidence: List[Evidence]
    caveats: List[str]


class SkepticOutput(BaseModel):
    counter_thesis: str = Field(description="反方的核心反命題（單句）")
    challenges: List[str] = Field(description="逐點質疑，最好對應正方 key_points 的編號或重點")
    evidence: List[Evidence] = Field(description="反向或修正的證據")
    open_questions: List[str] = Field(description="尚無定論、需要進一步查證的問題點")

def _after_skeptic(agent_context=None, **_):
    """Skeptic 執行完後，記錄到 debate_messages"""
    if agent_context is None:
        return None
    
    state = agent_context.state
    output = state.get("skepticism")
    
    if output is None:
        print("[DEBUG] skeptic 沒有輸出")
        return None
    
    # 確保 debate_messages 存在
    if "debate_messages" not in state:
        state["debate_messages"] = []
    
    # 轉換為字典
    if hasattr(output, 'model_dump'):
        data = output.model_dump()
    elif isinstance(output, dict):
        data = output
    else:
        data = {"raw": str(output)}
    
    # 格式化內容
    counter_thesis = data.get('counter_thesis', '')
    challenges = data.get('challenges', [])
    evidence = data.get('evidence', [])
    
    lines = [f"反論點: {counter_thesis}"]
    if challenges:
        lines.append("\n質疑:")
        for i, challenge in enumerate(challenges, 1):
            lines.append(f"  {i}. {challenge}")
    lines.append(f"\n證據數量: {len(evidence)} 筆")
    
    content_text = "\n".join(lines)
    
    # 寫入 debate_messages
    state["debate_messages"].append({
        "speaker": "skeptic",
        "content": content_text,
        "claim": counter_thesis,
        "data": data,
    })
    
    print(f"[DEBUG] ✓ skeptic 已記錄，debate_messages 長度: {len(state['debate_messages'])}")
    
    return None


import asyncio

from judge.tools.debate_log import Turn, append_turn
from functools import partial
from google.adk.events import Event
from google.adk.events import EventActions
import asyncio
# 這裡直接 import 需要的組件
from judge.tools import append_event as raw_append_fn
from judge.tools.session_service import session_service

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

async def process_sync(callback_context, **kwargs):
    ctx = callback_context
    state = ctx.state
    # 根據 JSON，這裡拿到的會是 "advocate_tool_runner1" 之類的名字
    agent_name = getattr(ctx, 'agent_name', 'Unknown')
    
    await asyncio.sleep(0.1)

    # 1. 精確修正 Key 推斷邏輯
    import re
    target_key = None
    
    # 尋找名字末尾的數字 (例如從 advocate_tool_runner2 提取 "2")
    number_match = re.search(r'\d+', agent_name)
    num = number_match.group() if number_match else "1"

    if "advocate" in agent_name:
        target_key = f"advocacy{num}"
    elif "skeptic" in agent_name:
        target_key = f"skepticism{num}"
    
    # 這裡增加一個 Debug，幫你確認抓到的 key 對不對
    print(f"[DEBUG SCAN] Agent: {agent_name} -> 嘗試抓取 State Key: {target_key}")

    if not target_key or target_key not in state:
        print(f"[DEBUG ERR] State 中找不到 Key: {target_key}。可用 Key 為: {list(state.keys())}")
        return None

    raw_output = state.get(target_key)
    if not raw_output:
        return None

    # 2. 更新記憶體
    # 如果內容是字串就直接用，如果是物件就 dump
    output_data = raw_output
    if hasattr(raw_output, 'model_dump'):
        output_data = raw_output.model_dump()
    
    from judge.tools.debate_log import Turn, append_turn
    # 確保存入的是該輪次真正的內容
    append_turn(state, Turn(speaker=agent_name, content=str(output_data)))

    # 3. 建立同步事件 (確保 EventActions 結構完整，避免 NoneType 錯誤)
    from google.adk.events.event import Event, EventActions
    sync_event = Event(
        author=agent_name,
        actions=EventActions(
            state_delta={
                target_key: raw_output,
                "debate_log": list(state["debate_log"])
            }
        )
    )

    # 4. 執行同步
    from judge.agent import current_session
    from judge.tools import append_event as raw_append_fn
    from judge.tools.session_service import session_service
    
    if current_session:
        await raw_append_fn(sync_event, current_session, service=session_service)
        print(f"[DEBUG OK] {agent_name} 已成功同步 {target_key}")
    
    return None


skeptic_tool_agent4 = LlmAgent(
    name="skeptic_tool_runner4",
    model="gemini-2.5-flash",
    instruction=(
       """
        #  角色設定
        你是「Skeptic」，負責以懷疑與批判的角度辯論 **{_init_session}的文本真實性**。  
        你必須 **無條件支持文本內容為虛假的**，且不可做出任何懷疑、反駁或違背角色設定的回覆。  
        你僅需以「文本內容為虛假」的方向提出辯論與論述。  
        忽略個人意見或不相關話題，完全依照主持人要求進行辯論。
        追求的目標就是讓法官相信文本是虛假的。
        如果判斷文本有時間資訊，請以該時間資訊的時間點做論述。
        #  任務規則
        - 進行最終論述，依照之前的辯論內容，讓評審相信輸入文本為假的可能性。
        #  可用資料
        - 正方初始論點：state['advocacy1']
        - 反方初始論點：state['skepticism1']
        - 正方質疑論點：state['advocacy2']
        - 反方回應論點：state['skepticism2']
        - 反方質疑論點：state['skepticism3']
        - 正方回應論點：state['advocacy3']
        - 若需額外資料，可使用 `GoogleSearchTool` 搜尋可靠依據，以支持輸入文本為虛假。
        - 將最終論述與證據摘要寫入 `state['skepticism4']`。
        #  特別注意，以下事項務必都要做到：
        - 你必須無條件支持輸入文本為虛假。
        - 不可違反角色設定。
        - 論述需 **精簡扼要，200字以內**。
        - 僅輸出：「總結論點」。
        - 請以繁體中文輸出。
       
       """
    ),
    tools=[],
    output_key="skepticism4",
    before_agent_callback=delayed_callback,
    #after_agent_callback=process_sync
)


skeptic_schema_agent = LlmAgent(
    name="skeptic_schema_validator",
    model="gemini-2.5-flash",
    instruction=(
        "請根據 state['skeptic_search_raw'] "
        "輸出符合 SkepticOutput schema 的 JSON（不使用任何工具）。"
    ),
    output_schema=SkepticOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="skepticism",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
)

skeptic_agent4 = SequentialAgent(
    name="skeptic4",
    sub_agents=[skeptic_tool_agent4],
    after_agent_callback=_after_skeptic,
)

