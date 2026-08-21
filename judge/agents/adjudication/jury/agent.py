from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent,SequentialAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
import json
from google.genai import types
from judge.tools import flatten_fallacies
import re




class Finding(BaseModel):
    point: str
    refs: List[str] = Field(default_factory=list, description="可附上引用的URL清單")


class JuryOutput(BaseModel):
    verdict: str = Field(description="簡短結論：如 '正方較有說服力' 或 '證據不足'")
    verdict_result: str = Field(description="清楚說明哪一方比較強，回答'正方'或'反方'，回答這兩個的其中一個")
    strengths: List[Finding] = Field(description="哪一方強在哪裡（2~5 條）")
    weaknesses: List[Finding] = Field(description="主要缺陷或風險（2~5 條）")
    flagged_fallacies: List[str] = Field(default_factory=list, description="主持人或評審辨識的邏輯謬誤")
    next_questions: List[str] = Field(default_factory=list, description="尚待澄清/查證的重點問題")



def _ensure_and_flatten_fallacies(callback_context=None, **_):
    if callback_context is None:
        return None
    state = callback_context.state
    # 保底確保存在辯論訊息陣列，避免 KeyError
    msgs = state.get("debate_messages") or []
    state["debate_messages"] = msgs
    state["fallacy_list"] = flatten_fallacies(msgs)
    return None
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

jury_pretty_after = None

def _build_jury_after():
    def _after(agent_context=None, **_):
        if agent_context is None:
            return None
        st = agent_context.state
        out = st.get("jury_result")
        if out is None:
            return None
        try:
            if hasattr(out, "model_dump"):
                data = out.model_dump()
            else:
                data = out
            msg = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            msg = str(out)
        return Event(author="jury", actions=EventActions(message=msg))
    return _after

jury_pretty_after = _build_jury_after()

jury_agent = LlmAgent(
    name="jury",
    model="gemini-2.5-flash",
    instruction=(
        "你是陪審團，請根據完整辯論紀錄與證據，對文本的真實性進行客觀量化評分並給出裁決。\n\n"
        "裁決目標是判斷輸入文本的真實性，請勿脫離判斷真實性的目標。\n\n"
        "如果判斷文本有時間資訊，請以該時間資訊的時間點判斷哪一方勝利。\n\n"
        "【判斷文本】\n"
        "{_init_session}\n\n"
        "【辯論紀錄】\n"
        "- 正方初始論點：state['advocacy1']\n\n"
        "- 反方初始論點：state['skepticism1']\n\n"
        "- 正方質疑論點：state['advocacy2']\n\n"
        "- 反方回應論點：state['skepticism2']\n\n"
        "- 反方質疑論點：state['skepticism3']\n\n"
        "- 正方回應論點：state['advocacy3']\n\n"
        "- 正方總結論點：state['advocacy4']\n\n"
        "- 反方總結論點：state['skepticism4']\n\n"
        "【證據】\n"
        "CURATION(JSON): {curation}\n"
        "SOCIAL_LOG(JSON): {social_log}\n\n"
        "【輸出】\n"
        "嚴格輸出 JSON，必須符合 JuryOutput schema；不要多餘文字。"
    ),
    output_schema=JuryOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="jury_result",
    before_agent_callback=_ensure_and_flatten_fallacies,
    after_agent_callback=delayed_callback,
)




