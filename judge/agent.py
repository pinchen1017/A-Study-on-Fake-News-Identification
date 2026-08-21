# judge/agent.py
from __future__ import annotations
import google.adk.tools.google_search_tool as gst
from google.adk.tools.google_search_tool import GoogleSearchTool

from functools import partial
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions.session import Session
from google.genai import types

from judge.tools.session_service import session_service

from judge.agents.llm.agent import llm_agent,fact_check_tool_agent  # ✅ 修正: 正確導入

from judge.agents.knowledge.curator.agent import curator_agent,curator_tool_agent
from judge.agents.moderator.devil.agent import devil_agent
from judge.agents.adjudication.agent import adjudication_agent
from judge.agents.adjudication.evidence.agent import evidence_agent,_evidence_tool_agent
from judge.agents.adjudication.jury import jury_agent
from judge.agents.adjudication.synthesizer.agent import synthesizer_agent
from judge.agents.knowledge.historian import historian_agent
from judge.agents.moderator.agent import orchestrator_agent
from judge.agents.moderator.tools import log_tool_output

from judge.agents.social.agent import social_summary_agent
from judge.agents.social.noise.agent import social_noise_agent

from judge.agents.classifier.agent import classifier_agent
from judge.agents.weight.agent import weight_agent
from judge.agents.debatelog.agent import debate_agent
from judge.tools import _before_init_session, append_event, make_record_callback
from judge.agents.moderator.advocate.agent import advocate_agent1,advocate_tool_agent1
from judge.agents.moderator.advocate2.agent import advocate_agent2,advocate_tool_agent2
from judge.agents.moderator.advocate3.agent import advocate_agent3,advocate_tool_agent3
from judge.agents.moderator.advocate4.agent import advocate_agent4,advocate_tool_agent4
from judge.agents.moderator.skeptic.agent import skeptic_agent1,skeptic_tool_agent1
from judge.agents.moderator.skeptic2.agent import skeptic_agent2,skeptic_tool_agent2
from judge.agents.moderator.skeptic3.agent import skeptic_agent3,skeptic_tool_agent3
from judge.agents.moderator.skeptic4.agent import skeptic_agent4,skeptic_tool_agent4

def create_session(state: dict | None = None) -> Session:
    
    default_state = {
        "debate_messages": [],
        "agents": [],
        "debate_turn": 0,
        "current_round": 1,
        "current_phase": "statement",
    }
    
    if state:
        default_state.update(state)

    return session_service.create_session_sync(
        app_name="agent_judge",
        user_id="user",
        state=default_state,
    )
import time
import logging
def delayed_callback(callback_context):
    # 從 context 中提取資訊 (如果需要的話)
    
    ctx = callback_context
    # 根據 JSON，這裡拿到的會是 "advocate_tool_runner1" 之類的名字
    agent_name = getattr(ctx, 'agent_name', 'Unknown')
    delay_seconds = 30 
    print(f"--- [系統訊息] {agent_name} 執行完畢，等待 {delay_seconds} 秒 ---")
    
    # 執行延遲
    time.sleep(delay_seconds)
    
    # 重要：回呼函式通常需要回傳 None 或特定的修改內容，
    # 在延遲需求中，回傳 None 即可讓工作流繼續。
    return None



current_session = None

def bind_session(session: Session) -> None:
    global current_session
    current_session = session  # 存入全局變數

    # 正反方代理人清單
    debate_agents = [
        advocate_agent1, advocate_agent2, advocate_agent3, advocate_agent4,
        skeptic_agent1, skeptic_agent2, skeptic_agent3, skeptic_agent4
    ]

    for agent in debate_agents:
        # ✅ 最簡單的綁定：直接丟函式進去
        # 因為 process_sync 現在只吃一個 callback_context 參數，ADK 會很滿意
        agent.after_agent_callback = process_sync

    print("[SYSTEM] 所有辯論代理人已完成自動化同步綁定。")


# 2. 實例化這個空殼工具
shared_tool = GoogleSearchTool()


# 2. 把所有需要用到搜尋的 agent 丟進這個清單
agents_need_search = [
    _evidence_tool_agent,
    curator_tool_agent,
    fact_check_tool_agent,
    advocate_tool_agent1,
    advocate_tool_agent2,
    advocate_tool_agent3,
    advocate_tool_agent4,
    skeptic_tool_agent1,
    skeptic_tool_agent2,
    skeptic_tool_agent3,
    skeptic_tool_agent4

]

# 3. 用一個迴圈直接塞進去
for agent in agents_need_search:
    agent.tools = [shared_tool]


# ========== Root Pipeline ==========
init_session = LlmAgent(
    name="init_session",
    model="gemini-2.5-flash",
    instruction=(
        "初始化 session。"
        "請直接將使用者輸入內容重新敘說一次，切勿改變使用者輸入內容，且勿產生任何其他不屬於使用者輸入的文字，保留最原始使用者輸入文字。"
    ),
    #before_agent_callback=_before_init_session,
    output_key="_init_session",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
)


# ✅ 修正: 使用正確初始化的 dynamic_fact_check
root_agent = SequentialAgent(
    name="root_pipeline",
    sub_agents=[
        init_session,
        curator_agent,
        historian_agent,
        orchestrator_agent,
        social_summary_agent,
        adjudication_agent,
        llm_agent,
        classifier_agent,
        weight_agent
    ],
)
agent = root_agent 

# 然後才是你剛才加的診斷碼（記得要把診斷碼移到 agent = root_agent 之後）
print("--- ADK 載入診斷開始 ---")
if 'agent' in globals():
    print(f"成功找到 agent 變數！內容為: {agent}")
else:
    print("致命錯誤：在此檔案中找不到名為 'agent' 的變數！")
print("--- ADK 載入診斷結束 ---")

if __name__ == "__main__":
    print("Testing framework overhead...")
    session = create_session()
    bind_session(session)
    print("End of test.")