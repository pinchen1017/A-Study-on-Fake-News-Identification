# root_agent.py
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool, ToolContext
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizerFast, BertModel
import logging
import os
from transformers import BertForSequenceClassification, BertTokenizerFast
from google.genai import types
from pydantic import BaseModel, Field
from google.genai.types import GenerateContentConfig, FunctionCallingConfig, Type

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# 初始化模型部分
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
class MLPHead(nn.Module):
    def __init__(self, hidden, num_labels=2):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
class Bert_ConcatLast3CLS_MLP(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese", output_hidden_states=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.mlp = MLPHead(768*3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_states = outputs.hidden_states
        cls_last3 = torch.cat([hidden_states[-3][:,0,:],
                               hidden_states[-2][:,0,:],
                               hidden_states[-1][:,0,:]], dim=1)
        logits = self.mlp(cls_last3)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


try:
    # 支援透過環境變數跳過模型載入（例如開發或 CI 時避免網路下載）
    if os.environ.get("AGENT_JUDGE_SKIP_MODEL_LOAD") == "1":
        logger.info("環境變數 AGENT_JUDGE_SKIP_MODEL_LOAD=1，跳過模型與 tokenizer 載入")
        tokenizer_test = None
        model_test = None
        id2label = {0: "真", 1: "假"}
    else:
        model_name_test = "bert-base-chinese"
        tokenizer_test = BertTokenizerFast.from_pretrained(model_name_test)
        logger.info(f"從 {model_name_test} 載入原始 Tokenizer")
        # 優先使用本地模型目錄（相對路徑）以避免在 Linux 上使用 Windows 路徑導致失敗
        from pathlib import Path

        # 本檔案的同級資料夾中預期有 bert_fake_news_model
        this_dir = Path(__file__).resolve().parent
        local_model_path = this_dir / "Bert_ConcatLast3CLS_MLP.pth"

        # 另外也嘗試 repo 相對路徑 judge/agents/classifier/bert_fake_news_model（以防工作目錄不同）
        repo_model_path = Path.cwd() / "judge" / "agents" / "classifier" / "Bert_ConcatLast3CLS_MLP.pth"

        chosen_pt_path = None
        if local_model_path.exists():
            chosen_pt_path = local_model_path
        elif repo_model_path.exists():
            chosen_pt_path = repo_model_path

        model_test = Bert_ConcatLast3CLS_MLP(freeze_bert=False).to(device)
        
        if chosen_pt_path is not None:
            logger.info(f"使用自定義權重檔案: {chosen_pt_path}")
            # 載入 state_dict
            state_dict = torch.load(chosen_pt_path, map_location=device)
            model_test.load_state_dict(state_dict)
        else:
            logger.warning("找不到 .pt 權重檔，模型將處於原始(未微調)狀態！")

        model_test.eval()  # 設定為評估模式

        id2label = {0: "真", 1: "假"}
    
    logger.info("BERT 模型載入成功")
    
except Exception as e:
    logger.error(f"模型載入失敗: {e}")
    raise

# -----------------------
# 定義分類函數作為工具 - 強制呼叫版本
# -----------------------
def classify_text_with_context(text: str, tool_context: ToolContext) -> dict:
    """
    使用 BERT 模型分類文本真假。
    
    **重要：這是唯一的文本分析工具，必須用於所有文本分類請求。**
    **當用戶提供任何需要分析的文本時，都必須呼叫這個函數。**
    
    Args:
        text: 要分類的文本內容
        
    Returns:
        包含分類結果的字典，格式：
        {
            'status': 'success'/'error',
            'label': '真'/'假',
            'probability': float (0-1之間的機率值),
            'input_text': 原始輸入文本
        }
    """
    logger.info(f"=== classify_text_with_context 被呼叫 ===")

    real_text = tool_context.state.get("_init_session")

    # 如果 state 沒有才退回使用 tool 參數
    if not real_text:
        real_text = text

    logger.info(f"實際分類文本: {real_text}")
    # 設置狀態標記，表示工具已被呼叫
    tool_context.state["classification_tool_called"] = True
    tool_context.state["analyzed_text"] = real_text
    
    try:
        with torch.no_grad():
            encoding = tokenizer_test(
                        real_text,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)

            # 丟進模型
            outputs = model_test(**encoding)
            logits = outputs["logits"]
            # softmax 機率
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 模型預測標籤 (0=真新聞, 1=假新聞)
            pred_label = probs.argmax()

            # 只取 label=0 (真新聞) 的機率
            prob_label0 = probs[0].item()
            pred_label_text = id2label.get(pred_label, str(pred_label))

        result = {
            "status": "success",
            "label": pred_label_text,
            "probability": prob_label0,
            "input_text": real_text
        }
        
        # 將結果存入狀態
        tool_context.state["classification_result"] = result
        
        logger.info(f"分類完成: {pred_label_text} (confidence: {prob_label0:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"分類過程中發生錯誤: {e}")
        error_result = {
            "status": "error",
            "label": "錯誤",
            "probability": 0.0,
            "input_text": real_text,
            "error": str(e)
        }
        tool_context.state["classification_result"] = error_result
        return error_result

def check_tool_usage(tool_context: ToolContext) -> dict:
    """
    檢查是否已經使用了分類工具。如果沒有使用，則提醒必須使用。
    """
    if not tool_context.state.get("classification_tool_called", False):
        return {
            "status": "warning",
            "message": "尚未執行文本分析，請先使用 classify_text_with_context 工具分析文本。"
        }
    return {
        "status": "ok",
        "message": "已完成文本分析"
    }

class classificationOutput(BaseModel):
    Probability: str = Field(description="真新聞的機率")
    classification: str = Field(description="真假分類：正確 或 錯誤")

# 創建工具實例
classification_tool = FunctionTool(func=classify_text_with_context)
check_tool = FunctionTool(func=check_tool_usage)

# -----------------------
# 使用 LlmAgent 來處理用戶輸入 - 強制工具呼叫版本
# -----------------------
bert_classifier_model_agent = LlmAgent(
    name="bert_classifier",
    model="gemini-2.5-flash",
    instruction="""你是文本分析專家。你只能透過呼叫工具來分析文本，不能直接判斷。

    == 強制執行規則 ==
    無論用戶輸入什麼內容，你都必須：
    1. 立即使用 classification_tool 工具
    2. 將用戶的完整輸入傳遞給工具
    3. 等待並報告工具結果
    4. 得到結果後，呼叫check_tool　判斷是否使用classification_tool　工具
    5. check_tool判斷沒有呼叫的話，請重新呼叫classification_tool得到真實classification_tool的結果。

    == 禁止行為 ==
    ✗ 不要嘗試自己分析文本
    ✗ 不要猜測結果
    ✗ 不要跳過工具呼叫
    ✗ 不要自己捏照結果

    == 必要步驟 ==
    第一步：呼叫 classification_tool(用戶輸入)
    第二步：回報工具分析結果
    第三步：呼叫 check_tool 檢查使否有呼叫classification_tool
    

    記住：classification_tool 工具是你唯一的分析手段。""",
    
    tools=[classification_tool, check_tool],
    output_key="classification_result_reply",
    generate_content_config=types.GenerateContentConfig(temperature=0.1)
)

classification_schema_agent = LlmAgent(
    name="bert_classifier_validator", 
    model="gemini-2.5-flash",
    instruction="""你負責將 state['classification_result'] 轉換為符合 classificationOutput schema 的 JSON 格式。

        **處理規則：**
        1. 檢查 state 中是否存在 'classification_result'
        2. 提取其中的 'probability' 和 'label' 資訊
        3. 將 'label' 轉換為 'classification' 欄位：
        - "真" → "正確"  
        - "假" → "錯誤"


        **輸出格式：**
        僅輸出符合 schema 的 JSON，不要包含任何額外文字或解釋。

        範例輸出：
        {
        "Probability": "0.85",
        "classification": "正確"
        }""",

    output_schema=classificationOutput,
    output_key="classification_json",
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

# -------- Step 3: Sequential pipeline --------
classifier_agent = SequentialAgent(
    name="bert_classifier_agent",
    sub_agents=[bert_classifier_model_agent, classification_schema_agent],
)