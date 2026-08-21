# 1. 選擇基礎映像檔：選擇適合您語言和版本的基礎映像檔
# Python 3.10 輕量版，適用於生產環境
FROM python:3.12-slim


# 3. 設定工作目錄
WORKDIR /app

# 4. 複製依賴文件並安裝依賴套件 (利用 Docker Layer Caching)
# 確保 requirements.txt 檔案存在
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製 Agent 程式碼到容器內
# 複製整個程式碼目錄到 /app
COPY . .

# 6. 宣告暴露埠號 (如果 Agent 程式是 Web 服務，如 FastAPI 或 Flask)
# 假設您的 Agent 在 8000 埠提供服務
EXPOSE 8000 

# 7. 設定容器啟動時的執行命令 (Entrypoint/CMD)
# 關鍵修正：必須傳遞 --host 0.0.0.0 參數，讓服務在容器內對外部連線開放
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000"]