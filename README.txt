116 畢專網頁服務設定
1. 後端服務開啟
(1) cmd(fast api)
cd D:\test3
.\.venv\Scripts\activate
adk api_server

(2) docker(cofact api)
開啟 docker desktop
開啟 cofacts_api_refactor_deploy

2. 前端服務開啟
(1) powershell(開啟 Node.js)
cd D:\test3\fact-check-system\server
$env:PORT = "4000"
$env:DB_SSL = "true"
npm.cmd run dev

(2) cmd
cd D:\test3
.\.venv\Scripts\activate
cd D:\test3\fact-check-system\echo_debate_of_school_project
npm run dev

3. 更換 gemini api
開啟 D:\test3\.env
更換 第2行 GOOGLE_API_KEY="..."