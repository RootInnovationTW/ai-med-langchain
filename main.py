#!/usr/bin/env python3
# ============================================================
# 🎯 AI-Powered Virtual Screening & Healthcare LangChain Agent
# 主程式入口
# 作者: Silvia 團隊
# ============================================================

import os
import requests
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from security.protect_agent import ProtectAgent  # 自訂安全模組
from services.docking_service import run_docking  # Docking (DiffDock)
from services.protein_service import predict_structure  # OpenFold2
from services.molecule_service import generate_molecule  # GenMol

# ============================================================
# 🔒 初始化安全防護 (例如 Prompt Injection, HIPAA, GDPR)
# ============================================================
security_layer = ProtectAgent()

# ============================================================
# ⚙️ NIM API Endpoints (可改成 .env 或 config.json)
# ============================================================
NIM_ENDPOINTS = {
    "msa": os.getenv("MSA_HOST", "http://localhost:8081"),
    "openfold2": os.getenv("OPENFOLD2_HOST", "http://localhost:8082"),
    "genmol": os.getenv("GENMOL_HOST", "http://localhost:8083"),
    "diffdock": os.getenv("DIFFDOCK_HOST", "http://localhost:8084"),
}

# ============================================================
# 🛠️ 定義工具給 LangChain Agent
# ============================================================
tools = [
    Tool(
        name="Protein Structure Prediction",
        func=predict_structure,
        description="使用 OpenFold2 NIM 預測蛋白質結構"
    ),
    Tool(
        name="Molecule Generation",
        func=generate_molecule,
        description="使用 GenMol NIM 生成小分子"
    ),
    Tool(
        name="Docking Simulation",
        func=run_docking,
        description="使用 DiffDock NIM 進行 docking 模擬"
    ),
]

# ============================================================
# 🤖 建立 LangChain Agent
# ============================================================
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# ============================================================
# 🚀 主流程 (簡單 Demo)
# ============================================================
def main():
    print("🚀 啟動 AI Healthcare LangChain Agent System...")

    # 1. 安全檢查
    query = "請幫我生成一個小分子並做 docking"
    if not security_layer.check_prompt(query):
        print("❌ 偵測到危險 Prompt，已阻擋！")
        return

    # 2. 呼叫 Agent
    response = agent.run(query)
    print("🤖 Agent 回應：", response)


if __name__ == "__main__":
    main()
