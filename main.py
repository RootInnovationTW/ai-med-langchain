#!/usr/bin/env python3
# ============================================================
# 🎯 AI-Powered Virtual Screening & Healthcare LangChain Agent
# 主程式入口
# 作者: Silvia 團隊
# ============================================================

import os
import requests
import logging
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from security.protect_agent import ProtectAgent  # 自訂安全模組
from services.docking_service import run_docking  # Docking (DiffDock)
from services.protein_service import predict_structure, analyze_protein_sequence, extract_protein_binding_sites  # OpenFold2
from services.molecule_service import generate_molecule  # GenMol
from visualization import visualize_protein_structure, create_analysis_dashboard  # 可視化模組

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
        description="使用 OpenFold2 NIM 預測蛋白質結構。輸入蛋白質氨基酸序列，返回結構預測結果包含置信度分數和PDB數據。"
    ),
    Tool(
        name="Protein Sequence Analysis", 
        func=analyze_protein_sequence,
        description="分析蛋白質序列屬性，包含分子量、疏水性、電荷分佈等理化性質計算。"
    ),
    Tool(
        name="Protein Binding Sites",
        func=extract_protein_binding_sites, 
        description="預測蛋白質結合位點，可選擇性分析與特定配體的兼容性。輸入蛋白質序列和可選的配體SMILES。"
    ),
    Tool(
        name="Molecule Generation",
        func=generate_molecule,
        description="使用 GenMol NIM 生成小分子。可根據目標屬性或參考分子生成新的候選化合物。"
    ),
    Tool(
        name="Docking Simulation",
        func=run_docking,
        description="使用 DiffDock NIM 進行分子對接模擬。輸入蛋白質結構(PDB文件路徑或序列)和配體SMILES，返回對接分數和結合模式。"
    ),
    Tool(
        name="Protein Structure Visualization",
        func=visualize_protein_structure,
        description="創建蛋白質結構的3D可視化。可顯示預測結構的置信度著色，支持多種顯示樣式(cartoon、stick、sphere)。"
    ),
    Tool(
        name="Analysis Dashboard",
        func=create_analysis_dashboard,
        description="生成綜合分析報告和交互式儀表板。整合蛋白質預測、分子對接等結果，生成HTML報告和可視化圖表。"
    ),
]

# ============================================================
# 🤖 建立 LangChain Agent
# ============================================================
def create_agent():
    """創建並配置 LangChain Agent"""
    try:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        agent = initialize_agent(
            tools, 
            llm, 
            agent="zero-shot-react-description", 
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        return agent
    except Exception as e:
        logging.error(f"Failed to create agent: {e}")
        return None

# ============================================================
# 🔧 輔助功能函數
# ============================================================

def validate_environment():
    """驗證環境配置"""
    issues = []
    
    # 檢查必要的環境變數
    required_vars = ["NVIDIA_API_KEY", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing environment variable: {var}")
    
    # 檢查NIM endpoints連接性
    for name, endpoint in NIM_ENDPOINTS.items():
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            if response.status_code != 200:
                issues.append(f"NIM endpoint {name} ({endpoint}) not healthy")
        except requests.RequestException:
            issues.append(f"Cannot connect to NIM endpoint {name} ({endpoint})")
    
    return issues

def run_interactive_mode():
    """交互式對話模式"""
    print("\n🤖 進入交互式模式 (輸入 'quit' 退出)")
    print("=" * 50)
    
    agent = create_agent()
    if not agent:
        print("❌ Agent 創建失敗")
        return
    
    while True:
        try:
            query = input("\n👤 您的問題: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                print("👋 再見！")
                break
            
            if not query:
                continue
            
            # 安全檢查
            if not security_layer.check_prompt(query):
                print("❌ 偵測到危險 Prompt，已阻擋！")
                continue
            
            # 執行查詢
            print("\n🔄 處理中...")
            response = agent.run(query)
            print(f"\n🤖 Agent 回應：\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 用戶中斷，退出...")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            logging.error(f"Interactive mode error: {e}")

def run_demo_workflow():
    """執行演示工作流程"""
    print("\n🧪 執行演示工作流程...")
    print("=" * 50)
    
    agent = create_agent()
    if not agent:
        print("❌ Agent 創建失敗")
        return
    
    # 演示查詢序列
    demo_queries = [
        "預測這個蛋白質序列的結構：GIVEQCCTSICSLYQLENYCN",
        "分析上述蛋白質的理化性質",
        "為COVID-19主蛋白酶生成一個潛在抑制劑分子",
        "進行蛋白質-配體對接分析",
        "創建綜合分析報告和可視化"
    ]
    
    results = []
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 步驟 {i}: {query}")
        print("-" * 30)
        
        try:
            # 安全檢查
            if not security_layer.check_prompt(query):
                print("❌ 安全檢查失敗，跳過此查詢")
                continue
            
            response = agent.run(query)
            results.append({"query": query, "response": response})
            print(f"✅ 完成步驟 {i}")
            
        except Exception as e:
            print(f"❌ 步驟 {i} 失敗: {e}")
            logging.error(f"Demo workflow error at step {i}: {e}")
    
    print(f"\n🎉 演示完成！成功執行 {len(results)}/{len(demo_queries)} 個步驟")
    return results

# ============================================================
# 🚀 主流程
# ============================================================
def main():
    """主程式入口"""
    print("🚀 啟動 AI Healthcare LangChain Agent System...")
    print("🔬 整合 NVIDIA BioNeMo NIMs + 可視化分析平台")
    print("=" * 60)
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 驗證環境
    print("🔍 驗證環境配置...")
    issues = validate_environment()
    
    if issues:
        print("⚠️  發現環境問題:")
        for issue in issues:
            print(f"  • {issue}")
        print("⚠️  系統可能無法完全正常運行")
    else:
        print("✅ 環境配置正常")
    
    # 顯示可用工具
    print(f"\n🛠️  可用工具 ({len(tools)} 個):")
    for tool in tools:
        print(f"  • {tool.name}")
    
    # 選擇運行模式
    print("\n🎯 選擇運行模式:")
    print("  1. 交互式對話模式")
    print("  2. 演示工作流程")
    print("  3. 單次查詢模式")
    
    try:
        mode = input("\n請選擇模式 (1/2/3): ").strip()
        
        if mode == "1":
            run_interactive_mode()
        elif mode == "2":
            run_demo_workflow()
        elif mode == "3":
            # 單次查詢模式 (原有的簡單演示)
            query = "請幫我預測一個蛋白質結構並進行可視化分析"
            
            if not security_layer.check_prompt(query):
                print("❌ 偵測到危險 Prompt，已阻擋！")
                return
            
            agent = create_agent()
            if agent:
                response = agent.run(query)
                print(f"\n🤖 Agent 回應：\n{response}")
        else:
            print("❌ 無效選擇，退出...")
            
    except KeyboardInterrupt:
        print("\n👋 用戶中斷，退出...")
    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
