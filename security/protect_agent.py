"""
protect_agent.py
醫療 AI 代理系統 (Medical AI Agent)
-------------------------------------------------------
特色：
1. 整合 LangChain + LangGraph
2. 具備 HIPAA 合規的醫療安全監控
3. 內建三大醫療工具：
   - 診斷分析 DiagnosticAnalysis
   - 治療建議 TreatmentRecommendation
   - 專科會診 SpecialistConsultation
4. 提供 SOP 式中文註解，協助新工程師快速上手
"""

from typing import List, Dict, Any, Optional, Type, TypedDict, Annotated
from datetime import datetime, timedelta
from enum import Enum
import re
import operator
import logging
import hashlib
import secrets
from dataclasses import dataclass

# === LangChain 套件 ===
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# === LangGraph 套件 ===
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

# === 醫療專用套件 ===
import numpy as np
from sklearn.ensemble import IsolationForest

# === 設定 logging，方便 debug ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ========================================================
# 1. 定義 Agent 狀態 (State)
# ========================================================

class MedicalAgentState(TypedDict):
    """醫療 AI 代理系統的狀態定義"""
    messages: Annotated[List[BaseMessage], operator.add]  # 目前對話紀錄
    patient_id: Optional[str]                             # 病人 ID
    case_priority: Optional[str]                          # 案件優先等級
    medical_context: Dict[str, Any]                       # 醫療上下文
    diagnosis_chain: List[Dict[str, Any]]                 # 診斷推理鏈
    security_context: Dict[str, Any]                      # 安全性監控結果
    risk_level: str                                       # 風險等級
    session_id: str                                       # 工作階段 ID
    specialist_consultations: List[Dict[str, Any]]        # 專科會診紀錄
    treatment_recommendations: List[Dict[str, Any]]       # 治療建議紀錄

# 風險等級
class MedicalRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

# 案件優先級
class CasePriority(Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"
    CRITICAL = "critical"

# 醫療科別
class MedicalSpecialty(Enum):
    CARDIOLOGY = "cardiology"              # 心臟科
    NEUROLOGY = "neurology"                # 神經科
    ONCOLOGY = "oncology"                  # 腫瘤科
    EMERGENCY = "emergency_medicine"       # 急診
    INTERNAL = "internal_medicine"         # 內科
    RADIOLOGY = "radiology"                # 放射科
    PATHOLOGY = "pathology"                # 病理科

# ========================================================
# 2. 病人資料結構 (Patient Record)
# ========================================================

@dataclass
class PatientRecord:
    """病人資料結構，符合 HIPAA 匿名化需求"""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    vital_signs: Dict[str, float]
    lab_results: Dict[str, Any]
    imaging_results: List[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime

    def to_anonymized_dict(self):
        """回傳匿名化後的病人資料，供 AI 使用"""
        return {
            "patient_hash": hashlib.sha256(self.patient_id.encode()).hexdigest()[:8],
            "demographics": {
                "age_group": f"{(self.age // 10) * 10}-{(self.age // 10) * 10 + 9}",
                "gender": self.gender
            },
            "chief_complaint": self.chief_complaint,
            "medical_history": self.medical_history,
            "current_medications": self.medications,
            "known_allergies": self.allergies,
            "vitals": self.vital_signs,
            "recent_labs": self.lab_results,
            "imaging": [{"type": img["type"], "findings": img["findings"]}
                        for img in self.imaging_results]
        }

# ========================================================
# 3. 醫療安全監控模組 (Security Monitor)
# ========================================================

class MedicalSecurityMonitor:
    """醫療 AI 系統的安全監控器，檢查 HIPAA 與臨床風險"""

    def __init__(self):
        self.security_events = []
        # PHI (Protected Health Information) 檢測
        self.phi_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",       # SSN
            r"\b\d{10,}\b",                  # 病歷號碼
            r"\b[A-Z]{2}\d{6,}\b",           # 保險號碼
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # 信用卡號
        ]
        # 臨床警示字詞
        self.medical_alert_patterns = [
            r"(?i)\b(suicide|self-harm|overdose)\b",
            r"(?i)\b(chest pain|myocardial infarction|stroke)\b",
            r"(?i)\b(allergic reaction|anaphylaxis)\b",
        ]
        # 使用 IsolationForest 機器學習檢測異常輸入
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def evaluate_medical_input_security(self, input_text: str, patient_context: Dict) -> Dict[str, Any]:
        """分析輸入訊息：檢測 PHI、臨床緊急情況、異常輸入"""
        risk_factors, clinical_alerts = [], []
        risk_level = MedicalRiskLevel.LOW

        # === PHI 檢測 ===
        for pattern in self.phi_patterns:
            if re.search(pattern, input_text):
                risk_factors.append(f"可能的 PHI 偵測: {pattern}")
                risk_level = max(risk_level, MedicalRiskLevel.HIGH)

        # === 臨床警示檢測 ===
        for pattern in self.medical_alert_patterns:
            matches = re.findall(pattern, input_text)
            if matches:
                clinical_alerts.extend(matches)
                if any(term in input_text.lower() for term in ["suicide", "self-harm"]):
                    risk_level = MedicalRiskLevel.EMERGENCY
                elif any(term in input_text.lower() for term in ["chest pain", "stroke"]):
                    risk_level = max(risk_level, MedicalRiskLevel.CRITICAL)

        # === 異常輸入檢測 ===
        if self._detect_input_anomaly(input_text):
            risk_factors.append("異常輸入模式偵測")
            risk_level = max(risk_level, MedicalRiskLevel.MEDIUM)

        return {
            "risk_level": risk_level.value,
            "risk_factors": risk_factors,
            "clinical_alerts": clinical_alerts,
            "requires_immediate_attention": risk_level in [
                MedicalRiskLevel.CRITICAL, MedicalRiskLevel.EMERGENCY
            ],
            "blocked": risk_level == MedicalRiskLevel.EMERGENCY and "suicide" in input_text.lower()
        }

    def _detect_input_anomaly(self, input_text: str) -> bool:
        """利用機器學習模型檢測異常輸入"""
        features = np.array([[
            len(input_text),
            len(input_text.split()),
            input_text.count("?"),
            input_text.count("!"),
            sum(1 for c in input_text if c.isupper()) / len(input_text) if input_text else 0,
        ]])
        try:
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            return anomaly_score < -0.5
        except Exception:
            return False

# ========================================================
# 4. 醫療工具模組 (Tools)
# ========================================================

class SecureDiagnosticAnalysisTool(BaseTool):
    """診斷分析工具"""
    name = "diagnostic_analysis"
    description = "進行病人診斷分析"
    args_schema: Type[BaseModel] = BaseModel
    def _run(self, **kwargs): return "診斷分析結果"

class SecureTreatmentRecommendationTool(BaseTool):
    """治療建議工具"""
    name = "treatment_recommendations"
    description = "提供治療建議"
    args_schema: Type[BaseModel] = BaseModel
    def _run(self, **kwargs): return "治療建議結果"

class SpecialistConsultationTool(BaseTool):
    """專科會診工具"""
    name = "specialist_consultation"
    description = "轉介病人至專科醫師"
    args_schema: Type[BaseModel] = BaseModel
    def _run(self, **kwargs): return "專科會診結果"

# ========================================================
# 5. 醫療 AI 代理系統 (Medical AI Agent)
# ========================================================

class MedicalAIAgent:
    """醫療 AI 代理：結合安全監控 + LLM + 專科工具"""

    def __init__(self, openai_api_key: Optional[str] = None, use_mock_llm: bool = True):
        # 啟用安全監控
        self.security_monitor = MedicalSecurityMonitor()

        # 綁定三個醫療工具
        self.tools = [
            SecureDiagnosticAnalysisTool(),
            SecureTreatmentRecommendationTool(),
            SpecialistConsultationTool()
        ]
        self.tool_executor = ToolExecutor(self.tools)

        # 使用 Mock LLM (模擬醫療 AI 回覆)，或連線至 OpenAI GPT
        self.llm = self._create_mock_llm() if (use_mock_llm or not openai_api_key) else ChatOpenAI(
            api_key=openai_api_key, model="gpt-4", temperature=0.1)

        # 建立 LangGraph 工作流
        self.graph = self._build_graph()
        self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _create_mock_llm(self):
        """建立一個模擬 LLM（方便測試，不需 API Key）"""
        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="您好，我是醫療 AI 助手，請提供病人資訊。")
        return MockLLM()

    def _build_graph(self) -> StateGraph:
        """建立醫療工作流：先過安全檢查，再做推理"""
        def security_gate(state: MedicalAgentState) -> MedicalAgentState:
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else ""
            security_result = self.security_monitor.evaluate_medical_input_security(last_message, {})
            return {"messages": [], "security_context": security_result}

        def reasoning_agent(state: MedicalAgentState) -> MedicalAgentState:
            messages = state.get("messages", [])
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(MedicalAgentState)
        workflow.add_node("security_gate", security_gate)
        workflow.add_node("reasoning_agent", reasoning_agent)
        workflow.add_edge("security_gate", "reasoning_agent")
        workflow.set_entry_point("security_gate")
        workflow.set_finish_point("reasoning_agent")
        return workflow

