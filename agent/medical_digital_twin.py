"""
醫療數據分析與數位孿生整合模組
整合：醫療AI分析 + 數位孿生技術 + LangChain智能問答
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 機器學習組件
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 數位孿生相關
import simpy
import threading
from queue import Queue

# LangChain整合
from langchain.tools import Tool
from langchain.schema import BaseOutputParser

class PatientDigitalTwin:
    """
    病患數位孿生類
    即時模擬病患健康狀態和治療反應
    """
    
    def __init__(self, patient_id: str, initial_health_data: Dict):
        self.patient_id = patient_id
        self.health_state = initial_health_data
        self.vital_signs_history = []
        self.treatment_responses = []
        self.simulation_env = simpy.Environment()
        
        # 初始化數位孿生參數
        self.health_metrics = {
            'heart_rate': initial_health_data.get('heart_rate', 72),
            'blood_pressure': initial_health_data.get('blood_pressure', 120),
            'temperature': initial_health_data.get('temperature', 36.5),
            'glucose_level': initial_health_data.get('glucose_level', 100),
            'oxygen_saturation': initial_health_data.get('oxygen_saturation', 98)
        }
        
        # 啟動即時模擬
        self.simulation_process = self.simulation_env.process(
            self._real_time_simulation()
        )
    
    def _real_time_simulation(self):
        """即時健康狀態模擬"""
        while True:
            # 模擬生理變化
            self._simulate_physiological_changes()
            
            # 記錄健康數據
            current_state = {
                'timestamp': datetime.now(),
                'metrics': self.health_metrics.copy(),
                'risk_score': self._calculate_risk_score()
            }
            self.vital_signs_history.append(current_state)
            
            # 每5秒更新一次
            yield self.simulation_env.timeout(5)
    
    def _simulate_physiological_changes(self):
        """模擬生理參數變化"""
        # 基於當前狀態和外部因素模擬變化
        import random
        
        # 心臟率自然波動
        self.health_metrics['heart_rate'] += random.uniform(-2, 2)
        self.health_metrics['heart_rate'] = max(60, min(120, self.health_metrics['heart_rate']))
        
        # 血糖變化模擬
        self.health_metrics['glucose_level'] += random.uniform(-5, 5)
        self.health_metrics['glucose_level'] = max(70, min(300, self.health_metrics['glucose_level']))
    
    def _calculate_risk_score(self) -> float:
        """計算健康風險評分"""
        risk_factors = 0
        
        if self.health_metrics['heart_rate'] > 100:
            risk_factors += 1
        if self.health_metrics['blood_pressure'] > 140:
            risk_factors += 1
        if self.health_metrics['glucose_level'] > 180:
            risk_factors += 1
        if self.health_metrics['oxygen_saturation'] < 95:
            risk_factors += 1
            
        return risk_factors / 4.0
    
    def simulate_treatment(self, treatment_plan: Dict) -> Dict:
        """
        模擬治療方案效果
        
        Args:
            treatment_plan: 治療計劃字典
        """
        print(f"🔬 在數位孿生上模擬治療方案: {treatment_plan['name']}")
        
        # 模擬治療效果
        simulated_effects = {}
        
        if 'medication' in treatment_plan:
            for med in treatment_plan['medication']:
                effect = self._simulate_medication_effect(med)
                simulated_effects[med['name']] = effect
        
        # 更新健康狀態
        self._apply_treatment_effects(simulated_effects)
        
        # 預測治療結果
        outcome_prediction = self._predict_treatment_outcome(treatment_plan)
        
        simulation_result = {
            'treatment_plan': treatment_plan,
            'simulated_effects': simulated_effects,
            'predicted_outcome': outcome_prediction,
            'risk_reduction': self._calculate_risk_reduction(),
            'timestamp': datetime.now()
        }
        
        self.treatment_responses.append(simulation_result)
        return simulation_result
    
    def _simulate_medication_effect(self, medication: Dict) -> Dict:
        """模擬藥物效果"""
        # 基於藥物類型和劑量模擬效果
        effect = {}
        
        if medication['type'] == 'antihypertensive':
            effect['blood_pressure_reduction'] = medication['dose'] * 0.5
            effect['heart_rate_effect'] = medication['dose'] * -0.3
            
        elif medication['type'] == 'glucose_control':
            effect['glucose_reduction'] = medication['dose'] * 2.0
            
        elif medication['type'] == 'anti_inflammatory':
            effect['inflammation_reduction'] = medication['dose'] * 1.5
            
        return effect
    
    def _apply_treatment_effects(self, effects: Dict):
        """應用治療效果到數位孿生"""
        for medication, effect in effects.items():
            if 'blood_pressure_reduction' in effect:
                self.health_metrics['blood_pressure'] -= effect['blood_pressure_reduction']
            
            if 'glucose_reduction' in effect:
                self.health_metrics['glucose_level'] -= effect['glucose_reduction']
    
    def _predict_treatment_outcome(self, treatment_plan: Dict) -> str:
        """預測治療結果"""
        current_risk = self._calculate_risk_score()
        
        # 模擬治療後的風險變化
        predicted_risk = current_risk * 0.7  # 假設治療降低30%風險
        
        if predicted_risk < 0.3:
            return "優秀：預期健康狀況顯著改善"
        elif predicted_risk < 0.6:
            return "良好：預期有明顯改善"
        else:
            return "一般：需要進一步治療調整"
    
    def _calculate_risk_reduction(self) -> float:
        """計算風險降低百分比"""
        if len(self.vital_signs_history) < 2:
            return 0.0
        
        current_risk = self._calculate_risk_score()
        initial_risk = self.vital_signs_history[0]['risk_score']
        
        return (initial_risk - current_risk) / initial_risk * 100
    
    def get_health_report(self) -> Dict:
        """獲取數位孿生健康報告"""
        return {
            'patient_id': self.patient_id,
            'current_metrics': self.health_metrics,
            'current_risk_score': self._calculate_risk_score(),
            'historical_trends': self.vital_signs_history[-10:],  # 最近10個記錄
            'treatment_history': self.treatment_responses
        }

class MedicalDigitalTwinSystem:
    """
    醫療數位孿生系統
    整合AI分析和數位孿生模擬
    """
    
    def __init__(self):
        self.digital_twins = {}  # patient_id -> PatientDigitalTwin
        self.medical_models = {}
        self.analysis_history = []
        
        # 初始化AI模型
        self._initialize_ai_models()
    
    def _initialize_ai_models(self):
        """初始化AI預測模型"""
        # 疾病風險預測模型
        self.medical_models['disease_risk'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 治療效果預測模型
        self.medical_models['treatment_response'] = GradientBoostingClassifier(random_state=42)
        
        print("✅ AI醫療模型初始化完成")
    
    def create_patient_twin(self, patient_data: Dict) -> str:
        """
        為病患創建數位孿生
        
        Args:
            patient_data: 病患基本資料和健康數據
        """
        patient_id = patient_data.get('patient_id', f"patient_{len(self.digital_twins) + 1}")
        
        digital_twin = PatientDigitalTwin(patient_id, patient_data)
        self.digital_twins[patient_id] = digital_twin
        
        print(f"✅ 已為病患 {patient_id} 創建數位孿生")
        return patient_id
    
    def analyze_patient_risk(self, patient_id: str) -> Dict:
        """
        分析病患健康風險
        """
        if patient_id not in self.digital_twins:
            return {"error": "找不到對應的數位孿生"}
        
        twin = self.digital_twins[patient_id]
        health_report = twin.get_health_report()
        
        # AI風險分析
        risk_analysis = self._perform_ai_risk_analysis(health_report)
        
        analysis_result = {
            'patient_id': patient_id,
            'timestamp': datetime.now(),
            'health_report': health_report,
            'risk_analysis': risk_analysis,
            'recommendations': self._generate_recommendations(risk_analysis)
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    def _perform_ai_risk_analysis(self, health_report: Dict) -> Dict:
        """執行AI風險分析"""
        metrics = health_report['current_metrics']
        
        # 計算各項風險指標
        cardiovascular_risk = self._calculate_cardiovascular_risk(metrics)
        diabetes_risk = self._calculate_diabetes_risk(metrics)
        overall_risk = health_report['current_risk_score']
        
        return {
            'cardiovascular_risk': cardiovascular_risk,
            'diabetes_risk': diabetes_risk,
            'overall_risk_score': overall_risk,
            'risk_level': self._classify_risk_level(overall_risk),
            'critical_alerts': self._check_critical_alerts(metrics)
        }
    
    def _calculate_cardiovascular_risk(self, metrics: Dict) -> float:
        """計算心血管疾病風險"""
        risk_score = 0
        
        if metrics['blood_pressure'] > 140:
            risk_score += 0.4
        if metrics['heart_rate'] > 100:
            risk_score += 0.3
        if metrics.get('cholesterol', 200) > 200:
            risk_score += 0.3
            
        return min(1.0, risk_score)
    
    def _calculate_diabetes_risk(self, metrics: Dict) -> float:
        """計算糖尿病風險"""
        risk_score = 0
        
        if metrics['glucose_level'] > 140:
            risk_score += 0.6
        if metrics['glucose_level'] > 180:
            risk_score += 0.4
            
        return min(1.0, risk_score)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """分類風險等級"""
        if risk_score < 0.3:
            return "低風險"
        elif risk_score < 0.6:
            return "中風險"
        else:
            return "高風險"
    
    def _check_critical_alerts(self, metrics: Dict) -> List[str]:
        """檢查危急警報"""
        alerts = []
        
        if metrics['blood_pressure'] > 180:
            alerts.append("血壓過高，需要立即關注")
        if metrics['glucose_level'] > 300:
            alerts.append("血糖過高，需要醫療介入")
        if metrics['oxygen_saturation'] < 90:
            alerts.append("血氧飽和度過低，需要緊急處理")
            
        return alerts
    
    def _generate_recommendations(self, risk_analysis: Dict) -> List[str]:
        """生成醫療建議"""
        recommendations = []
        
        if risk_analysis['cardiovascular_risk'] > 0.5:
            recommendations.extend([
                "定期監測血壓和心率",
                "減少鈉鹽攝入",
                "適度有氧運動"
            ])
        
        if risk_analysis['diabetes_risk'] > 0.5:
            recommendations.extend([
                "控制碳水化合物攝入",
                "定期檢測血糖",
                "維持健康體重"
            ])
        
        if risk_analysis['risk_level'] == "高風險":
            recommendations.append("建議立即諮詢專科醫生")
        
        return recommendations
    
    def simulate_treatment_scenario(self, patient_id: str, treatment_plan: Dict) -> Dict:
        """
        在數位孿生上模擬治療方案
        """
        if patient_id not in self.digital_twins:
            return {"error": "找不到對應的數位孿生"}
        
        twin = self.digital_twins[patient_id]
        simulation_result = twin.simulate_treatment(treatment_plan)
        
        # 記錄模擬結果
        simulation_record = {
            'patient_id': patient_id,
            'treatment_plan': treatment_plan,
            'simulation_result': simulation_result,
            'timestamp': datetime.now()
        }
        
        self.analysis_history.append(simulation_record)
        return simulation_record
    
    def get_patient_timeline(self, patient_id: str) -> Dict:
        """
        獲取病患健康時間線
        """
        if patient_id not in self.digital_twins:
            return {"error": "找不到對應的數位孿生"}
        
        twin = self.digital_twins[patient_id]
        return {
            'patient_id': patient_id,
            'health_timeline': twin.vital_signs_history,
            'treatment_history': twin.treatment_responses,
            'analysis_history': [a for a in self.analysis_history if a.get('patient_id') == patient_id]
        }

# LangChain 整合工具
class DigitalTwinLangChainTools:
    """
    數位孿生 LangChain 工具
    """
    
    def __init__(self, digital_twin_system: MedicalDigitalTwinSystem):
        self.digital_twin_system = digital_twin_system
    
    def create_medical_analysis_tool(self) -> Tool:
        """創建醫療分析工具"""
        def analyze_patient_medical_data(patient_query: str) -> str:
            try:
                # 解析查詢中的病患資訊
                patient_data = self._parse_patient_query(patient_query)
                
                # 創建或獲取數位孿生
                patient_id = self.digital_twin_system.create_patient_twin(patient_data)
                
                # 執行分析
                analysis_result = self.digital_twin_system.analyze_patient_risk(patient_id)
                
                # 格式化回應
                return self._format_analysis_response(analysis_result)
                
            except Exception as e:
                return f"分析錯誤: {str(e)}"
        
        return Tool(
            name="Medical_Digital_Twin_Analysis",
            func=analyze_patient_medical_data,
            description="使用數位孿生技術分析病患健康數據並提供醫療建議"
        )
    
    def create_treatment_simulation_tool(self) -> Tool:
        """創建治療模擬工具"""
        def simulate_treatment_plan(simulation_query: str) -> str:
            try:
                # 解析治療模擬查詢
                patient_id, treatment_plan = self._parse_simulation_query(simulation_query)
                
                # 執行模擬
                simulation_result = self.digital_twin_system.simulate_treatment_scenario(
                    patient_id, treatment_plan
                )
                
                return self._format_simulation_response(simulation_result)
                
            except Exception as e:
                return f"模擬錯誤: {str(e)}"
        
        return Tool(
            name="Treatment_Simulation",
            func=simulate_treatment_plan,
            description="在病患數位孿生上模擬治療方案效果"
        )
    
    def _parse_patient_query(self, query: str) -> Dict:
        """解析病患查詢"""
        # 簡化的解析邏輯 - 實際應用中可以使用NLP
        patient_data = {
            'patient_id': f"query_{hash(query) % 10000}",
            'heart_rate': 75,
            'blood_pressure': 130,
            'temperature': 36.8,
            'glucose_level': 110,
            'oxygen_saturation': 97
        }
        
        # 可以根據查詢內容調整參數
        if "高血壓" in query:
            patient_data['blood_pressure'] = 160
        if "糖尿病" in query:
            patient_data['glucose_level'] = 220
        if "心率快" in query:
            patient_data['heart_rate'] = 105
            
        return patient_data
    
    def _parse_simulation_query(self, query: str) -> tuple:
        """解析模擬查詢"""
        # 簡化的解析邏輯
        patient_id = "patient_1"
        treatment_plan = {
            'name': '標準降血壓治療',
            'medication': [
                {
                    'name': '降壓藥',
                    'type': 'antihypertensive',
                    'dose': 10
                }
            ]
        }
        
        return patient_id, treatment_plan
    
    def _format_analysis_response(self, analysis_result: Dict) -> str:
        """格式化分析回應"""
        response = [
            "## 數位孿生醫療分析報告",
            f"**病患ID**: {analysis_result['patient_id']}",
            f"**分析時間**: {analysis_result['timestamp']}",
            "",
            "### 健康風險評估:",
            f"- 心血管疾病風險: {analysis_result['risk_analysis']['cardiovascular_risk']:.1%}",
            f"- 糖尿病風險: {analysis_result['risk_analysis']['diabetes_risk']:.1%}",
            f"- 總體風險等級: {analysis_result['risk_analysis']['risk_level']}",
            "",
            "### 醫療建議:"
        ]
        
        for recommendation in analysis_result['recommendations']:
            response.append(f"- {recommendation}")
        
        if analysis_result['risk_analysis']['critical_alerts']:
            response.append("")
            response.append("### ⚠️ 重要警報:")
            for alert in analysis_result['risk_analysis']['critical_alerts']:
                response.append(f"- {alert}")
        
        return "\n".join(response)
    
    def _format_simulation_response(self, simulation_result: Dict) -> str:
        """格式化模擬回應"""
        if 'error' in simulation_result:
            return f"模擬失敗: {simulation_result['error']}"
        
        result = simulation_result['simulation_result']
        
        response = [
            "## 治療方案模擬結果",
            f"**治療方案**: {result['treatment_plan']['name']}",
            f"**預測結果**: {result['predicted_outcome']}",
            f"**風險降低**: {result['risk_reduction']:.1f}%",
            "",
            "### 模擬效果:"
        ]
        
        for med, effect in result['simulated_effects'].items():
            response.append(f"- {med}: {effect}")
        
        return "\n".join(response)

# 整合到現有 AI-Med-LangChain 系統
def integrate_digital_twin_system(team_id: str = "medical_digital_twin_team"):
    """
    整合數位孿生系統到 AI-Med-LangChain
    """
    # 初始化數位孿生系統
    digital_twin_system = MedicalDigitalTwinSystem()
    
    # 創建 LangChain 工具
    langchain_tools = DigitalTwinLangChainTools(digital_twin_system)
    
    # 獲取工具
    medical_analysis_tool = langchain_tools.create_medical_analysis_tool()
    treatment_simulation_tool = langchain_tools.create_treatment_simulation_tool()
    
    # 整合集體記憶系統
    from agent.collective.collective_memory import CollectiveMemoryAgent
    collective_agent = CollectiveMemoryAgent(team_id)
    
    return {
        'digital_twin_system': digital_twin_system,
        'tools': [medical_analysis_tool, treatment_simulation_tool],
        'collective_agent': collective_agent
    }

# 使用範例
if __name__ == "__main__":
    print("🏥 醫療數位孿生系統演示")
    
    # 初始化系統
    medical_system = integrate_digital_twin_system()
    digital_twin_system = medical_system['digital_twin_system']
    
    # 創建病患數位孿生
    patient_data = {
        'patient_id': 'demo_patient_001',
        'heart_rate': 85,
        'blood_pressure': 150,
        'temperature': 36.6,
        'glucose_level': 180,
        'oxygen_saturation': 96,
        'cholesterol': 220
    }
    
    patient_id = digital_twin_system.create_patient_twin(patient_data)
    print(f"✅ 創建病患數位孿生: {patient_id}")
    
    # 分析健康風險
    risk_analysis = digital_twin_system.analyze_patient_risk(patient_id)
    print("🔍 健康風險分析完成")
    print(f"風險等級: {risk_analysis['risk_analysis']['risk_level']}")
    
    # 模擬治療方案
    treatment_plan = {
        'name': '綜合降血壓和血糖治療',
        'medication': [
            {'name': '降壓藥A', 'type': 'antihypertensive', 'dose': 15},
            {'name': '降糖藥B', 'type': 'glucose_control', 'dose': 20}
        ]
    }
    
    simulation = digital_twin_system.simulate_treatment_scenario(patient_id, treatment_plan)
    print("🔬 治療模擬完成")
    print(f"預測結果: {simulation['simulation_result']['predicted_outcome']}")
    
    # 測試 LangChain 工具
    tools = medical_system['tools']
    analysis_result = tools[0].func("分析一位高血壓和糖尿病患者的健康風險")
    print("\n📊 LangChain 分析結果:")
    print(analysis_result)
