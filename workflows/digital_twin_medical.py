"""
數位孿生醫療工作流程
整合：病患監測 + AI預警 + 治療模擬 + 集體記憶
"""

from agent.medical_digital_twin import MedicalDigitalTwinSystem, integrate_digital_twin_system
from agent.collective.collective_memory import CollectiveMemoryAgent
import asyncio
from datetime import datetime

class DigitalTwinMedicalWorkflow:
    """
    數位孿生醫療工作流程
    """
    
    def __init__(self, team_id: str = "hospital_digital_twin"):
        self.medical_system = integrate_digital_twin_system(team_id)
        self.digital_twin_system = self.medical_system['digital_twin_system']
        self.collective_agent = self.medical_system['collective_agent']
        self.monitoring_patients = {}
    
    async start_patient_monitoring(self, patient_data: Dict, user_id: str):
        """
        開始病患即時監測
        """
        print(f"🔍 開始監測病患: {patient_data['patient_id']}")
        
        # 創建數位孿生
        patient_id = self.digital_twin_system.create_patient_twin(patient_data)
        
        # 初始分析
        initial_analysis = self.digital_twin_system.analyze_patient_risk(patient_id)
        
        # 記錄到集體記憶
        workflow_id = self.collective_agent.capture_expert_workflow(
            expert_id=user_id,
            workflow_name=f"病患數位孿生監測 {patient_id}",
            steps=[{
                "step": 1,
                "action": "創建數位孿生",
                "result": f"病患 {patient_id} 數位孿生創建完成"
            }],
            context={
                "patient_data": patient_data,
                "initial_analysis": initial_analysis
            }
        )
        
        # 開始即時監測
        self.monitoring_patients[patient_id] = {
            'workflow_id': workflow_id,
            'last_analysis': initial_analysis,
            'alert_count': 0
        }
        
        return {
            "patient_id": patient_id,
            "workflow_id": workflow_id,
            "initial_analysis": initial_analysis
        }
    
    async def simulate_treatment_options(self, patient_id: str, treatment_options: List[Dict], user_id: str):
        """
        模擬多個治療方案
        """
        print(f"🔬 為病患 {patient_id} 模擬 {len(treatment_options)} 個治療方案")
        
        simulation_results = []
        
        for i, treatment in enumerate(treatment_options, 1):
            # 在數位孿生上模擬治療
            simulation_result = self.digital_twin_system.simulate_treatment_scenario(
                patient_id, treatment
            )
            
            simulation_results.append({
                "treatment_option": i,
                "treatment_name": treatment['name'],
                "simulation_result": simulation_result
            })
            
            # 記錄到集體記憶
            self.collective_agent.add_workflow_step(
                workflow_id=self.monitoring_patients[patient_id]['workflow_id'],
                step={
                    "step": len(simulation_results) + 1,
                    "action": f"模擬治療方案: {treatment['name']}",
                    "result": simulation_result
                }
            )
        
        # 找出最佳治療方案
        best_treatment = self._select_best_treatment(simulation_results)
        
        return {
            "patient_id": patient_id,
            "simulation_results": simulation_results,
            "recommended_treatment": best_treatment
        }
    
    def _select_best_treatment(self, simulation_results: List[Dict]) -> Dict:
        """選擇最佳治療方案"""
        best_score = -1
        best_treatment = None
        
        for result in simulation_results:
            sim_result = result['simulation_result']['simulation_result']
            risk_reduction = sim_result.get('risk_reduction', 0)
            
            if risk_reduction > best_score:
                best_score = risk_reduction
                best_treatment = result
        
        return best_treatment
    
    async def generate_medical_report(self, patient_id: str) -> Dict:
        """
        生成完整醫療報告
        """
        # 獲取當前健康狀態
        current_analysis = self.digital_twin_system.analyze_patient_risk(patient_id)
        
        # 獲取時間線數據
        timeline = self.digital_twin_system.get_patient_timeline(patient_id)
        
        # 生成報告
        report = {
            "patient_id": patient_id,
            "report_date": datetime.now(),
            "executive_summary": self._generate_executive_summary(current_analysis),
            "current_health_status": current_analysis,
            "health_timeline": timeline,
            "treatment_recommendations": self._generate_treatment_recommendations(current_analysis),
            "monitoring_plan": self._generate_monitoring_plan(current_analysis)
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        """生成執行摘要"""
        risk_level = analysis['risk_analysis']['risk_level']
        critical_alerts = analysis['risk_analysis']['critical_alerts']
        
        summary = f"病患當前健康風險等級: {risk_level}\n"
        
        if critical_alerts:
            summary += f"重要警報: {len(critical_alerts)} 個需要立即關注的問題\n"
        else:
            summary += "目前無緊急警報\n"
            
        summary += f"建議: {analysis['recommendations'][0] if analysis['recommendations'] else '定期監測'}"
        
        return summary
    
    def _generate_treatment_recommendations(self, analysis: Dict) -> List[Dict]:
        """生成治療建議"""
        recommendations = []
        
        if analysis['risk_analysis']['cardiovascular_risk'] > 0.6:
            recommendations.append({
                "type": "cardiovascular",
                "priority": "high",
                "recommendation": "啟動降血壓治療方案",
                "medications": ["ACE抑制劑", "β受體阻斷劑"]
            })
        
        if analysis['risk_analysis']['diabetes_risk'] > 0.6:
            recommendations.append({
                "type": "diabetes",
                "priority": "high", 
                "recommendation": "血糖控制治療",
                "medications": ["Metformin", "胰島素"]
            })
        
        return recommendations
    
    def _generate_monitoring_plan(self, analysis: Dict) -> Dict:
        """生成監測計劃"""
        risk_level = analysis['risk_analysis']['risk_level']
        
        if risk_level == "高風險":
            return {
                "frequency": "每日",
                "parameters": ["血壓", "血糖", "心率", "血氧"],
                "alerts": "即時通知"
            }
        elif risk_level == "中風險":
            return {
                "frequency": "每週3次", 
                "parameters": ["血壓", "血糖"],
                "alerts": "每日彙總"
            }
        else:
            return {
                "frequency": "每週1次",
                "parameters": ["血壓", "血糖"],
                "alerts": "每週檢查"
            }

# 使用範例
async def demo_digital_twin_workflow():
    """演示數位孿生工作流程"""
    print("🚀 啟動數位孿生醫療工作流程演示")
    
    workflow = DigitalTwinMedicalWorkflow()
    
    # 病患數據
    patient_data = {
        'patient_id': 'demo_hypertension_patient',
        'heart_rate': 95,
        'blood_pressure': 165,
        'glucose_level': 210,
        'oxygen_saturation': 95
    }
    
    # 開始監測
    monitoring_result = await workflow.start_patient_monitoring(
        patient_data, "doctor_zhang"
    )
    
    print(f"✅ 監測啟動: {monitoring_result['patient_id']}")
    
    # 治療方案模擬
    treatment_options = [
        {
            'name': '標準降血壓治療',
            'medication': [
                {'name': '降壓藥A', 'type': 'antihypertensive', 'dose': 10}
            ]
        },
        {
            'name': '強化綜合治療', 
            'medication': [
                {'name': '降壓藥A', 'type': 'antihypertensive', 'dose': 15},
                {'name': '降糖藥B', 'type': 'glucose_control', 'dose': 25}
            ]
        }
    ]
    
    simulation_results = await workflow.simulate_treatment_options(
        monitoring_result['patient_id'], treatment_options, "doctor_zhang"
    )
    
    print(f"🎯 推薦治療方案: {simulation_results['recommended_treatment']['treatment_name']}")
    
    # 生成報告
    medical_report = await workflow.generate_medical_report(monitoring_result['patient_id'])
    print(f"📋 醫療報告生成完成: {medical_report['executive_summary']}")

if __name__ == "__main__":
    asyncio.run(demo_digital_twin_workflow())
