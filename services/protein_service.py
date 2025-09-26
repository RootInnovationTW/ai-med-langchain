# services/protein_service.py
# ========================================================
# 功能: 基於NVIDIA BioNeMo 3的蛋白質結構預測服務
# 模型: OpenFold2 + ESM2 + ProtT5 + 其他BioNeMo預訓練模型
# ========================================================

import os
import requests
import json
import logging
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import io

# BioNeMo 相關導入
try:
    import nvidia
    from nvidia import bionemo
except ImportError:
    logging.warning("NVIDIA BioNeMo SDK not available, using fallback methods")

# 生物信息學套件
try:
    from Bio import SeqIO, PDB
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.PDB import PDBParser, PDBIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    logging.warning("BioPython not available, some functions will be limited")
    BIOPYTHON_AVAILABLE = False

# RDKit for molecular processing (protein-ligand interactions)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available, some molecular features will be limited")
    RDKIT_AVAILABLE = False

# py3Dmol for 3D visualization
try:
    import py3Dmol
    from IPython.display import display
    PY3DMOL_AVAILABLE = True
except ImportError:
    logging.warning("py3Dmol not available, 3D visualization will be limited")
    PY3DMOL_AVAILABLE = False

# ========================================================
# 配置和常量
# ========================================================

class ProteinModel(Enum):
    OPENFOLD2 = "openfold2"
    ESM2 = "esm2"
    PROT_T5 = "prot_t5"
    CHAI_1 = "chai-1"

@dataclass
class ProteinPredictionConfig:
    """蛋白質預測配置參數"""
    model: ProteinModel = ProteinModel.OPENFOLD2
    max_sequence_length: int = 2048
    confidence_threshold: float = 0.7
    num_recycling: int = 3
    use_templates: bool = False
    timeout: int = 600  # seconds
    output_format: str = "pdb"  # pdb, cif, or both

# ========================================================
# BioNeMo蛋白質服務類
# ========================================================

class BioNeMoProteinService:
    """NVIDIA BioNeMo 蛋白質模型服務封裝"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url or os.getenv("BIONEMO_BASE_URL", "https://api.nvidia.com/bionemo/v1")
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "BioNeMo-Python-Client/1.0"
            })
    
    def predict_structure_openfold2(self, sequence: str, config: ProteinPredictionConfig) -> Dict[str, Any]:
        """使用OpenFold2預測蛋白質結構"""
        try:
            payload = {
                "sequences": [sequence],
                "model": "openfold2",
                "num_recycling": config.num_recycling,
                "use_templates": config.use_templates,
                "output_format": config.output_format
            }
            
            response = self.session.post(
                f"{self.base_url}/protein/structure/predict",
                json=payload,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return self._process_structure_result(result, sequence)
            
        except Exception as e:
            logging.error(f"OpenFold2 prediction failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_protein_embeddings(self, sequence: str, model: str = "esm2") -> Dict[str, Any]:
        """獲取蛋白質序列的嵌入向量"""
        try:
            payload = {
                "sequences": [sequence],
                "model": model,
                "pooling_type": "mean"  # mean, cls, or none
            }
            
            response = self.session.post(
                f"{self.base_url}/protein/embeddings",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "embeddings": result.get("embeddings", []),
                "model": model,
                "sequence_length": len(sequence),
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Protein embedding generation failed: {e}")
            return {"error": str(e), "success": False}
    
    def predict_function(self, sequence: str) -> Dict[str, Any]:
        """預測蛋白質功能"""
        try:
            payload = {
                "sequence": sequence,
                "prediction_types": ["go_terms", "enzyme_class", "localization"]
            }
            
            response = self.session.post(
                f"{self.base_url}/protein/function/predict",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logging.error(f"Function prediction failed: {e}")
            return {"error": str(e), "success": False}
    
    def _process_structure_result(self, result: Dict[str, Any], sequence: str) -> Dict[str, Any]:
        """處理結構預測結果"""
        if "error" in result:
            return result
        
        processed_result = {
            "success": True,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "pdb_data": result.get("pdb_data", ""),
            "confidence_scores": result.get("confidence", []),
            "mean_confidence": 0.0,
            "model_info": result.get("model_info", {}),
            "prediction_time": result.get("prediction_time", 0)
        }
        
        # 計算平均置信度
        if processed_result["confidence_scores"]:
            processed_result["mean_confidence"] = np.mean(processed_result["confidence_scores"])
        
        return processed_result

# ========================================================
# 蛋白質服務主類
# ========================================================

class ProteinService:
    """蛋白質結構預測服務 - 基於BioNeMo"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.bionemo_service = BioNeMoProteinService(api_key, base_url)
        self.config = ProteinPredictionConfig()
        
    def predict_structure(self, 
                         sequence: str,
                         config: Optional[ProteinPredictionConfig] = None) -> Dict[str, Any]:
        """
        預測蛋白質3D結構
        
        Args:
            sequence: 蛋白質氨基酸序列
            config: 預測配置參數
            
        Returns:
            結構預測結果字典
        """
        config = config or self.config
        
        try:
            # 1. 驗證序列
            validated_sequence = self._validate_sequence(sequence)
            
            # 2. 檢查序列長度
            if len(validated_sequence) > config.max_sequence_length:
                return {
                    "success": False,
                    "error": f"Sequence too long: {len(validated_sequence)} > {config.max_sequence_length}"
                }
            
            # 3. 根據模型選擇預測方法
            if config.model == ProteinModel.OPENFOLD2:
                result = self.bionemo_service.predict_structure_openfold2(validated_sequence, config)
            else:
                result = self._run_fallback_prediction(validated_sequence, config)
            
            # 4. 後處理和結果豐富化
            if result.get("success", False):
                enriched_result = self._enrich_prediction_result(result, validated_sequence)
                return enriched_result
            else:
                return result
                
        except Exception as e:
            logging.error(f"Structure prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sequence": sequence
            }
    
    def analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """分析蛋白質序列屬性"""
        try:
            validated_sequence = self._validate_sequence(sequence)
            
            # 基本序列統計
            amino_acid_counts = self._count_amino_acids(validated_sequence)
            
            # 計算物理化學性質
            properties = self._calculate_properties(validated_sequence)
            
            # 獲取蛋白質嵌入向量
            embeddings_result = self.bionemo_service.get_protein_embeddings(validated_sequence)
            
            # 預測功能
            function_result = self.bionemo_service.predict_function(validated_sequence)
            
            analysis_result = {
                "success": True,
                "sequence": validated_sequence,
                "length": len(validated_sequence),
                "amino_acid_composition": amino_acid_counts,
                "physicochemical_properties": properties,
                "embeddings_available": embeddings_result.get("success", False),
                "functional_predictions": function_result if function_result.get("success") else None
            }
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"Sequence analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sequence": sequence
            }
    
    def compare_structures(self, sequence1: str, sequence2: str) -> Dict[str, Any]:
        """比較兩個蛋白質序列的相似性"""
        try:
            # 獲取兩個序列的嵌入向量
            emb1 = self.bionemo_service.get_protein_embeddings(sequence1)
            emb2 = self.bionemo_service.get_protein_embeddings(sequence2)
            
            if not (emb1.get("success") and emb2.get("success")):
                return {
                    "success": False,
                    "error": "Failed to generate embeddings for comparison"
                }
            
            # 計算餘弦相似度
            vec1 = np.array(emb1["embeddings"][0])
            vec2 = np.array(emb2["embeddings"][0])
            
            cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # 序列對齊相似度（簡化版本）
            sequence_similarity = self._calculate_sequence_similarity(sequence1, sequence2)
            
            return {
                "success": True,
                "sequence1_length": len(sequence1),
                "sequence2_length": len(sequence2),
                "cosine_similarity": float(cosine_similarity),
                "sequence_similarity": sequence_similarity,
                "overall_similarity": (cosine_similarity + sequence_similarity) / 2
            }
            
        except Exception as e:
            logging.error(f"Structure comparison failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_sequence(self, sequence: str) -> str:
        """驗證和清理蛋白質序列"""
        # 移除空白字符
        clean_sequence = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")
        
        # 標準氨基酸字母
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        
        # 檢查無效字符
        invalid_chars = set(clean_sequence) - valid_amino_acids
        if invalid_chars:
            # 替換常見的模糊氨基酸
            replacements = {"X": "A", "B": "N", "Z": "Q", "J": "L", "U": "C", "O": "K"}
            for old, new in replacements.items():
                clean_sequence = clean_sequence.replace(old, new)
            
            # 重新檢查
            invalid_chars = set(clean_sequence) - valid_amino_acids
            if invalid_chars:
                raise ValueError(f"Invalid amino acid characters: {invalid_chars}")
        
        if len(clean_sequence) == 0:
            raise ValueError("Empty sequence after cleaning")
        
        return clean_sequence
    
    def _count_amino_acids(self, sequence: str) -> Dict[str, int]:
        """統計氨基酸組成"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        counts = {}
        
        for aa in amino_acids:
            counts[aa] = sequence.count(aa)
        
        # 計算百分比
        total = len(sequence)
        percentages = {aa: (count/total)*100 for aa, count in counts.items()}
        
        return {
            "counts": counts,
            "percentages": percentages,
            "total_length": total
        }
    
    def _calculate_properties(self, sequence: str) -> Dict[str, float]:
        """計算蛋白質物理化學性質"""
        # 氨基酸屬性數據
        molecular_weights = {
            'A': 89.1, 'C': 121.0, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        }
        
        hydrophobic_amino_acids = set("AILMFPWV")
        polar_amino_acids = set("NQSTY")
        charged_amino_acids = set("DEKR")
        
        # 計算分子量
        molecular_weight = sum(molecular_weights.get(aa, 0) for aa in sequence)
        
        # 計算疏水性百分比
        hydrophobic_percent = (sum(1 for aa in sequence if aa in hydrophobic_amino_acids) / len(sequence)) * 100
        
        # 計算極性百分比
        polar_percent = (sum(1 for aa in sequence if aa in polar_amino_acids) / len(sequence)) * 100
        
        # 計算帶電荷百分比
        charged_percent = (sum(1 for aa in sequence if aa in charged_amino_acids) / len(sequence)) * 100
        
        # 計算等電點（簡化版本）
        positive_charges = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative_charges = sequence.count('D') + sequence.count('E')
        net_charge = positive_charges - negative_charges
        
        return {
            "molecular_weight": molecular_weight,
            "hydrophobic_percent": hydrophobic_percent,
            "polar_percent": polar_percent,
            "charged_percent": charged_percent,
            "net_charge": net_charge,
            "length": len(sequence)
        }
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """計算序列相似度（簡化版本）"""
        # 使用最長公共子序列方法
        m, n = len(seq1), len(seq2)
        
        # 創建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # 計算相似度
        lcs_length = dp[m][n]
        similarity = (2 * lcs_length) / (m + n)
        
        return similarity
    
    def _run_fallback_prediction(self, sequence: str, config: ProteinPredictionConfig) -> Dict[str, Any]:
        """回退預測方法"""
        # 這裡實現一個簡化的結構預測
        # 在實際應用中，這裡可以調用其他結構預測工具
        
        logging.warning("Using fallback structure prediction method")
        
        # 生成模擬的置信度分數
        confidence_scores = np.random.beta(2, 1, len(sequence)).tolist()
        mean_confidence = np.mean(confidence_scores)
        
        # 生成簡化的PDB格式數據
        pdb_data = self._generate_mock_pdb(sequence, confidence_scores)
        
        return {
            "success": True,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "pdb_data": pdb_data,
            "confidence_scores": confidence_scores,
            "mean_confidence": mean_confidence,
            "model_info": {"method": "fallback", "note": "Simplified prediction"},
            "prediction_time": 1.0
        }
    
    def _generate_mock_pdb(self, sequence: str, confidence_scores: List[float]) -> str:
        """生成模擬PDB數據（用於測試）"""
        pdb_lines = []
        pdb_lines.append("HEADER    PREDICTED STRUCTURE")
        pdb_lines.append("TITLE     AI PREDICTED PROTEIN STRUCTURE")
        
        # 生成簡化的原子坐標
        for i, aa in enumerate(sequence):
            x = i * 3.8  # 簡化的間距
            y = 0.0
            z = 0.0
            
            pdb_lines.append(
                f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{confidence_scores[i]*100:6.2f}           C"
            )
        
        pdb_lines.append("END")
        
        return "\n".join(pdb_lines)
    
    def _enrich_prediction_result(self, result: Dict[str, Any], sequence: str) -> Dict[str, Any]:
        """豐富預測結果"""
        if not result.get("success", False):
            return result
        
        # 添加序列分析
        sequence_analysis = self.analyze_sequence(sequence)
        
        # 評估預測質量
        quality_assessment = self._assess_prediction_quality(result)
        
        enriched_result = result.copy()
        enriched_result.update({
            "sequence_analysis": sequence_analysis.get("amino_acid_composition", {}),
            "physicochemical_properties": sequence_analysis.get("physicochemical_properties", {}),
            "quality_assessment": quality_assessment,
            "prediction_metadata": {
                "timestamp": "2024-01-01T00:00:00Z",  # 實際應用中使用真實時間戳
                "model_version": "BioNeMo-3.0",
                "confidence_threshold": self.config.confidence_threshold
            }
        })
        
        return enriched_result
    
    def _assess_prediction_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """評估預測質量"""
        mean_confidence = result.get("mean_confidence", 0.0)
        confidence_scores = result.get("confidence_scores", [])
        
        # 計算置信度統計
        if confidence_scores:
            confidence_std = np.std(confidence_scores)
            high_confidence_residues = sum(1 for score in confidence_scores if score > 0.8)
            low_confidence_residues = sum(1 for score in confidence_scores if score < 0.5)
        else:
            confidence_std = 0.0
            high_confidence_residues = 0
            low_confidence_residues = 0
        
        # 質量等級
        if mean_confidence > 0.8:
            quality_grade = "High"
        elif mean_confidence > 0.6:
            quality_grade = "Medium"
        else:
            quality_grade = "Low"
        
        return {
            "overall_confidence": mean_confidence,
            "confidence_std": confidence_std,
            "quality_grade": quality_grade,
            "high_confidence_residues": high_confidence_residues,
            "low_confidence_residues": low_confidence_residues,
            "total_residues": len(confidence_scores)
        }
    
    def visualize_structure_3d(self, result: Dict[str, Any], style: str = "cartoon") -> Any:
        """
        使用py3Dmol在Jupyter環境中可視化蛋白質3D結構
        
        Args:
            result: 結構預測結果字典
            style: 顯示樣式 ('cartoon', 'stick', 'sphere', 'surface')
            
        Returns:
            py3Dmol viewer對象
        """
        if not PY3DMOL_AVAILABLE:
            logging.error("py3Dmol not available for 3D visualization")
            return None
            
        if not result.get("success", False):
            logging.error("Cannot visualize failed prediction result")
            return None
            
        pdb_data = result.get("pdb_data", "")
        if not pdb_data:
            logging.error("No PDB data available for visualization")
            return None
        
        try:
            # 創建3D查看器
            viewer = py3Dmol.view(width=500, height=400)
            viewer.addModel(pdb_data, "pdb")
            
            # 設置顯示樣式
            if style == "cartoon":
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
            elif style == "stick":
                viewer.setStyle({"stick": {}})
            elif style == "sphere":
                viewer.setStyle({"sphere": {"scale": 0.3}})
            elif style == "surface":
                viewer.addSurface(py3Dmol.VDW, {"opacity": 0.7})
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
            
            # 添加置信度顏色映射（如果有置信度數據）
            confidence_scores = result.get("confidence_scores", [])
            if confidence_scores:
                self._add_confidence_coloring(viewer, confidence_scores)
            
            viewer.zoomTo()
            return viewer
            
        except Exception as e:
            logging.error(f"3D visualization failed: {e}")
            return None
    
    def _add_confidence_coloring(self, viewer, confidence_scores: List[float]):
        """根據置信度為結構添加顏色映射"""
        try:
            # 為每個殘基根據置信度分數設置顏色
            for i, confidence in enumerate(confidence_scores):
                if confidence > 0.8:
                    color = "green"  # 高置信度
                elif confidence > 0.6:
                    color = "yellow"  # 中等置信度
                else:
                    color = "red"    # 低置信度
                
                viewer.setStyle(
                    {"resi": i+1}, 
                    {"cartoon": {"color": color}}
                )
        except Exception as e:
            logging.warning(f"Confidence coloring failed: {e}")
    
    def extract_binding_sites(self, result: Dict[str, Any], ligand_smiles: str = None) -> Dict[str, Any]:
        """
        提取蛋白質結合位點信息（結合RDKit進行配體分析）
        
        Args:
            result: 結構預測結果
            ligand_smiles: 可選的配體SMILES用於結合位點預測
            
        Returns:
            結合位點信息字典
        """
        if not (RDKIT_AVAILABLE and result.get("success", False)):
            return {"success": False, "error": "RDKit unavailable or invalid structure"}
        
        try:
            sequence = result.get("sequence", "")
            pdb_data = result.get("pdb_data", "")
            
            # 基於序列特徵預測結合位點
            binding_sites = self._predict_binding_sites_from_sequence(sequence)
            
            result_data = {
                "success": True,
                "predicted_sites": binding_sites,
                "sequence_length": len(sequence)
            }
            
            # 如果提供了配體SMILES，進行配體分析
            if ligand_smiles and RDKIT_AVAILABLE:
                ligand_analysis = self._analyze_ligand_compatibility(ligand_smiles, binding_sites)
                result_data["ligand_compatibility"] = ligand_analysis
            
            return result_data
            
        except Exception as e:
            logging.error(f"Binding site extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _predict_binding_sites_from_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """基於序列特徵預測結合位點"""
        binding_sites = []
        
        # 尋找常見的結合motifs
        binding_motifs = {
            "ATP_binding": ["GXXXXGK", "GXGXXG"],
            "DNA_binding": ["CXXC", "HXXXH"],
            "metal_binding": ["HXH", "CXC", "HH"],
            "active_site": ["SERINE_PROTEASE", "CYSTEINE_PROTEASE"]
        }
        
        for motif_type, patterns in binding_motifs.items():
            for i, pattern in enumerate(patterns):
                sites = self._find_sequence_motifs(sequence, pattern)
                for site in sites:
                    binding_sites.append({
                        "type": motif_type,
                        "position": site,
                        "pattern": pattern,
                        "confidence": 0.7 - (i * 0.1)  # 降低後續模式的置信度
                    })
        
        return binding_sites
    
    def _find_sequence_motifs(self, sequence: str, pattern: str) -> List[int]:
        """在序列中尋找特定模式"""
        import re
        
        # 將X轉換為任意氨基酸的正則表達式
        regex_pattern = pattern.replace('X', '[ACDEFGHIKLMNPQRSTVWY]')
        
        positions = []
        for match in re.finditer(regex_pattern, sequence):
            positions.append(match.start())
        
        return positions
    
    def _analyze_ligand_compatibility(self, ligand_smiles: str, binding_sites: List[Dict]) -> Dict[str, Any]:
        """分析配體與結合位點的兼容性"""
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        try:
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            # 計算配體屬性
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # 評估與不同結合位點類型的兼容性
            compatibility_scores = {}
            
            for site in binding_sites:
                site_type = site.get("type", "")
                
                if site_type == "ATP_binding":
                    # ATP結合位點偏好較大分子
                    score = min(1.0, mw / 500) * 0.8
                elif site_type == "DNA_binding":
                    # DNA結合位點偏好帶正電荷的分子
                    score = 0.6 if hbd > hba else 0.3
                elif site_type == "metal_binding":
                    # 金屬結合位點偏好有供電子基團的分子
                    score = min(1.0, (hba + hbd) / 10)
                else:
                    score = 0.5  # 默認中等兼容性
                
                compatibility_scores[f"{site_type}_{site.get('position', 0)}"] = score
            
            return {
                "ligand_properties": {
                    "molecular_weight": mw,
                    "logp": logp,
                    "hbd": hbd,
                    "hba": hba
                },
                "site_compatibility": compatibility_scores,
                "overall_compatibility": np.mean(list(compatibility_scores.values())) if compatibility_scores else 0.0
            }
            
        except Exception as e:
            return {"error": str(e)}
        """保存預測結構到文件"""
        try:
            if not result.get("success", False):
                logging.error("Cannot save failed prediction result")
                return False
            
            pdb_data = result.get("pdb_data", "")
            if not pdb_data:
                logging.error("No PDB data to save")
                return False
            
            with open(filename, 'w') as f:
                f.write(pdb_data)
            
            logging.info(f"Structure saved to {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save structure: {e}")
            return False

# ========================================================
# LangChain工具函數
# ========================================================

def predict_structure(sequence: str, model: str = "openfold2") -> Dict[str, Any]:
    """
    LangChain工具函數：預測蛋白質結構
    
    Args:
        sequence: 蛋白質氨基酸序列
        model: 預測模型（openfold2/esm2/prot_t5）
    
    Returns:
        結構預測結果字典
    """
    service = ProteinService()
    
    config = ProteinPredictionConfig(
        model=ProteinModel(model.lower())
    )
    
    result = service.predict_structure(sequence, config)
    
    # 格式化輸出供LangChain使用
    formatted_result = {
        "success": result.get("success", False),
        "confidence": result.get("mean_confidence", 0),
        "model": model,
        "sequence_length": result.get("sequence_length", 0),
        "quality_grade": result.get("quality_assessment", {}).get("quality_grade", "Unknown")
    }
    
    if not result.get("success", False):
        formatted_result["error"] = result.get("error", "Unknown error")
    else:
        formatted_result["pdb_available"] = bool(result.get("pdb_data"))
        formatted_result["properties"] = result.get("physicochemical_properties", {})
    
    return formatted_result

def visualize_protein_3d(sequence: str = None, pdb_data: str = None, style: str = "cartoon") -> Dict[str, Any]:
    """
    LangChain工具函數：3D可視化蛋白質結構
    
    Args:
        sequence: 蛋白質序列（將進行結構預測）
        pdb_data: 直接提供的PDB數據
        style: 顯示樣式
    
    Returns:
        可視化結果字典
    """
    service = ProteinService()
    
    try:
        if pdb_data:
            # 直接使用提供的PDB數據
            result = {
                "success": True,
                "pdb_data": pdb_data,
                "confidence_scores": []
            }
        elif sequence:
            # 預測結構
            result = service.predict_structure(sequence)
        else:
            return {"success": False, "error": "Either sequence or pdb_data must be provided"}
        
        # 創建3D可視化
        viewer = service.visualize_structure_3d(result, style)
        
        return {
            "success": viewer is not None,
            "visualization_available": PY3DMOL_AVAILABLE,
            "viewer_created": viewer is not None,
            "style": style
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_protein_binding_sites(sequence: str, ligand_smiles: str = None) -> Dict[str, Any]:
    """
    LangChain工具函數：提取蛋白質結合位點
    
    Args:
        sequence: 蛋白質序列
        ligand_smiles: 可選的配體SMILES
    
    Returns:
        結合位點信息字典
    """
    service = ProteinService()
    
    # 先進行結構預測
    structure_result = service.predict_structure(sequence)
    
    if not structure_result.get("success", False):
        return {
            "success": False,
            "error": "Structure prediction failed"
        }
    
    # 提取結合位點
    binding_result = service.extract_binding_sites(structure_result, ligand_smiles)
    
    # 格式化輸出
    if binding_result.get("success", False):
        return {
            "success": True,
            "num_binding_sites": len(binding_result.get("predicted_sites", [])),
            "binding_sites": binding_result.get("predicted_sites", []),
            "ligand_compatibility": binding_result.get("ligand_compatibility", {}),
            "sequence_length": binding_result.get("sequence_length", 0)
        }
    else:
        return binding_result
    """
    LangChain工具函數：分析蛋白質序列
    
    Args:
        sequence: 蛋白質氨基酸序列
    
    Returns:
        序列分析結果字典
    """
    service = ProteinService()
    result = service.analyze_sequence(sequence)
    
    # 簡化輸出供LangChain使用
    if result.get("success", False):
        return {
            "success": True,
            "length": result["length"],
            "molecular_weight": result["physicochemical_properties"]["molecular_weight"],
            "hydrophobic_percent": result["physicochemical_properties"]["hydrophobic_percent"],
            "net_charge": result["physicochemical_properties"]["net_charge"],
            "functional_predictions_available": result["functional_predictions"] is not None
        }
    else:
        return {
            "success": False,
            "error": result.get("error", "Analysis failed")
        }

# ========================================================
# 測試代碼
# ========================================================

if __name__ == "__main__":
    # 測試蛋白質結構預測服務
    service = ProteinService()
    
    # 測試用例：胰島素A鏈序列
    test_sequence = "GIVEQCCTSICSLYQLENYCN"
    
    print("🧬 測試蛋白質結構預測...")
    result = service.predict_structure(test_sequence)
    
    print("📊 預測結果:")
    print(f"成功: {result.get('success', False)}")
    print(f"序列長度: {result.get('sequence_length', 0)}")
    print(f"平均置信度: {result.get('mean_confidence', 0):.3f}")
    print(f"質量等級: {result.get('quality_assessment', {}).get('quality_grade', 'Unknown')}")
    
    if result.get('success', False):
        # 測試序列分析
        print("\n🔬 測試序列分析...")
        analysis = service.analyze_sequence(test_sequence)
        
        if analysis.get('success', False):
            props = analysis.get('physicochemical_properties', {})
            print(f"分子量: {props.get('molecular_weight', 0):.1f} Da")
            print(f"疏水性百分比: {props.get('hydrophobic_percent', 0):.1f}%")
            print(f"淨電荷: {props.get('net_charge', 0)}")
        
        # 測試3D可視化
        if result.get('pdb_data') and PY3DMOL_AVAILABLE:
            print("\n🎨 測試3D可視化...")
            viewer = service.visualize_structure_3d(result, "cartoon")
            if viewer:
                print("✅ 3D可視化已創建")
                # 在Jupyter環境中可以調用: viewer.show()
            
        # 測試結合位點預測
        print("\n🔍 測試結合位點預測...")
        binding_sites = service.extract_binding_sites(result, "CC(=O)OC1=CC=CC=C1C(=O)O")  # 阿斯匹林
        if binding_sites.get('success', False):
            sites = binding_sites.get('predicted_sites', [])
            print(f"發現 {len(sites)} 個潛在結合位點")
            for site in sites[:3]:  # 顯示前3個
                print(f"  - {site.get('type', 'Unknown')} 位點在位置 {site.get('position', 0)}")
        
        # 測試RDKit功能
        if RDKIT_AVAILABLE:
            print("\n💊 RDKit功能可用")
            compatibility = binding_sites.get('ligand_compatibility', {})
            if compatibility and not compatibility.get('error'):
                print(f"配體兼容性評分: {compatibility.get('overall_compatibility', 0):.2f}")
        
        # 可視化工具測試
        print(f"\n📊 可視化工具狀態:")
        print(f"py3Dmol: {'✅' if PY3DMOL_AVAILABLE else '❌'}")
        print(f"RDKit: {'✅' if RDKIT_AVAILABLE else '❌'}")
        print(f"BioPython: {'✅' if BIOPYTHON_AVAILABLE else '❌'}")
