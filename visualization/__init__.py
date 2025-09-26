# visualization/__init__.py
# ========================================================
# 功能: 統一的可視化模組接口
# 整合: py3Dmol + matplotlib + RDKit + plotly
# 支持: 蛋白質結構、分子、對接結果、數據分析可視化
# ========================================================

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# 核心可視化庫
try:
    import py3Dmol
    from IPython.display import display, HTML
    PY3DMOL_AVAILABLE = True
except ImportError:
    logging.warning("py3Dmol not available, 3D molecular visualization will be limited")
    PY3DMOL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("matplotlib/seaborn not available, some plotting features will be limited")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("plotly not available, interactive plots will be limited")
    PLOTLY_AVAILABLE = False

# RDKit for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available, molecular drawing will be limited")
    RDKIT_AVAILABLE = False

# 導入服務模組
try:
    from ..services.protein_service import ProteinService
    from ..services.molecule_service import MoleculeService  
    from ..services.docking_service import DockingService
    SERVICES_AVAILABLE = True
except ImportError:
    logging.warning("Service modules not available, some features will be limited")
    SERVICES_AVAILABLE = False

# ========================================================
# 可視化管理器主類
# ========================================================

class VisualizationManager:
    """統一的可視化管理器"""
    
    def __init__(self):
        self.available_backends = {
            "py3dmol": PY3DMOL_AVAILABLE,
            "matplotlib": MATPLOTLIB_AVAILABLE, 
            "plotly": PLOTLY_AVAILABLE,
            "rdkit": RDKIT_AVAILABLE
        }
        
        # 設置可視化主題
        self._setup_themes()
        
    def _setup_themes(self):
        """設置可視化主題和樣式"""
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        # 自定義顏色方案
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72", 
            "accent": "#F18F01",
            "success": "#C73E1D",
            "protein": "#1f77b4",
            "ligand": "#ff7f0e",
            "interaction": "#2ca02c",
            "high_confidence": "#008000",
            "medium_confidence": "#FFD700",
            "low_confidence": "#FF4500"
        }
    
    def get_capabilities(self) -> Dict[str, bool]:
        """返回可用的可視化能力"""
        return self.available_backends.copy()
    
    # ========================================================
    # 蛋白質結構可視化
    # ========================================================
    
    def visualize_protein_structure_3d(self, 
                                     structure_result: Dict[str, Any],
                                     style: str = "cartoon",
                                     color_by_confidence: bool = True,
                                     width: int = 600,
                                     height: int = 400) -> Any:
        """
        3D蛋白質結構可視化
        
        Args:
            structure_result: 蛋白質結構預測結果
            style: 顯示樣式 ('cartoon', 'stick', 'sphere', 'surface')
            color_by_confidence: 是否根據置信度著色
            width, height: 視圖大小
            
        Returns:
            py3Dmol viewer對象或錯誤信息
        """
        if not PY3DMOL_AVAILABLE:
            return {"error": "py3Dmol not available"}
        
        if not structure_result.get("success", False):
            return {"error": "Invalid structure data"}
        
        pdb_data = structure_result.get("pdb_data", "")
        if not pdb_data:
            return {"error": "No PDB data available"}
        
        try:
            viewer = py3Dmol.view(width=width, height=height)
            viewer.addModel(pdb_data, "pdb")
            
            # 設置基本樣式
            if style == "cartoon":
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
            elif style == "stick":
                viewer.setStyle({"stick": {"radius": 0.3}})
            elif style == "sphere":
                viewer.setStyle({"sphere": {"scale": 0.3, "colorscheme": "Jmol"}})
            elif style == "surface":
                viewer.addSurface(py3Dmol.VDW, {"opacity": 0.8, "color": "white"})
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
            
            # 根據置信度著色
            if color_by_confidence and structure_result.get("confidence_scores"):
                self._apply_confidence_coloring(viewer, structure_result["confidence_scores"])
            
            viewer.zoomTo()
            viewer.spin(False)  # 禁用自動旋轉
            
            return viewer
            
        except Exception as e:
            logging.error(f"3D protein visualization failed: {e}")
            return {"error": str(e)}
    
    def _apply_confidence_coloring(self, viewer, confidence_scores: List[float]):
        """應用置信度顏色映射"""
        for i, confidence in enumerate(confidence_scores):
            if confidence > 0.8:
                color = self.colors["high_confidence"]
            elif confidence > 0.6:
                color = self.colors["medium_confidence"] 
            else:
                color = self.colors["low_confidence"]
            
            viewer.setStyle(
                {"resi": i+1},
                {"cartoon": {"color": color}}
            )
    
    def plot_confidence_distribution(self, 
                                   structure_result: Dict[str, Any],
                                   save_path: Optional[str] = None) -> Optional[str]:
        """繪製置信度分佈圖"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        confidence_scores = structure_result.get("confidence_scores", [])
        if not confidence_scores:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 置信度序列圖
            ax1.plot(range(1, len(confidence_scores) + 1), confidence_scores, 
                    color=self.colors["protein"], linewidth=2)
            ax1.fill_between(range(1, len(confidence_scores) + 1), confidence_scores, 
                           alpha=0.3, color=self.colors["protein"])
            ax1.axhline(y=0.8, color=self.colors["high_confidence"], 
                       linestyle='--', label='High confidence')
            ax1.axhline(y=0.6, color=self.colors["medium_confidence"], 
                       linestyle='--', label='Medium confidence')
            ax1.set_xlabel('Residue Position')
            ax1.set_ylabel('Confidence Score')
            ax1.set_title('Confidence Scores by Position')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 置信度分佈直方圖
            ax2.hist(confidence_scores, bins=20, color=self.colors["protein"], 
                    alpha=0.7, edgecolor='black')
            ax2.axvline(x=np.mean(confidence_scores), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(confidence_scores):.3f}')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Confidence Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logging.error(f"Confidence plot failed: {e}")
            return None
    
    # ========================================================
    # 分子可視化
    # ========================================================
    
    def visualize_molecule_2d(self, 
                            smiles: str,
                            size: Tuple[int, int] = (300, 300),
                            save_path: Optional[str] = None) -> Optional[Any]:
        """2D分子結構可視化"""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 生成2D坐標
            rdDepictor.Compute2DCoords(mol)
            
            # 創建繪圖器
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # 獲取圖像數據
            img_data = drawer.GetDrawingText()
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(img_data)
            
            return img_data
            
        except Exception as e:
            logging.error(f"2D molecule visualization failed: {e}")
            return None
    
    def visualize_molecule_3d(self, 
                            smiles: str,
                            style: str = "stick",
                            width: int = 400,
                            height: int = 400) -> Any:
        """3D分子結構可視化"""
        if not (PY3DMOL_AVAILABLE and RDKIT_AVAILABLE):
            return {"error": "Required libraries not available"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            # 生成3D構象
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 轉換為PDB格式
            pdb_block = Chem.MolToPDBBlock(mol)
            
            # 創建3D視圖
            viewer = py3Dmol.view(width=width, height=height)
            viewer.addModel(pdb_block, "pdb")
            
            if style == "stick":
                viewer.setStyle({"stick": {"radius": 0.2}})
            elif style == "sphere":
                viewer.setStyle({"sphere": {"scale": 0.3}})
            elif style == "ball_stick":
                viewer.setStyle({"stick": {"radius": 0.1}, "sphere": {"scale": 0.2}})
            
            viewer.zoomTo()
            return viewer
            
        except Exception as e:
            logging.error(f"3D molecule visualization failed: {e}")
            return {"error": str(e)}
    
    def plot_molecular_properties(self, 
                                smiles_list: List[str],
                                property_names: List[str] = None,
                                save_path: Optional[str] = None) -> Optional[str]:
        """繪製分子屬性分佈圖"""
        if not (MATPLOTLIB_AVAILABLE and RDKIT_AVAILABLE):
            return None
        
        if property_names is None:
            property_names = ["MW", "LogP", "HBD", "HBA", "TPSA"]
        
        try:
            # 計算分子屬性
            properties_data = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                props = {
                    "MW": rdMolDescriptors.CalcExactMolWt(mol),
                    "LogP": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "TPSA": rdMolDescriptors.CalcTPSA(mol)
                }
                properties_data.append(props)
            
            if not properties_data:
                return None
            
            # 創建子圖
            n_props = len(property_names)
            fig, axes = plt.subplots(2, (n_props + 1) // 2, figsize=(15, 10))
            axes = axes.flatten() if n_props > 1 else [axes]
            
            for i, prop in enumerate(property_names):
                if i >= len(axes):
                    break
                
                values = [data[prop] for data in properties_data if prop in data]
                if values:
                    axes[i].hist(values, bins=20, alpha=0.7, color=self.colors["primary"])
                    axes[i].set_title(f'{prop} Distribution')
                    axes[i].set_xlabel(prop)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # 隱藏未使用的子圖
            for i in range(len(property_names), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logging.error(f"Molecular properties plot failed: {e}")
            return None
    
    # ========================================================
    # 對接結果可視化
    # ========================================================
    
    def visualize_docking_complex(self,
                                docking_result: Dict[str, Any],
                                protein_structure: Dict[str, Any],
                                ligand_smiles: str,
                                width: int = 600,
                                height: int = 400) -> Any:
        """可視化蛋白質-配體對接複合物"""
        if not PY3DMOL_AVAILABLE:
            return {"error": "py3Dmol not available"}
        
        try:
            viewer = py3Dmol.view(width=width, height=height)
            
            # 添加蛋白質結構
            protein_pdb = protein_structure.get("pdb_data", "")
            if protein_pdb:
                viewer.addModel(protein_pdb, "pdb")
                viewer.setStyle({"model": 0}, {"cartoon": {"color": self.colors["protein"]}})
            
            # 添加配體（如果有對接位置信息）
            if RDKIT_AVAILABLE and docking_result.get("success", False):
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    ligand_pdb = Chem.MolToPDBBlock(mol)
                    
                    viewer.addModel(ligand_pdb, "pdb") 
                    viewer.setStyle({"model": 1}, {
                        "stick": {"color": self.colors["ligand"], "radius": 0.3},
                        "sphere": {"color": self.colors["ligand"], "scale": 0.2}
                    })
            
            viewer.zoomTo()
            return viewer
            
        except Exception as e:
            logging.error(f"Docking complex visualization failed: {e}")
            return {"error": str(e)}
    
    def plot_docking_scores(self,
                          docking_results: List[Dict[str, Any]],
                          method_names: List[str] = None,
                          save_path: Optional[str] = None) -> Optional[str]:
        """繪製對接分數比較圖"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            scores = [result.get("docking_score", 0) for result in docking_results]
            methods = method_names or [f"Method_{i+1}" for i in range(len(scores))]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 對接分數條形圖
            colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
            bars = ax1.bar(methods, scores, color=colors)
            ax1.set_xlabel('Docking Method')
            ax1.set_ylabel('Docking Score (kcal/mol)')
            ax1.set_title('Docking Scores Comparison')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{score:.2f}', ha='center', va='bottom')
            
            # 置信度散點圖
            confidences = [result.get("confidence", 0) for result in docking_results]
            ax2.scatter(scores, confidences, c=colors, s=100, alpha=0.7)
            ax2.set_xlabel('Docking Score (kcal/mol)')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Score vs Confidence')
            
            # 添加方法標籤
            for i, (score, conf, method) in enumerate(zip(scores, confidences, methods)):
                ax2.annotate(method, (score, conf), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logging.error(f"Docking scores plot failed: {e}")
            return None
    
    # ========================================================
    # 交互式可視化 (Plotly)
    # ========================================================
    
    def create_interactive_dashboard(self,
                                   protein_result: Dict[str, Any],
                                   docking_results: List[Dict[str, Any]],
                                   save_path: Optional[str] = None) -> Optional[str]:
        """創建交互式儀表板"""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # 創建子圖佈局
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confidence Scores', 'Docking Scores', 
                               'Molecular Properties', 'Quality Assessment'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "domain"}]]
            )
            
            # 1. 置信度分數線圖
            confidence_scores = protein_result.get("confidence_scores", [])
            if confidence_scores:
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(confidence_scores) + 1)),
                             y=confidence_scores,
                             mode='lines+markers',
                             name='Confidence',
                             line=dict(color=self.colors["primary"])),
                    row=1, col=1
                )
            
            # 2. 對接分數條形圖
            if docking_results:
                scores = [result.get("docking_score", 0) for result in docking_results]
                methods = [result.get("method", f"Method_{i}") for i, result in enumerate(docking_results)]
                
                fig.add_trace(
                    go.Bar(x=methods, y=scores,
                          name='Docking Scores',
                          marker_color=self.colors["secondary"]),
                    row=1, col=2
                )
            
            # 3. 分子屬性雷達圖
            fig.add_trace(
                go.Scatter(x=['MW', 'LogP', 'HBD', 'HBA', 'TPSA'],
                         y=[0.8, 0.6, 0.7, 0.9, 0.5],  # 示例數據
                         mode='lines+markers',
                         name='Properties',
                         line=dict(color=self.colors["accent"])),
                row=2, col=1
            )
            
            # 4. 質量評估餅圖
            quality_data = protein_result.get("quality_assessment", {})
            if quality_data:
                high_conf = quality_data.get("high_confidence_residues", 0)
                med_conf = quality_data.get("total_residues", 100) - high_conf - quality_data.get("low_confidence_residues", 0)
                low_conf = quality_data.get("low_confidence_residues", 0)
                
                fig.add_trace(
                    go.Pie(labels=['High', 'Medium', 'Low'],
                          values=[high_conf, med_conf, low_conf],
                          name="Quality"),
                    row=2, col=2
                )
            
            # 更新佈局
            fig.update_layout(
                title_text="BioNeMo Analysis Dashboard",
                title_x=0.5,
                showlegend=True,
                height=800
            )
            
            if save_path:
                fig.write_html(save_path)
                return save_path
            else:
                fig.show()
                return None
                
        except Exception as e:
            logging.error(f"Interactive dashboard creation failed: {e}")
            return None
    
    # ========================================================
    # 工具函數
    # ========================================================
    
    def save_visualization_report(self,
                                protein_result: Dict[str, Any],
                                docking_results: List[Dict[str, Any]] = None,
                                output_dir: str = "./visualization_output") -> Dict[str, str]:
        """生成完整的可視化報告"""
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = {}
        
        try:
            # 1. 置信度分佈圖
            conf_plot = self.plot_confidence_distribution(
                protein_result, 
                save_path=os.path.join(output_dir, "confidence_distribution.png")
            )
            if conf_plot:
                generated_files["confidence_plot"] = conf_plot
            
            # 2. 對接分數比較
            if docking_results:
                docking_plot = self.plot_docking_scores(
                    docking_results,
                    save_path=os.path.join(output_dir, "docking_scores.png")
                )
                if docking_plot:
                    generated_files["docking_plot"] = docking_plot
            
            # 3. 交互式儀表板
            dashboard = self.create_interactive_dashboard(
                protein_result,
                docking_results or [],
                save_path=os.path.join(output_dir, "interactive_dashboard.html")
            )
            if dashboard:
                generated_files["dashboard"] = dashboard
            
            # 4. 生成摘要報告
            report_path = os.path.join(output_dir, "visualization_report.html")
            self._generate_html_report(protein_result, docking_results, 
                                     generated_files, report_path)
            generated_files["report"] = report_path
            
            return generated_files
            
        except Exception as e:
            logging.error(f"Visualization report generation failed: {e}")
            return {}
    
    def _generate_html_report(self,
                            protein_result: Dict[str, Any],
                            docking_results: List[Dict[str, Any]],
                            generated_files: Dict[str, str],
                            output_path: str):
        """生成HTML格式的可視化報告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BioNeMo Analysis Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid {self.colors["primary"]}; }}
                .metric {{ display: inline-block; margin: 10px 15px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BioNeMo Analysis Visualization Report</h1>
                <p>Generated on: {np.datetime64('now')}</p>
            </div>
            
            <div class="section">
                <h2>Protein Structure Analysis</h2>
                <div class="metric">
                    <strong>Sequence Length:</strong> {protein_result.get('sequence_length', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Mean Confidence:</strong> {protein_result.get('mean_confidence', 'N/A'):.3f}
                </div>
                <div class="metric">
                    <strong>Quality Grade:</strong> {protein_result.get('quality_assessment', {}).get('quality_grade', 'N/A')}
                </div>
            </div>
            
            {"<div class='section'><h2>Docking Results</h2>" if docking_results else ""}
            {"".join([f"<div class='metric'><strong>Method:</strong> {r.get('method', 'N/A')} | <strong>Score:</strong> {r.get('docking_score', 0):.2f} kcal/mol</div>" for r in (docking_results or [])]) }
            {"</div>" if docking_results else ""}
            
            <div class="section">
                <h2>Generated Visualizations</h2>
                {"".join([f"<p><strong>{name}:</strong> <a href='{path}'>{path}</a></p>" for name, path in generated_files.items()])}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

# ========================================================
# 模組級別的便捷函數
# ========================================================

# 創建全域可視化管理器實例
_viz_manager = VisualizationManager()

def get_visualization_capabilities() -> Dict[str, bool]:
    """獲取可用的可視化能力"""
    return _viz_manager.get_capabilities()

def visualize_protein_3d(structure_result: Dict[str, Any], **kwargs) -> Any:
    """快速蛋白質3D可視化"""
    return _viz_manager.visualize_protein_structure_3d(structure_result, **kwargs)

def visualize_molecule_3d(smiles: str, **kwargs) -> Any:
    """快速分子3D可視化"""
    return _viz_manager.visualize_molecule_3d(smiles, **kwargs)

def visualize_docking_complex(docking_result: Dict[str, Any], 
                            protein_structure: Dict[str, Any],
                            ligand_smiles: str, **kwargs) -> Any:
    """快速對接複合物可視化"""
    return _viz_manager.visualize_docking_complex(
        docking_result, protein_structure, ligand_smiles, **kwargs
    )

def create_visualization_report(protein_result: Dict[str, Any],
                              docking_results: List[Dict[str, Any]] = None,
                              output_dir: str = "./visualization_output") -> Dict[str, str]:
    """創建完整的可視化報告"""
    return _viz_manager.save_visualization_report(protein_result, docking_results, output_dir)

# ========================================================
# 模組初始化
# ========================================================

__all__ = [
    'VisualizationManager',
    'get_visualization_capabilities', 
    'visualize_protein_3d',
    'visualize_molecule_3d',
    'visualize_docking_complex',
    'create_visualization_report'
]

# 初始化日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 顯示可用的可視化能力
capabilities = get_visualization_capabilities()
logger.info(f"Visualization capabilities: {capabilities}")

if not any(capabilities.values()):
    logger.warning("No visualization libraries available! Please install py3Dmol, matplotlib, plotly, and/or RDKit")
else:
    available_libs = [lib for lib, available in capabilities.items() if available]
    logger.info(f"Available visualization libraries: {', '.join(available_libs)}")

# ========================================================
# LangChain工具函數
# ========================================================

def visualize_protein_structure(sequence: str = None, 
                               pdb_data: str = None,
                               style: str = "cartoon") -> Dict[str, Any]:
    """
    LangChain工具函數：蛋白質結構3D可視化
    
    Args:
        sequence: 蛋白質序列（將進行結構預測）
        pdb_data: 直接提供的PDB數據
        style: 顯示樣式
    
    Returns:
        可視化結果字典
    """
    if not PY3DMOL_AVAILABLE:
        return {"success": False, "error": "py3Dmol not available"}
    
    try:
        if pdb_data:
            structure_result = {"success": True, "pdb_data": pdb_data, "confidence_scores": []}
        elif sequence and SERVICES_AVAILABLE:
            from ..services.protein_service import ProteinService
            service = ProteinService()
            structure_result = service.predict_structure(sequence)
        else:
            return {"success": False, "error": "Either sequence or pdb_data must be provided"}
        
        viewer = visualize_protein_3d(structure_result, style=style)
        
        return {
            "success": viewer is not None and not isinstance(viewer, dict),
            "viewer_created": viewer is not None and not isinstance(viewer, dict),
            "style": style,
            "capabilities": get_visualization_capabilities()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_analysis_dashboard(protein_result: Dict[str, Any],
                             docking_results: List[Dict[str, Any]] = None,
                             output_dir: str = "./reports") -> Dict[str, Any]:
    """
    LangChain工具函數：創建分析儀表板
    
    Args:
        protein_result: 蛋白質分析結果
        docking_results: 對接結果列表
        output_dir: 輸出目錄
    
    Returns:
        生成的報告文件信息
    """
    try:
        generated_files = create_visualization_report(
            protein_result, 
            docking_results or [], 
            output_dir
        )
        
        return {
            "success": bool(generated_files),
            "generated_files": generated_files,
            "output_directory": output_dir,
            "file_count": len(generated_files)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
