# services/molecule_service.py
# ========================================================
# 功能: 提供分子處理、計算與可視化
# 套件: RDKit + py3Dmol
# ========================================================

from rdkit import Chem
from rdkit.Chem import Draw, QED, AllChem
import py3Dmol
from IPython.display import display

class MoleculeService:
    """Molecule processing service using RDKit + py3Dmol"""

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"❌ 無法解析 SMILES: {smiles}")

    def get_basic_info(self) -> dict:
        """回傳基本資訊 (分子式, QED Score)"""
        formula = Chem.rdMolDescriptors.CalcMolFormula(self.mol)
        qed_score = QED.qed(self.mol)
        return {
            "smiles": self.smiles,
            "formula": formula,
            "QED": round(qed_score, 4)
        }

    def draw_2d(self, filename: str = None):
        """生成 2D 分子圖 (可選擇存檔)"""
        img = Draw.MolToImage(self.mol, size=(300, 300))
        if filename:
            img.save(filename)
            print(f"📂 2D 分子圖已存檔: {filename}")
        return img

    def generate_3d(self, optimize: bool = True):
        """生成 3D 分子結構 (使用 RDKit 轉 3D)"""
        mol_3d = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        if optimize:
            AllChem.UFFOptimizeMolecule(mol_3d)
        return mol_3d

    def show_3d(self):
        """在 Jupyter/Notebook 環境可視化 3D 分子"""
        mol_3d = self.generate_3d()
        pdb_block = Chem.MolToPDBBlock(mol_3d)
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(pdb_block, "pdb")
        viewer.setStyle({"stick": {}})
        viewer.zoomTo()
        return viewer.show()

# === 測試區 ===
if __name__ == "__main__":
    # Paxlovid 成分 Nirmatrelvir 的簡化 SMILES
    smiles = "CC1=NC(=O)N(C1=O)C2CC2C(=O)NCC3=CC=CC=C3"
    svc = MoleculeService(smiles)

    print("🔬 分子資訊:", svc.get_basic_info())
    img = svc.draw_2d("nirmatrelvir.png")  # 輸出一張 2D 圖
    img.show()  # 顯示在 Python 環境
    svc.show_3d()  # 顯示 3D 結構

