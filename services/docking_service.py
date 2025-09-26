def run_docking(molecule: str, protein: str) -> str:
    """
    使用 DiffDock NIM 執行 docking 模擬
    :param molecule: 小分子 SMILES 字串
    :param protein: 蛋白質結構 PDB ID
    :return: 模擬結果 (字串格式)
    """
    return f"Docking result: {molecule} vs {protein}"
