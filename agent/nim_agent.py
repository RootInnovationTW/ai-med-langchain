import os
import requests
from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

# === NIM Endpoints ===
MSA_HOST = 'http://localhost:8081'
OPENFOLD2_HOST = 'http://localhost:8082'
GENMOL_HOST = 'http://localhost:8083'
DIFFDOCK_HOST = 'http://localhost:8084'

# === 工具函數 ===
def check_service_health():
    results = {}
    for nim_url in [MSA_HOST, OPENFOLD2_HOST, GENMOL_HOST, DIFFDOCK_HOST]:
        try:
            response = requests.get(os.path.join(nim_url, "v1/health/ready"))
            results[nim_url] = (response.status_code, response.text)
        except Exception as e:
            results[nim_url] = f"Error: {e}"
    return results

def fold_protein(sequence: str):
    msa_response = requests.post(
        f'{MSA_HOST}/biology/colabfold/msa-search/predict',
        json={
            "sequence": sequence,
            "e_value": 0.0001,
            "iterations": 1,
            "search_type": "alphafold2",
            "output_alignment_formats": ["fasta", "a3m"],
            "databases": ["Uniref30_2302", "colabfold_envdb_202108", "PDB70_220313"]
        }
    ).json()

    of2_response = requests.post(
        f'{OPENFOLD2_HOST}/biology/openfold/openfold2/predict-structure-from-msa-and-template',
        json={
            "sequence": sequence,
            "use_templates": False,
            "relaxed_prediction": False,
            "alignments": msa_response['alignments']
        }
    ).json()

    return of2_response["structures_in_ranked_order"][0]["structure"]

def generate_molecules(smiles: str):
    response = requests.post(
        f'{GENMOL_HOST}/generate',
        json={
            "smiles": smiles,
            "num_molecules": 5,
            "temperature": 1,
            "noise": 0.2,
            "step_size": 4,
            "scoring": "QED"
        }
    ).json()
    return response["molecules"]

def docking(protein: str, ligands: str):
    response = requests.post(
        f'{DIFFDOCK_HOST}/molecular-docking/diffdock/generate',
        json={
            "protein": protein,
            "ligand": ligands,
            "ligand_file_type": "txt",
            "num_poses": 10,
            "time_divisions": 20,
            "num_steps": 18
        }
    ).json()
    return response

# === LangChain Agent Tools ===
tools = [
    Tool(
        name="Check NIM Services",
        func=lambda _: check_service_health(),
        description="檢查所有 BioNeMo NIM 是否在運行"
    ),
    Tool(
        name="Fold Protein",
        func=fold_protein,
        description="輸入蛋白質序列，獲得預測結構"
    ),
    Tool(
        name="Generate Molecules",
        func=generate_molecules,
        description="輸入 SMILES，產生候選小分子"
    ),
    Tool(
        name="Docking",
        func=lambda x: docking(x['protein'], x['ligands']),
        description="執行蛋白質-小分子 docking"
    )
]

# === 初始化 LangChain Agent ===
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)

# === 範例自然語言查詢 ===
if __name__ == "__main__":
    agent.run("幫我用 SARS-CoV-2 主蛋白質，折疊結構，並生成5個候選分子，最後做 docking")
