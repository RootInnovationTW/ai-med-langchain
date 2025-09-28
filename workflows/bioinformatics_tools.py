from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool

class GeneSequenceInput(BaseModel):
    sequence_a: str
    sequence_b: str

class GeneSequenceAnalyzer(BaseTool):
    name = "GeneSequenceAnalyzer"
    description = "Compare two DNA sequences for similarity."
    args_schema: Type[BaseModel] = GeneSequenceInput
    return_direct: bool = True

    def _calc_similarity(self, a, b):
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return (matches / max(len(a), len(b))) * 100

    def _run(self, sequence_a, sequence_b, run_manager: Optional = None):
        return f"Similarity: {self._calc_similarity(sequence_a, sequence_b):.2f}%"

class DilutionInput(BaseModel):
    aliquot_volume_ml: float
    final_volume_ml: float

@tool(args_schema=DilutionInput)
def calculate_dilution(aliquot_volume_ml, final_volume_ml, run_manager: Optional = None):
    if aliquot_volume_ml <= 0 or final_volume_ml <= 0:
        return "Volumes must be > 0."
    if aliquot_volume_ml > final_volume_ml:
        return "Aliquot cannot exceed final volume."
    return f"Dilution factor: {final_volume_ml / aliquot_volume_ml:.2f}"
