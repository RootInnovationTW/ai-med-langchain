from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class PatientAssessment(BaseModel):
    diagnosis: str = Field(description="Primary medical diagnosis")
    pain_level: int = Field(description="Pain level 0-10")
    symptoms: List[str] = Field(description="Reported symptoms")
    requires_hospitalization: bool = Field(description="Hospitalization needed")

parser = PydanticOutputParser(pydantic_object=PatientAssessment)

template = """
Based on the following patient record, provide a structured medical assessment:
Patient: {patient_info}
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["patient_info"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
chain = prompt | llm | parser

result = chain.invoke({
    "patient_info": "45yo male, chest pain, shortness of breath, fever 101F, hypertension"
})
print(result)
