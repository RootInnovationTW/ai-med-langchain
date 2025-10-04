#!/bin/bash

# Setup script for LangChain Chapter 3 examples
# Extracts and organizes code from "Introducing LangChain" 

set -e

PROJECT_ROOT="$(pwd)"
CHAPTER3_DIR="$PROJECT_ROOT/examples"

echo "🚀 Setting up LangChain Chapter 3 examples in ai-med-langchain project..."

# Create directory structure
create_directories() {
    echo "📁 Creating directory structure..."
    
    mkdir -p "$CHAPTER3_DIR"/{
        basic_components,
        vector_stores,
        chains_and_lcel,
        prompts_and_memory,
        tools_and_agents,
        complete_examples
    }
    echo "✅ Directory structure created"
}

# Create basic components examples
create_basic_components() {
    echo "📝 Creating basic components examples..."
    
    cat > "$CHAPTER3_DIR/basic_components/faiss_example.py" << 'EOF'
"""
Example 3-1: FAISS similarity search with score
From Chapter 3: Introducing LangChain
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def setup_faiss_vectorstore():
    """Setup FAISS vector store with life science documents"""
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create life science documents
    doc1 = """The human genome consists of approximately 3 billion base pairs of
    DNA. These sequences contain the instructions for building and maintaining the
    human body."""
    
    doc2 = """CRISPR-Cas9 is a revolutionary gene-editing technology that allows for
    precise, directed changes to genomic DNA. It has the potential to correct
    genetic defects and treat diseases."""
    
    doc3 = """Photosynthesis is the process by which green plants and some other
    organisms use sunlight to synthesize foods with the help of chlorophyll from
    carbon dioxide and water."""
    
    # This step is only because the demo sentences are too short for actual
    # documents to mimic a document being split.
    documents = "\n".join([doc1, doc2, doc3])
    
    # Split documents into chunks (usually needed)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50, 
        chunk_overlap=10,
        separators=["\n"], 
        keep_separator=False
    )
    chunks = text_splitter.split_text(documents)
    
    # Initialize FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    return vector_store

def search_example():
    """Example of similarity search with scores"""
    vector_store = setup_faiss_vectorstore()
    
    query = "What is CRISPR-Cas9?"
    results = vector_store.similarity_search_with_score(query)
    
    print("Search Results:")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
        print("-" * 50)
    
    return results

if __name__ == "__main__":
    search_example()
EOF

    echo "✅ FAISS example created"
}

# Create LCEL chain examples
create_lcel_examples() {
    echo "📝 Creating LCEL chain examples..."
    
    cat > "$CHAPTER3_DIR/chains_and_lcel/temperature_conversion.py" << 'EOF'
"""
Example 3-2 & 3-3: RunnableSequence and RunnableParallel
Temperature conversion examples from Chapter 3
"""

from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel

def create_fahrenheit_to_celsius_chain():
    """Example 3-2: RunnableSequence for temperature conversion"""
    
    # A RunnableSequence constructed using the `|` operator
    sequence = RunnableLambda(lambda x: x - 32) | RunnableLambda(lambda x: x * 5/9)
    
    # Alternative construction
    sequence_alt = RunnableSequence(
        first=RunnableLambda(lambda x: x - 32),
        last=RunnableLambda(lambda x: x * 5/9)
    )
    
    return sequence

def create_parallel_temperature_chain():
    """Example 3-3: RunnableParallel for multiple conversions"""
    
    # A sequence that contains a RunnableParallel constructed using a dict literal
    sequence = RunnableLambda(lambda x: x - 32) | {
        'to_celsius': RunnableLambda(lambda x: x * 5/9),
        'to_reaumur': RunnableLambda(lambda x: x * 4/9)
    }
    
    # Alternative construction
    sequence_alt = RunnableLambda(lambda x: x - 32) | RunnableParallel(
        to_celsius=RunnableLambda(lambda x: x * 5/9),
        to_reaumur=RunnableLambda(lambda x: x * 4/9)
    )
    
    return sequence

def test_conversions():
    """Test the conversion chains"""
    
    # Test single conversion
    celsius_chain = create_fahrenheit_to_celsius_chain()
    
    print("Single Conversion (Fahrenheit to Celsius):")
    print(f"32°F = {celsius_chain.invoke(32):.1f}°C")
    print(f"Batch: [32, 0, -40]°F = {celsius_chain.batch([32, 0, -40])}")
    
    # Test parallel conversion
    parallel_chain = create_parallel_temperature_chain()
    
    print("\nParallel Conversion:")
    result = parallel_chain.invoke(32)
    print(f"32°F = {result}")
    
    batch_results = parallel_chain.batch([32, 0, -40])
    print(f"Batch results: {batch_results}")

if __name__ == "__main__":
    test_conversions()
EOF

    echo "✅ LCEL examples created"
}

# Create patient assessment chain example
create_patient_assessment() {
    echo "📝 Creating patient assessment example..."
    
    cat > "$CHAPTER3_DIR/chains_and_lcel/patient_assessment.py" << 'EOF'
"""
Example 3-4: Running patient assessment chain
Output parsers and structured data from Chapter 3
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# Define the patient data structure
class PatientAssessment(BaseModel):
    diagnosis: str = Field(description="Primary medical diagnosis")
    pain_level: int = Field(description="Pain level on scale of 0-10")
    symptoms: List[str] = Field(description="List of reported symptoms")
    requires_hospitalization: bool = Field(
        description="Whether the patient needs to be hospitalized"
    )

def create_patient_assessment_chain():
    """Create a chain for structured patient assessment"""
    
    # Create the parser
    parser = PydanticOutputParser(pydantic_object=PatientAssessment)
    
    # Create the prompt template
    template = """
    Based on the following patient information, provide a medical assessment:
    Patient Information: {patient_info}
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["patient_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Set up the chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = prompt | llm | parser
    
    return chain

def test_patient_assessment():
    """Test the patient assessment chain"""
    
    chain = create_patient_assessment_chain()
    
    patient_info = """45-year-old male presenting with chest pain,
    shortness of breath, and fever of 101°F for the past 2 days. 
    History of hypertension."""
    
    result = chain.invoke({"patient_info": patient_info})
    
    print("Patient Assessment Results:")
    print(f"Diagnosis: {result.diagnosis}")
    print(f"Pain Level: {result.pain_level}/10")
    print(f"Symptoms: {', '.join(result.symptoms)}")
    print(f"Hospitalization Required: {result.requires_hospitalization}")
    print(f"Type: {type(result)}")
    
    return result

if __name__ == "__main__":
    test_patient_assessment()
EOF

    echo "✅ Patient assessment example created"
}

# Create chemistry prompt examples
create_chemistry_prompts() {
    echo "📝 Creating chemistry prompt examples..."
    
    cat > "$CHAPTER3_DIR/prompts_and_memory/chemistry_prompts.py" << 'EOF'
"""
Example 3-5: Chemistry prompt example
Demonstrates prompt engineering for chemistry models
"""

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def setup_chemistry_model():
    """Setup the chemistry-specific T5 model"""
    
    # Load the model and tokenizer
    model_name = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a pipeline with the model and tokenizer
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0,
        max_length=512,
        num_beams=5
    )
    
    # Create a LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

def create_chemistry_chains():
    """Create different prompt chains for chemistry"""
    
    llm = setup_chemistry_model()
    
    # The text to be continued
    TEXT = "The formula of dihydrogen monoxide is"
    
    # Define prompt templates
    basic_prompt = PromptTemplate.from_template("{input_text}")
    prompt1 = PromptTemplate.from_template(
        "Continue the following phrase as a chemist: {input_text}"
    )
    prompt2 = PromptTemplate.from_template(
        """You are a professional chemistry researcher. Finish the following sentence:
        {input_text}"""
    )
    
    # Create LCEL chains for each prompt
    basic_chain = basic_prompt | llm | StrOutputParser()
    prompt1_chain = prompt1 | llm | StrOutputParser()
    prompt2_chain = prompt2 | llm | StrOutputParser()
    
    return basic_chain, prompt1_chain, prompt2_chain, TEXT

def test_chemistry_prompts():
    """Test different chemistry prompts"""
    
    basic_chain, prompt1_chain, prompt2_chain, TEXT = create_chemistry_chains()
    
    print("Chemistry Prompt Engineering Results:")
    print(f"Input: {TEXT}")
    print("-" * 50)
    
    print("Basic prompt result:")
    print(basic_chain.invoke({"input_text": TEXT}))
    print("-" * 50)
    
    print("Chemistry prompt result:")
    print(prompt1_chain.invoke({"input_text": TEXT}))
    print("-" * 50)
    
    print("Professional researcher prompt result:")
    print(prompt2_chain.invoke({"input_text": TEXT}))

if __name__ == "__main__":
    test_chemistry_prompts()
EOF

    echo "✅ Chemistry prompts example created"
}

# Create few-shot learning example
create_few_shot_example() {
    echo "📝 Creating few-shot learning example..."
    
    cat > "$CHAPTER3_DIR/prompts_and_memory/few_shot_medical.py" << 'EOF'
"""
Example 3-6: Few-shot examples for medical reasoning
Demonstrates few-shot learning for medical questions
"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def create_few_shot_medical_chain():
    """Create a few-shot learning chain for medical questions"""
    
    examples = [
        {
            "question": "Is Penicillin effective against E. coli?",
            "answer": """Are follow-up questions needed here: Yes.
            Follow up: What class of antibiotics does Penicillin belong to?
            Intermediate answer: Penicillin belongs to the beta-lactam class of
            antibiotics.
            ...
            Follow up: Is E. coli resistant to beta-lactam antibiotics?
            Intermediate answer: Many strains of E. coli have developed resistance to
            beta-lactam antibiotics, including Penicillin. So the final answer is:
            Penicillin is generally not effective against E. coli due to resistance.
            """,
        },
        {
            "question": "Do Aspirin and Ibuprofen have the same mechanism of action?",
            "answer": """Are follow-up questions needed here: Yes.
            Follow up: What is the mechanism of action of aspirin?
            Intermediate Answer: Aspirin works by inhibiting the enzyme
            cyclooxygenase (COX), which reduces the formation of prostaglandins
            and thromboxanes, leading to its anti-inflammatory and anticoagulant
            effects.
            Follow up: What is the mechanism of action of Ibuprofen?
            Intermediate Answer: Ibuprofen also inhibits the cyclooxygenase (COX)
            enzyme, reducing the production of prostaglandins.
            So the final answer is: Yes, both Aspirin and Ibuprofen have the same
            mechanism of action, which is the inhibition of the COX enzyme.
            """,
        },
    ]
    
    # setting the prompt template to process the examples list
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\n{answer}"
    )
    
    # setting the prompt template to be used with the LLM
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    few_shot_chain = prompt | llm | StrOutputParser()
    
    return few_shot_chain, llm

def test_few_shot_learning():
    """Test few-shot learning approach"""
    
    TEXT = """How do genetic mutations in oncogenes and tumor suppressor genes
    interact to drive the progression of metastatic cancer?"""
    
    few_shot_chain, llm = create_few_shot_medical_chain()
    
    print("Direct LLM response:")
    print(llm.invoke(TEXT).content[:200] + "...")
    print("\n" + "="*50 + "\n")
    
    print("Few-shot chain response:")
    print(few_shot_chain.invoke({"input": TEXT}))

if __name__ == "__main__":
    test_few_shot_learning()
EOF

    echo "✅ Few-shot learning example created"
}

# Create memory example
create_memory_example() {
    echo "📝 Creating memory example..."
    
    cat > "$CHAPTER3_DIR/prompts_and_memory/clinical_memory.py" << 'EOF'
"""
Example 3-8 & 3-9: LangChain memory setup and tests
Demonstrates conversation memory for medical applications
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI

class MedicalChat:
    """Medical chat system with conversation memory"""
    
    def __init__(self, store=None, model=None):
        self.store = store if store is not None else {}
        self.model = model if model is not None else ChatOpenAI(temperature=0)
        self.prompt_template = self._create_prompt_template()
        self.runnable = self.prompt_template | self.model
        self.chat = RunnableWithMessageHistory(
            self.runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    @staticmethod
    def _create_prompt_template():
        """Constructs and returns a chat prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """You're an assistant who's good at understanding
            medicine side effects and contraindications"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return self.store.setdefault(session_id, ChatMessageHistory())
    
    def run_chat(self, input_text, session_id: str) -> str:
        response = self.chat.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content

def test_medical_memory():
    """Test memory functionality with medical conversations"""
    
    # initiate chat
    med_chat = MedicalChat()
    
    print("Testing Medical Chat Memory")
    print("=" * 40)
    
    # session abc123
    print("Session abc123 - First question:")
    response1 = med_chat.run_chat("What are the side effects of taking aspirin?", "abc123")
    print(response1[:200] + "...")
    
    print("\nSession abc123 - Follow-up question:")
    response2 = med_chat.run_chat("Can I take it before a flight?", "abc123")
    print(response2[:200] + "...")
    
    print("\nSession xyz123 - Same question, no context:")
    response3 = med_chat.run_chat("Can I take it before a flight?", "xyz123")
    print(response3)
    
    return med_chat

if __name__ == "__main__":
    test_medical_memory()
EOF

    echo "✅ Memory example created"
}

# Create custom tools examples
create_tools_examples() {
    echo "📝 Creating custom tools examples..."
    
    cat > "$CHAPTER3_DIR/tools_and_agents/bioinformatics_tools.py" << 'EOF'
"""
Example 3-10 & 3-11: Custom tool examples
GeneSequenceAnalyzer and DilutionCalculator tools
"""

from typing import Type, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from operator import itemgetter

# GeneSequenceAnalyzer - Option 1 through BaseTool subclass
class GeneSequenceInput(BaseModel):
    sequence_a: str = Field(description="First DNA sequence")
    sequence_b: str = Field(description="Second DNA sequence")

class GeneSequenceAnalyzer(BaseTool):
    name = "GeneSequenceAnalyzer"
    description = """Compares two DNA sequences to determine their similarity
    percentage."""
    args_schema: Type[BaseModel] = GeneSequenceInput
    return_direct: bool = True

    def _calculate_similarity(self, sequence_a: str, sequence_b: str) -> float:
        """Calculates the similarity percentage between two DNA sequences."""
        matches = sum(1 for a, b in zip(sequence_a, sequence_b) if a == b)
        length = max(len(sequence_a), len(sequence_b))
        return (matches / length) * 100 if length > 0 else 0

    def _run(
        self, sequence_a: str, sequence_b: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Compares two DNA sequences."""
        similarity_percentage = self._calculate_similarity(sequence_a, sequence_b)
        return f"""The similarity between the given DNA sequences is
        {similarity_percentage:.2f}%."""

# DilutionCalculator - Option 2 through @tool
class DilutionInput(BaseModel):
    aliquot_volume_ml: float = Field(
        ..., description="Volume of the solution to be diluted, in milliliters"
    )
    final_volume_ml: float = Field(
        ..., description="Final volume after dilution, in milliliters"
    )

@tool(args_schema=DilutionInput)
def calculate_dilution(
    aliquot_volume_ml: float,
    final_volume_ml: float,
    run_manager: Optional[CallbackManagerForToolRun] = None
) -> str:
    """Calculate the dilution factor."""
    if aliquot_volume_ml <= 0 or final_volume_ml <= 0:
        return "Both aliquot volume and final volume must be greater than 0."
    if aliquot_volume_ml > final_volume_ml:
        return "Aliquot volume cannot be greater than the final volume."
    dilution_factor = final_volume_ml / aliquot_volume_ml
    return f"The dilution factor is {dilution_factor:.2f}"

def create_bioinformatics_chain():
    """Create a chain with bioinformatics tools"""
    
    tools = [GeneSequenceAnalyzer(), calculate_dilution]
    tool_map = {tool.name: tool for tool in tools}
    
    # bind our model to use tools
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(tools)
    
    def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
        """Function for dynamically constructing the end of the chain based on the
        model-selected tool."""
        tool = tool_map[tool_invocation["type"]]
        return RunnablePassthrough.assign(output=itemgetter("args") | tool)
    
    # .map() allows us to apply a function to a list of inputs.
    call_tool_list = RunnableLambda(call_tool).map()
    chain = model_with_tools | JsonOutputToolsParser() | call_tool_list
    
    return chain

def test_bioinformatics_tools():
    """Test the bioinformatics tools chain"""
    
    chain = create_bioinformatics_chain()
    
    print("Testing Bioinformatics Tools")
    print("=" * 40)
    
    # Test gene sequence comparison
    print("1. Gene sequence similarity:")
    result1 = chain.invoke("How similar are AGCTGACCTG and AGCTTACCGT gene sequences?")
    print(result1)
    print()
    
    # Test dilution calculation
    print("2. Dilution calculation:")
    result2 = chain.invoke("""How diluted will the solution be if I add 100 ml to my current
    50 ml solution?""")
    print(result2)
    print()
    
    # Test another dilution
    print("3. Another dilution calculation:")
    result3 = chain.invoke("""I need to get 1 l solution from a 40 ml I initially have. What
    will be the dilution factor?""")
    print(result3)
    print()
    
    # Test with non-tool question
    print("4. Non-tool question:")
    result4 = chain.invoke("What is the difference between a drug's efficacy and its potency?")
    print(result4)

if __name__ == "__main__":
    test_bioinformatics_tools()
EOF

    echo "✅ Bioinformatics tools example created"
}

# Create agent examples
create_agent_examples() {
    echo "📝 Creating agent examples..."
    
    cat > "$CHAPTER3_DIR/tools_and_agents/gene_sequence_agent.py" << 'EOF'
"""
Example 3-12: Agent gene sequence comparison
ReAct agent for finding closest gene pairs
"""

from pydantic import BaseModel, Field
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain_core.tools import tool

# Modified tool for single input
class GeneSequenceSingleInput(BaseModel):
    pair: str = Field(..., description="Pair of DNA sequences to be compared")

@tool(args_schema=GeneSequenceSingleInput)
def calculate_dna_similarity(pair: str) -> float:
    """Calculates the similarity percentage between two DNA sequences."""
    sequence_a, sequence_b = pair.split(", ")
    matches = sum(1 for a, b in zip(sequence_a, sequence_b) if a == b)
    length = max(len(sequence_a), len(sequence_b))
    return (matches / length) * 100 if length > 0 else 0

def create_gene_sequence_agent():
    """Create a ReAct agent for gene sequence analysis"""
    
    tools = [calculate_dna_similarity]
    prompt = hub.pull("hwchase17/react")
    model = OpenAI(temperature=0)
    
    # Construct the ReAct agent
    agent = create_react_agent(model, tools, prompt)
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def test_gene_sequence_agent():
    """Test the gene sequence agent"""
    
    agent_executor = create_gene_sequence_agent()
    
    print("Testing Gene Sequence Agent")
    print("=" * 40)
    
    result = agent_executor.invoke({
        "input": """Find two closest genes from the given list:
        AGCTA, CTTAC, AGCTG, AGAGA"""
    })
    
    print(f"\nFinal Result: {result['output']}")
    return result

if __name__ == "__main__":
    test_gene_sequence_agent()
EOF

    echo "✅ Agent examples created"
}

# Create complete Nobel Prize example
create_complete_example() {
    echo "📝 Creating complete Nobel Prize Q&A example..."
    
    cat > "$CHAPTER3_DIR/complete_examples/nobel_prize_qa.py" << 'EOF'
"""
Complete LangChain application: Nobel Prize Q&A system
Combines all components: models, indexes, chains, prompts, memory, tools, agents
"""

# Models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Index and vector stores
from langchain_community.vectorstores import Chroma

# Chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# Memory
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory

# Prompts
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# Tools
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# Agents
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser
)

MEMORY_KEY = "chat_history"

def setup_conversation_memory(model_name):
    """Initialize conversation memory with a warm-up exchange"""
    
    # Create history with initial warm-up messages
    history = ChatMessageHistory()
    history.add_user_message("""Hi, I want you to help me to answer some
    questions and complete a couple of tasks""")
    history.add_ai_message("Hello! I can help you. Can you specify your task?")
    
    # Create memory that can summarize conversation as it grows
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0, model_name=model_name),
        return_messages=True,
        memory_key=MEMORY_KEY,
        chat_memory=history
    )
    
    return memory, history

def create_prompt_template(memory_key):
    """Create a prompt template with system message and placeholders"""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an assistant who helps answer scientific questions.
            Use tools you have if required.
            Be sure to understand the question correctly.
            If you don't know the answer - respond, "Sorry, I don't know."
            """,
        ),
        MessagesPlaceholder(variable_name=memory_key),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def setup_rag_pipeline():
    """Setup RAG pipeline with Nobel Prize document"""
    
    # For demo purposes, we'll use a placeholder link
    # In practice, you would download the actual PDF
    link = "https://www.nobelprize.org/uploads/2023/10/advanced-chemistryprize2023.pdf"
    
    # You can also load from local file:
    # pdf_loader = PyPDFLoader("nobel_prize_2023_chemistry.pdf")
    
    print("Note: For this demo, you need to download the Nobel Prize PDF manually")
    print(f"Download from: {link}")
    
    # Create sample documents for demonstration
    sample_docs = [
        """The Nobel Prize in Chemistry 2023 was awarded to Moungi G. Bawendi, 
        Louis E. Brus, and Aleksey I. Yekimov for the discovery and synthesis of quantum dots.""",
        
        """Quantum dots are nanometre-sized semiconductor crystals whose properties 
        are determined by quantum size effects. They can be made from various 
        semiconductor materials such as cadmium selenide (CdSe), lead sulfide (PbS), 
        and indium phosphide (InP).""",
        
        """The research showed that CuCl crystals forming in glass matrix can be 
        controlled by varying temperature and duration of heat treatment. Using 
        small-angle X-ray scattering, researchers were able to control the average 
        size of CuCl crystals."""
    ]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=200,
        keep_separator=False
    )
    
    # For demonstration with sample docs
    chunks = []
    for doc in sample_docs:
        chunks.extend(text_splitter.split_text(doc))
    
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=OpenAIEmbeddings()
    )
    
    return vectorstore

def create_nobel_qa_system():
    """Create complete Nobel Prize Q&A system"""
    
    model_name = "gpt-4o-mini"
    
    # Setup memory
    memory, history = setup_conversation_memory(model_name)
    
    # Setup prompt
    prompt = create_prompt_template(MEMORY_KEY)
    
    # Setup RAG pipeline
    vectorstore = setup_rag_pipeline()
    retriever = vectorstore.as_retriever()
    
    # Create tools
    retrieve_tool = create_retriever_tool(
        retriever,
        "search_through_pdf_text",
        """This function searches and returns data from pdf text regarding
        Nobel prize winners and their work""",
    )
    
    tools = [PubmedQueryRun(), retrieve_tool]
    formatted_functions = [format_tool_to_openai_function(t) for t in tools]
    
    # Setup LLM with tools
    llm = ChatOpenAI(temperature=0, model_name=model_name)
    llm_with_tools = llm.bind(functions=formatted_functions)
    
    # Create agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | RunnablePassthrough.assign(
            **{MEMORY_KEY: RunnableLambda(memory.load_memory_variables) |
               itemgetter(MEMORY_KEY)}
        )
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    
    qa = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    return qa

def test_nobel_qa_system():
    """Test the complete Nobel Prize Q&A system"""
    
    qa = create_nobel_qa_system()
    
    print("Testing Nobel Prize Q&A System")
    print("=" * 50)
    
    # Test queries
    queries = [
        "Who won the Nobel prize in Literature in 2023 and what did they win it for?",
        "Sorry, I meant chem",
        "What crystals were used by the authors to create nanoparticles?",
        "What are the titles of the 3 most recent publications for each of the crystals mentioned in the paper?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        try:
            result = qa.invoke({"input": query})
            print(f"Response: {result.get('output', 'No output')}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    test_nobel_qa_system()
EOF

    cat > "$CHAPTER3_DIR/complete_examples/retrieval_comparison.py" << 'EOF'
"""
Retrieval technique comparison from Chapter 3
Demonstrates different retrieval strategies: Standard, MMR, and Filtered
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_sample_vectorstore():
    """Create a sample vector store for testing retrieval methods"""
    
    # Sample documents simulating Nobel Prize content
    documents = [
        """The Nobel Prize in Chemistry 2023 was awarded to Moungi G. Bawendi, 
        Louis E. Brus, and Aleksey I. Yekimov for their discovery and synthesis of quantum dots.""",
        
        """Quantum dots constitute a new class of materials that is neither molecular nor bulk. 
        They have the same structure and atomic composition as bulk materials, but their properties 
        can be tuned using a single parameter, the particle's size.""",
        
        """The researchers attributed this observation to the formation of a crystalline phase 
        of CuCl in the glass matrix as a result of phase decomposition of a supersaturated 
        solution during heat treatment.""",
        
        """Furthermore, by varying temperature and duration of the heat treatment, they were 
        able to control the average size of CuCl crystals forming in the glass melt.""",
        
        """Using small-angle X-ray scattering, researchers demonstrated control over crystal 
        formation and size distribution in semiconductor materials.""",
        
        """References: Berry, C. R. Structure and Optical Absorption of AgI Microcrystals. 
        Physical Review 153, 989 (1967).""",
        
        """Golubkov, V. V.; Yekimov, A. I.; Onushchenko, A. A.; Tsekhomskii, V. A. 
        Growth kinetics of microcrystals in glass matrix. Soviet Glass Physics and Chemistry."""
    ]
    
    # Add metadata to simulate page numbers
    metadatas = [
        {"page": 1}, {"page": 5}, {"page": 5}, {"page": 5}, 
        {"page": 12}, {"page": 16}, {"page": 15}
    ]
    
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=OpenAIEmbeddings(),
        metadatas=metadatas
    )
    
    return vectorstore

def compare_retrieval_methods():
    """Compare different retrieval techniques"""
    
    vectorstore = create_sample_vectorstore()
    query = "What crystals were used by the authors to create nanoparticles?"
    
    print("Retrieval Methods Comparison")
    print("=" * 50)
    print(f"Query: {query}")
    print()
    
    # 1. Standard retrieval
    print("1. Standard Retrieval (k=3):")
    print("-" * 30)
    retriever_standard = vectorstore.as_retriever(search_kwargs={"k": 3})
    results_standard = retriever_standard.invoke(query)
    
    for i, doc in enumerate(results_standard):
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  Doc {i+1} (Page {page}): {content}")
    print()
    
    # 2. MMR retrieval
    print("2. MMR Retrieval (k=3, fetch_k=10, lambda_mult=0.25):")
    print("-" * 30)
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.25}
    )
    results_mmr = retriever_mmr.invoke(query)
    
    for i, doc in enumerate(results_mmr):
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  Doc {i+1} (Page {page}): {content}")
    print()
    
    # 3. Filtered retrieval
    print("3. Filtered Retrieval (k=3, pages <= 12):")
    print("-" * 30)
    retriever_filtered = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"page": {"$lte": 12}}}
    )
    results_filtered = retriever_filtered.invoke(query)
    
    for i, doc in enumerate(results_filtered):
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  Doc {i+1} (Page {page}): {content}")
    print()
    
    # Summary comparison
    print("Summary Comparison:")
    print("-" * 30)
    
    def extract_pages(results):
        return [doc.metadata.get('page', 'Unknown') for doc in results]
    
    def count_crystals(results):
        crystal_count = 0
        for doc in results:
            if 'CuCl' in doc.page_content:
                crystal_count += 1
        return crystal_count
    
    methods = [
        ("Standard", results_standard),
        ("MMR", results_mmr), 
        ("Filtered", results_filtered)
    ]
    
    for method_name, results in methods:
        pages = extract_pages(results)
        crystal_mentions = count_crystals(results)
        print(f"{method_name:10} | Pages: {pages} | Crystal mentions: {crystal_mentions}")

if __name__ == "__main__":
    compare_retrieval_methods()
EOF

    echo "✅ Complete examples created"
}

# Create requirements file
create_requirements() {
    echo "📝 Creating requirements file..."
    
    cat > "$CHAPTER3_DIR/requirements.txt" << 'EOF'
# LangChain Chapter 3 Requirements
# Core LangChain
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-huggingface>=0.1.0

# Vector stores and embeddings
chromadb>=0.5.0
faiss-cpu>=1.7.0

# Output parsers and tools
pydantic>=2.0.0

# Document processing
pypdf>=4.0.0

# Transformers for chemistry model
transformers>=4.30.0
torch>=2.0.0

# PubMed integration
xmltodict>=0.13.0

# Optional: For better performance
tiktoken>=0.5.0
numpy>=1.24.0
EOF

    echo "✅ Requirements file created"
}

# Create main documentation
create_documentation() {
    echo "📝 Creating documentation..."
    
    cat > "$CHAPTER3_DIR/README.md" << 'EOF'
# LangChain Chapter 3 Examples

This directory contains all code examples from Chapter 3: "Introducing LangChain" extracted from the PDF.

## Directory Structure

```
examples/
├── basic_components/        # Basic LangChain components
│   └── faiss_example.py    # FAISS vector store example
├── vector_stores/          # Vector store implementations  
├── chains_and_lcel/        # LCEL and chain examples
│   ├── temperature_conversion.py  # RunnableSequence/Parallel
│   └── patient_assessment.py      # Structured output parsing
├── prompts_and_memory/     # Prompt engineering and memory
│   ├── chemistry_prompts.py        # Chemistry model prompts
│   ├── few_shot_medical.py         # Few-shot learning
│   └── clinical_memory.py          # Conversation memory
├── tools_and_agents/       # Custom tools and agents
│   ├── bioinformatics_tools.py     # Gene analysis tools
│   └── gene_sequence_agent.py      # ReAct agent example
├── complete_examples/      # Full applications
│   ├── nobel_prize_qa.py           # Complete Q&A system
│   └── retrieval_comparison.py     # Retrieval methods
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Install dependencies:
```bash
cd examples
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export HUGGINGFACE_API_TOKEN="your-hf-token"  # Optional for chemistry models
```

## Running Examples

### Basic Components
```bash
python basic_components/faiss_example.py
```

### Chains and LCEL
```bash
python chains_and_lcel/temperature_conversion.py
python chains_and_lcel/patient_assessment.py
```

### Prompts and Memory
```bash
python prompts_and_memory/chemistry_prompts.py
python prompts_and_memory/few_shot_medical.py
python prompts_and_memory/clinical_memory.py
```

### Tools and Agents
```bash
python tools_and_agents/bioinformatics_tools.py
python tools_and_agents/gene_sequence_agent.py
```

### Complete Examples
```bash
python complete_examples/nobel_prize_qa.py
python complete_examples/retrieval_comparison.py
```

## Key Examples Explained

### Example 3-1: FAISS Vector Store
- Demonstrates similarity search with life science documents
- Shows embedding and chunk processing
- Includes similarity scoring

### Example 3-2 & 3-3: LCEL Chains
- Temperature conversion using RunnableSequence
- Parallel processing with RunnableParallel
- Demonstrates chain composition patterns

### Example 3-4: Patient Assessment
- Structured output parsing with Pydantic
- Medical data validation
- Type-safe LLM outputs

### Example 3-5: Chemistry Prompts
- Domain-specific prompt engineering
- Chemistry model integration
- Prompt effectiveness comparison

### Example 3-6: Few-Shot Learning
- Medical reasoning with examples
- Chain-of-thought prompting
- Structured medical responses

### Example 3-8 & 3-9: Memory System
- Conversation memory for medical applications
- Session-based context preservation
- Clinical decision support continuity

### Example 3-10 & 3-11: Custom Tools
- Bioinformatics tool creation
- Gene sequence analysis
- Dilution calculation tools

### Example 3-12: ReAct Agent
- Autonomous gene sequence comparison
- Multi-step reasoning
- Tool orchestration

## Integration with Main Project

These examples are designed to integrate with the main ai-med-langchain project structure:

- Copy relevant tools to `agent/tools.py`
- Integrate workflows into `agent/workflow.py` 
- Use patterns in main application development
- Apply security controls from `security/guardrails.py`

## Notes

- Some examples require API keys (OpenAI, Hugging Face)
- Chemistry models may require significant memory
- PubMed queries are rate-limited
- Vector stores will create local databases

## Further Reading

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
EOF

    echo "✅ Documentation created"
}

# Create integration script
create_integration_script() {
    echo "📝 Creating integration script..."
    
    cat > "$CHAPTER3_DIR/integrate_with_main.sh" << 'EOF'
#!/bin/bash

# Integration script to merge Chapter 3 examples with main project
# Run this from the examples directory

MAIN_PROJECT_ROOT="../"
AGENT_DIR="$MAIN_PROJECT_ROOT/agent"
WORKFLOWS_DIR="$MAIN_PROJECT_ROOT/workflows"

echo "🔗 Integrating Chapter 3 examples with main ai-med-langchain project..."

# Create backup
backup_dir="$MAIN_PROJECT_ROOT/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

if [ -d "$AGENT_DIR" ]; then
    cp -r "$AGENT_DIR" "$backup_dir/"
    echo "✅ Backup created at $backup_dir"
fi

# Integrate bioinformatics tools
echo "📝 Integrating bioinformatics tools..."
if [ -f "tools_and_agents/bioinformatics_tools.py" ]; then
    # Extract just the tool classes for integration
    echo "# Bioinformatics tools from Chapter 3" >> "$AGENT_DIR/tools.py"
    echo "" >> "$AGENT_DIR/tools.py"
    
    # Extract GeneSequenceAnalyzer and DilutionCalculator
    sed -n '/class GeneSequenceAnalyzer/,/return f"The dilution factor is {dilution_factor:.2f}"/p' \
        tools_and_agents/bioinformatics_tools.py >> "$AGENT_DIR/tools.py"
fi

# Integrate memory patterns
echo "📝 Integrating memory patterns..."
if [ -f "prompts_and_memory/clinical_memory.py" ]; then
    cp prompts_and_memory/clinical_memory.py "$WORKFLOWS_DIR/"
fi

# Integrate patient assessment
echo "📝 Integrating patient assessment..."
if [ -f "chains_and_lcel/patient_assessment.py" ]; then
    cp chains_and_lcel/patient_assessment.py "$WORKFLOWS_DIR/"
fi

# Create workflow integration examples
cat > "$WORKFLOWS_DIR/integration_example.py" << 'PYEOF'
"""
Chapter 3 Integration Example
Demonstrates how to use Chapter 3 patterns in the main project
"""

from agent.tools import GeneSequenceAnalyzer, calculate_dilution
from workflows.clinical_memory import MedicalChat
from workflows.patient_assessment import create_patient_assessment_chain

def integrated_bioinformatics_workflow():
    """Example of integrated bioinformatics workflow"""
    
    # Use Chapter 3 tools
    gene_analyzer = GeneSequenceAnalyzer()
    
    # Example analysis
    result = gene_analyzer._run("ATCG", "ATCG")
    print(f"Gene analysis result: {result}")
    
    # Use dilution calculator
    dilution_result = calculate_dilution(50.0, 150.0)
    print(f"Dilution result: {dilution_result}")

def integrated_medical_workflow():
    """Example of integrated medical workflow"""
    
    # Use Chapter 3 memory system
    med_chat = MedicalChat()
    
    # Use Chapter 3 patient assessment
    assessment_chain = create_patient_assessment_chain()
    
    print("Integrated medical workflow ready")

if __name__ == "__main__":
    integrated_bioinformatics_workflow()
    integrated_medical_workflow()
PYEOF

echo "✅ Integration complete!"
echo "📁 New files created in workflows/"
echo "🔧 Tools added to agent/tools.py"
echo "💾 Backup saved to $backup_dir"
echo ""
echo "Next steps:"
echo "1. Review integrated code"
echo "2. Update main.py to use new workflows"
echo "3. Test integration: python workflows/integration_example.py"
EOF

    chmod +x "$CHAPTER3_DIR/integrate_with_main.sh"
    echo "✅ Integration script created"
}

# Main execution
main() {
    echo "🚀 Starting LangChain Chapter 3 extraction and setup..."
    
    create_directories
    create_basic_components
    create_lcel_examples
    create_patient_assessment
    create_chemistry_prompts
    create_few_shot_example
    create_memory_example
    create_tools_examples
    create_agent_examples
    create_complete_example
    create_requirements
    create_documentation
    create_integration_script
    
    echo ""
    echo "🎉 LangChain Chapter 3 setup complete!"
    echo ""
    echo "📁 Created: $CHAPTER3_DIR"
    echo "📖 Read: $CHAPTER3_DIR/README.md for usage instructions"
    echo "🔗 Run: $CHAPTER3_DIR/integrate_with_main.sh to merge with main project"
    echo ""
    echo "🚀 Quick start:"
    echo "  cd $CHAPTER3_DIR"
    echo "  pip install -r requirements.txt"
    echo "  export OPENAI_API_KEY='your-key'"
    echo "  python basic_components/faiss_example.py"
}

# Run main function
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
main
