from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

documents = [
    "Ligand A shows high affinity for kinase domain in docking experiments.",
    "CRISPR-Cas9 enables precise gene editing with therapeutic potential.",
    "Photosynthesis converts CO2 into glucose using sunlight."
]

splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_text("\n".join(documents))

vector_store = FAISS.from_texts(chunks, embedding_model)

query = "What ligand is relevant for kinase docking?"
results = vector_store.similarity_search_with_score(query)

for doc, score in results:
    print(f"[Score={score:.4f}] {doc.page_content}")
