import os
import configparser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.tools import Tool


# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

VECTOR_DB_PATH = config["DEFAULT"]["vector_store_db_path"]

# Few-shot examples to be embedded (run once)
FEW_SHOT_EXAMPLES = [
    Document(page_content="""
    Q: Which state has the highest number of high-risk patients?
    A: 
    SELECT state, COUNT(*) AS count
    FROM patient_matrix
    WHERE adherance_score <= 0.4>
    GROUP BY state
    ORDER BY count DESC
    LIMIT 1;
    
    → State: Texas (Count: 124)
"""),
    Document(page_content="""
    Q: What is the average adherence score of chronic patients?
    A: 
    SELECT AVG(adherence_score)
    FROM patient_matrix
    WHERE condition = 'chronic';

    → Result: 0.63
    """)
]

def get_few_shot_tool(llm):
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load or create FAISS index
    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding_model)
    else:
        vectorstore = FAISS.from_documents(FEW_SHOT_EXAMPLES, embedding_model)
        vectorstore.save_local(VECTOR_DB_PATH)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return Tool(
        name="retrieve_few_shot_examples",
        description="Use this to retrieve similar examples of past adherence predictions based on patient info.",
        func=qa_chain.run
    )
