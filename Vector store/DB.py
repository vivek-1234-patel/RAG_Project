from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
   Document(
    page_content="Data Analytics involves collecting, cleaning, and analyzing data to identify trends, discover patterns, and support data-driven business decisions. Common tools include Python, SQL, Excel, and Power BI.",
    metadata={"source": "data_analytics.txt"}
),

Document(
    page_content="Data Science combines statistics, programming, machine learning, and data visualization to extract meaningful insights from data. It involves exploratory data analysis, feature engineering, predictive modeling, and building machine learning models using Python libraries such as Pandas, NumPy, and Scikit-learn.",
    metadata={"source": "data_science.txt"}
),
]   

embedding_model  = GoogleGenerativeAIEmbeddings(
     model="models/gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
    )

results = vectorstore.similarity_search("What is Data Analytics?", k=1)  

for result in results:
    print(result.page_content)
    print(result.metadata)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is Data Science?")

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
