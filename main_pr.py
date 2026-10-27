
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

data = PyPDFLoader("Document_loaders/data_science_learning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that summarizes text.
        Summarize the given text in a maximum of 5 bullet points.
        Use simple English.
        Keep the total response under 100 words.
        Include only the most important information."""
    ),
    ("user", "{data}")
])

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=150,
    temperature=0.2
)

prompt = template.format_messages(
    data=chunks[0].page_content
)

response = model.invoke(prompt)

print(response.content)
