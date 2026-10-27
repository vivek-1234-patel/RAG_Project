from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(  model="models/gemini-embedding-001")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,
                   "lambda_mult": 0.5,
                   "fetch_k": 10}

)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash")

#prompt template for the llm to answer the question based on the retrieved documents
prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("human", "Answer the question based on the context below."
         " If the answer is not contained within the text below, "
         "say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}")
])

print("RAG system created")

print("press 0 to exit ")

while True:
    query = input("you: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_with_context = prompt.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(prompt_with_context)

    print("\nAnswer:\n")
    print(response.content)

