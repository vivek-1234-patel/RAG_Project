from langchain_community.document_loaders import WebBaseLoader  

url = "https://aiengineeringfromscratch.com/"

data = WebBaseLoader([url])

docs = data.load()

print(docs[0].page_content)