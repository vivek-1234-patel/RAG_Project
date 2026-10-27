# from langchain_community.document_loaders import PyPDFLoader
# #Token based text splitter
# from langchain_text_splitters import TokenTextSplitter


# data = PyPDFLoader("Document_loaders/data_science_learning.pdf")

# docs = data.load()

# splitter = TokenTextSplitter(
#     chunk_size=100, chunk_overlap=10)

# chunks = splitter.split_documents(docs)
# print(chunks[1].page_content)

#print(len(chunks))


## Recursive Character based text splitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("Document_loaders/data_science_learning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=10)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)
