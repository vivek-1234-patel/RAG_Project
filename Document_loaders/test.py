from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter
# character based text splitter
splitter = CharacterTextSplitter(
    separator = " ",
    chunk_size = 100,
    chunk_overlap=10
)

data = TextLoader(
    "Document_loaders/rag_learning.txt",
    encoding="utf-8"
)

docs = data.load()

print(docs[0].page_content)