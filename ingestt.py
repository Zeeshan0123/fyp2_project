import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')

# print(embeddings)

loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

texts = text_splitter.split_documents(documents)

print(texts[1])

url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embed_model,
    url=url,
    prefer_grpc=False,
    collection_name="med_embeddings"
)

print("Vector DB Successfully Created!")