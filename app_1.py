# import nest_asyncio

# nest_asyncio.apply()

import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from langchain_community.llms import CTransformers
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores import Qdrant

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client import models


local_llm = "BioMistral-7B.Q4_K_M.gguf"

    
# config = {
#     'max_new_tokens': 1024,
#     'repetition_penality': 1.1,
#     'temperature': 0.1,
#     'top_k':6,
#     'top_p':1.1,
    
#     # 'threads': int(os.cpu_count()) / 2, 
# }

# llm = CTransformers(
#     model=local_llm,
#     model_type= "mistral",
#     lib= "avx2",
#     **config
# )

llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    n_ctx = 1024
    
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.


Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')

url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embed_model, collection_name="med_embeddings")    

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


from langchain.chains import RetrievalQA


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain

llm_chain = LLMChain(llm=llm, prompt=prompt)
# retriever = MultiQueryRetriever(retriever=db.as_retriever(search_kwargs={"k":5}),llm_chain=llm_chain)
retriever = db.as_retriever(search_kwargs={"k":1})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

result = qa({"query": "Can you give examples of major line items on each of the financial statements?"})
print(result)

# chain_type_kwargs = {"prompt": prompt}
# retriever = db.as_retriever(search_kwargs={"k":5})
# chain_input = {
# "question": "Can you give examples of major line items on each of the financial statements?"
# }
# response = qa(chain_input)
# print(response)

# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model, chunk_size=1024,
# )


# index = VectorStoreIndex.from_documents(
#     documents=documents, service_context=service_context, show_progress=True
# )


