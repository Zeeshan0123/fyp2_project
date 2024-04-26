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

# llm = LlamaCpp(
#     model_path= local_llm,
#     temperature=0.3,
#     max_tokens=2048,
#     top_p=1,
#     n_ctx = 1536
# )

# print("LLM Initialized....")

# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.


# Question: {question}

# Only return the helpful answer. Answer must be detailed and well explained.
# Helpful answer:
# """

# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')

# url = "http://localhost:6333"
# client = QdrantClient(
#     url=url, prefer_grpc=False
# )

# db = Qdrant(client=client, embeddings=embed_model, collection_name="med_embeddings")    

# from langchain_core.prompts import PromptTemplate
# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


# from langchain.chains import RetrievalQA


# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.chains import LLMChain

# llm_chain = LLMChain(llm=llm, prompt=prompt)
# # retriever = MultiQueryRetriever(retriever=db.as_retriever(search_kwargs={"k":5}),llm_chain=llm_chain)

# retriever = db.as_retriever(search_kwargs={"k":2})
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

# result = qa({"query": "How do you value banks and financial institutions differently from other companies?"})
# print(result)


























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



















local_llm = "BioMistral-7B.Q4_K_M.gguf"

local_llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    n_ctx = 1536
)

from llama_index.core import Settings
Settings.llm = local_llm

print("LLM Initialized....")

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')



from llama_index.vector_stores.qdrant import QdrantVectorStore

url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

collection_name = "med_embeddings"
vector_store = QdrantVectorStore(
    client=client, 
    collection_name=collection_name,
    embed_model=embed_model
    # vector_name="base_nodes",
    
    
)



vector_index_chunk = VectorStoreIndex.from_vector_store(embed_model=embed_model,vector_store=vector_store)


from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=3,
    model="BAAI/bge-reranker-large",
)

# recursive_query_engine = vector_index_chunk.as_query_engine(
#     similarity_top_k=2,
#     # node_postprocessors=[reranker],
#     verbose=True
# )

# from llama_index.prompts import PromptTemplate

# Define your prompt
prompt_str = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
# from llama_index.core.prompts import PromptTemplate

# # Create a PromptTemplate
# prompt = PromptTemplate(prompt_str)
# # prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # Update the prompts of the query engine
# recursive_query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt})
# # response_2= recursive_query_engine.query("How do you value banks and financial institutions differently from other companies?")
# # response_1 =recursive_query_engine.query("How do you value banks and financial institutions differently from other companies?") 
# # print(response_2)

# question = "How do you value banks and financial institutions differently from other companies?"
# response = recursive_query_engine.query(question)
# print(response)






# from llama_index.core.prompts import LangchainPromptTemplate

# prompt = PromptTemplate(template=prompt_str,input_variables=["question"]) 

# lc_prompt_tmpl = LangchainPromptTemplate(
#     template=prompt,
#     # template_var_mappings={"query_str": "question"},
#     template_var_mappings={"question": "question"},
# )

# recursive_query_engine.update_prompts(
#     {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
# )

# # question = "How do you value banks and financial institutions differently from other companies?"

# # recursive_query_engine.set_input_variables({"question": question})
# # Query the engine

# response = recursive_query_engine.query({"question": "How do you value Net Operating Losses and take them into account in a valuation?"})

# # Print the response
# print(response)



from llama_index.core import Prompt

# Define a custom prompt
template = (
    "We have provided context information below. \n"
    "Use the following pieces of information to answer the user's question."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question and each answer should start with code word AI Demos: {query_str}\n"
    "Only return the helpful answer. Answer must be detailed and well explained."
    "Helpful answer:"
)
qa_template = Prompt(template)

recursive_query_engine = vector_index_chunk.as_query_engine(
    similarity_top_k=7,
    text_qa_template=qa_template,
    node_postprocessors=[reranker],
    verbose=True,
)

# Use the custom prompt when querying
# query_engine = recursive_query_engine.as_query_engine(text_qa_template=qa_template)
response = recursive_query_engine.query("How far back and forward do we usually go for public company comparable and precedent transaction multiples?")
print(response)
























# from llama_index.core.prompts import LangchainPromptTemplate

# lc_prompt_tmpl = LangchainPromptTemplate(
#     template=prompt_str,
#     template_var_mappings={"question": "query_str"},
# )

# recursive_query_engine.update_prompts(
#     {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
# )

# # Define your query and context
# query = "Your query here"
# # context = "Your context here"

# # Use the query engine to get a response
# response = recursive_query_engine.query(query)

# print(response)