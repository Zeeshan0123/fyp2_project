# import nest_asyncio

# nest_asyncio.apply()

import os

from llama_parse import LlamaParse


from llama_index.core import StorageContext
from llama_index.core import set_global_service_context
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores import Qdrant

import qdrant_client
from qdrant_client import models

import pickle

def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        print("here sdfhsidfhjaksdf")
        
        # parser = LlamaParse(api_key='llx-5kzdBlTk6VJLcAcfpaLZrVRfEX4uNtpEiDh7yr9lKsuBGhgq', result_type="markdown", verbose=True)
        # llama_parse_documents = parser.load_data("./data/400_Questions_Technicals.pdf")
        parser = LlamaParse(
        api_key="llx-5kzdBlTk6VJLcAcfpaLZrVRfEX4uNtpEiDh7yr9lKsuBGhgq",  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown"  # "markdown" and "text" are available
        )

        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
        llama_parse_documents = reader.load_data()
                

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data


# Error while using RecursiveCharacterTextSplitter we dont directly
# pass a documents to that first we have to make make output.md file

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using pubmedEmbeddings,
    and finally persists the embeddings into a Qdrant vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    # print(llama_parse_documents[1].text[:100])
    print("Llama parser works successfully")
    print("Now moving to the next step ....")

    
    with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')
    
    loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    print("Directory loader is working fine")
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # from llama_index.core.node_parser import SentenceSplitter
    # text_splitter = SentenceSplitter(
    #     chunk_size=1024,
    #     chunk_overlap=20,
    # )
    # docs = text_splitter.get_nodes_from_documents(documents)
    
    # from llama_index.core.node_parser import SentenceSplitter
    # from llama_index.core import Settings

    # Settings.text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
    
    #len(docs)
    #docs[0]
    
    # Initialize Embeddings
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')
    
   
    collection_name="med_embeddings"
    url = "http://localhost:6333/dashboard"
    client =  qdrant_client.QdrantClient(
        url=url,
        prefer_grpc=False,
        timeout=100       
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768, distance=models.Distance.COSINE, on_disk=True
        )   
    )
    
    
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embed_model
    )
    
    vectorstore.add_documents(docs)
    
    print('Vector DB created successfully !')


if __name__ == "__main__":
    create_vector_database()