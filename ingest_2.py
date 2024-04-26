import nest_asyncio

nest_asyncio.apply()

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
from langchain_community.vectorstores import Qdrant
from llama_index.vector_stores.qdrant import QdrantVectorStore

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
        result_type="markdown",  # "markdown" and "text" are available
        language='en',
        )

        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
        llama_parse_documents = reader.load_data()
        # documents = parser.load_data("./data/snell_anatomy.pdf")
        # llama_parse_documents = await parser.aload_data("./data/snell_anatomy.pdf")    

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data




# # Error while using RecursiveCharacterTextSplitter we dont directly
# # pass a documents to that first we have to make make output.md file

# def create_vector_database():
#     """
#     Creates a vector database using document loaders and embeddings.

#     This function loads urls,
#     splits the loaded documents into chunks, transforms them into embeddings using pubmedEmbeddings,
#     and finally persists the embeddings into a Qdrant vector database.

#     """
#     # Call the function to either load or parse the data
#     llama_parse_documents = load_or_parse_data()
#     # print(llama_parse_documents[1].text[:100])
#     print("Llama parser works successfully")
#     print("Now moving to the next step ....")

    
#     with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
#         for doc in llama_parse_documents:
#             f.write(doc.text + '\n')
    
#     loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
#     documents = loader.load()
#     print("Directory loader is working fine")
#     # Split loaded documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.split_documents(documents)
    
#     # from llama_index.core.node_parser import SentenceSplitter
#     # text_splitter = SentenceSplitter(
#     #     chunk_size=1024,
#     #     chunk_overlap=20,
#     # )
#     # docs = text_splitter.get_nodes_from_documents(documents)
    
#     # from llama_index.core.node_parser import SentenceSplitter
#     # from llama_index.core import Settings

#     # Settings.text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
    
#     #len(docs)
#     #docs[0]
    
#     # Initialize Embeddings
#     from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
#     embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')
    
   
#     collection_name="med_embeddings"
#     url = "http://localhost:6333/dashboard"
#     client =  qdrant_client.QdrantClient(
#         url=url,
#         prefer_grpc=False,
#         timeout=100       
#     )

#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(
#             size=768, distance=models.Distance.COSINE, on_disk=True
#         )   
#     )
    
    
#     vectorstore = Qdrant(
#         client=client,
#         collection_name=collection_name,
#         embeddings=embed_model
#     )
    
#     vectorstore.add_documents(docs)
    
#     print('Vector DB created successfully !')


# if __name__ == "__main__":
#     create_vector_database()
    
    
    
    








# llama_parse_documents = load_or_parse_data()
documents =  load_or_parse_data()
# print(llama_parse_documents[1].text[:100])
print("Llama parser works successfully")
print("Now moving to the next step ....")


# with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
#     for doc in llama_parse_documents:
#         f.write(doc.text + '\n')

# loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
# documents = loader.load()    
    

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')
      

    
collection_name="med_embeddings"
url = "http://localhost:6333/dashboard"
client =  qdrant_client.QdrantClient(
    url=url,
    prefer_grpc=False,
    timeout=100       
)

# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=models.VectorParams(
#         size=768, distance=models.Distance.COSINE, on_disk=True
#     )
#     vectors_config={
#     "nodes": models.VectorParams(
#         size=786,
#         distance=models.Distance.EUCLID,on_disk=True
#     ),
#     "objects": models.VectorParams(
#         size=786,
#         distance=models.Distance.COSINE,on_disk=True
#     ),
# }   
#   vectors_config={
#         "base_nodes": models.VectorParams(size=786, distance=models.Distance.DOT),
#         # "objects": models.VectorParams(size=786, distance=models.Distance.COSINE),
#     },
    
# )
    
    
# vectorstore = Qdrant(
#     client=client,
#     collection_name=collection_name,
#     embeddings=embed_model,
#     vector_name="base_nodes"
# )    
vectorstore = QdrantVectorStore(
    client=client, 
    collection_name=collection_name,
    embed_model=embed_model
    # vector_name="base_nodes",
    
    
)


from llama_index.core import Settings
Settings.embed_model = embed_model



from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

Settings.transformations = [LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100))]


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from llama_index.core.node_parser import LangchainNodeParser

# parser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
# nodes = parser.get_nodes_from_documents(documents)

    
  
    
    
    
    
    
    

    
    
    
    # # using Markdown with some updates
    # node_parser = MarkdownElementNodeParser(llm = None, num_workers=8)
    
    # nodes = node_parser.get_nodes_from_documents(documents)
    
    # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    
    
    # storage_context = StorageContext.from_defaults(
    # vector_store=vectorstore,
    # )
    # index_with_obj = VectorStoreIndex.from_documents(nodes=base_nodes+objects,storage_context=storage_context)  
    # # VectorStoreIndex.from_documents(nodes=nodes,storage_context = storage_context))
    
    
    # print(index_with_obj)





# text_splitter =  RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 100
# )
# texts = text_splitter.split_documents(documents)
# # text_splitter.split_text(documents)

# print("Text split successfully")

# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# embed_model = HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings-matryoshka')

# nomic_api_key = "nk-Me3vpYbbabIAKBQ6eIa7_QnoWYWV32AY9tJa56pWAME"
# nomic.cli.login(nomic_api_key)
# embed_model = NomicEmbedding(
#     api_key=nomic_api_key,
#     dimensionality=768,
#     model_name="nomic-embed-text-v1",
#     task_type="search_document"
# )

# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# storage_context = StorageContext.from_defaults(vector_store=vector_store)


# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model, chunk_size=1024,llm=None
# )
# set_global_service_context(service_context)



# index = VectorStoreIndex.from_documents(
#     documents=documents,storage_context=storage_context, service_context=service_context, show_progress=True
# )



import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
from tqdm import tqdm

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document, IndexNode, TextNode
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_SUMMARY_QUERY_STR = """\
What is this table about? Give a very concise summary (imagine you are adding a new caption and summary for this table), \
and output the real/existing table title/caption if context provided.\
and output the real/existing table id if context provided.\
and also output whether or not the table should be kept.\
"""


class TableColumnOutput(BaseModel):
    """Output from analyzing a table column."""

    col_name: str
    col_type: str
    summary: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return (
            f"Column: {self.col_name}\nType: {self.col_type}\nSummary: {self.summary}"
        )


class TableOutput(BaseModel):
    """Output from analyzing a table."""

    summary: str
    table_title: Optional[str] = None
    table_id: Optional[str] = None
    columns: List[TableColumnOutput]


class Element(BaseModel):
    """Element object."""

    id: str
    type: str
    element: Any
    title_level: Optional[int] = None
    table_output: Optional[TableOutput] = None
    table: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


class BaseElementNodeParser(NodeParser):
    """
    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects.

    Supports text and tables currently.
    """

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    llm: Optional[LLM] = Field(
        default=None, description="LLM model to use for summarization."
    )
    summary_query_str: str = Field(
        default=DEFAULT_SUMMARY_QUERY_STR,
        description="Query string to use for summarization.",
    )
    num_workers: int = Field(
        default=DEFAULT_NUM_WORKERS,
        description="Num of works for async jobs.",
    )

    show_progress: bool = Field(default=True, description="Whether to show progress.")

    @classmethod
    def class_name(cls) -> str:
        return "BaseStructuredNodeParser"

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> "BaseElementNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            callback_manager=callback_manager,
            **kwargs,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    @abstractmethod
    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""

    @abstractmethod
    def extract_elements(self, text: str, **kwargs: Any) -> List[Element]:
        """Extract elements from text."""

    def get_table_elements(self, elements: List[Element]) -> List[Element]:
        """Get table elements."""
        return [e for e in elements if e.type == "table" or e.type == "table_text"]

    def get_text_elements(self, elements: List[Element]) -> List[Element]:
        """Get text elements."""
        # TODO: There we should maybe do something with titles
        # and other elements in the future?
        return [e for e in elements if e.type != "table"]

    def extract_table_summaries(self, elements: List[Element]) -> None:
        """Go through elements, extract out summaries that are tables."""
        from llama_index.core.indices.list.base import SummaryIndex
        from llama_index.core.service_context import ServiceContext
        
        llm = self.llm
        ## Changes
#         if self.llm:
#             llm = self.llm
#         else:
#             try:
#                 from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
#             except ImportError as e:
#                 raise ImportError(
#                     "`llama-index-llms-openai` package not found."
#                     " Please install with `pip install llama-index-llms-openai`."
#                 )
#             llm = OpenAI()
        llm = cast(LLM, llm)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)

        table_context_list = []
        for idx, element in tqdm(enumerate(elements)):
            if element.type not in ("table", "table_text"):
                continue
            table_context = str(element.element)
            if idx > 0 and str(elements[idx - 1].element).lower().strip().startswith(
                "table"
            ):
                table_context = str(elements[idx - 1].element) + "\n" + table_context
            if idx < len(elements) + 1 and str(
                elements[idx - 1].element
            ).lower().strip().startswith("table"):
                table_context += "\n" + str(elements[idx + 1].element)

            table_context_list.append(table_context)
        
        ## Changes
        async def _get_table_output(table_context: str, summary_query_str: str) -> Any:
#             index = SummaryIndex.from_documents(
#                 [Document(text=table_context)], service_context=service_context
#             )
#             query_engine = index.as_query_engine(llm=llm, output_cls=TableOutput)
#             try:
#                 response = await query_engine.aquery(summary_query_str)
#                 return cast(PydanticResponse, response).response
#             except ValidationError:
#                 # There was a pydantic validation error, so we will run with text completion
#                 # fill in the summary and leave other fields blank
#                 query_engine = index.as_query_engine()
#                 response_txt = await query_engine.aquery(summary_query_str)
            return TableOutput(summary=str(table_context), columns=[])

        summary_jobs = [
            _get_table_output(table_context, self.summary_query_str)
            for table_context in table_context_list
        ]
        summary_outputs = asyncio.run(
            run_jobs(
                summary_jobs, show_progress=self.show_progress, workers=self.num_workers
            )
        )
        for element, summary_output in zip(elements, summary_outputs):
            element.table_output = summary_output

    def get_base_nodes_and_mappings(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[BaseNode], Dict]:
        """Get base nodes and mappings.

        Given a list of nodes and IndexNode objects, return the base nodes and a mapping
        from index id to child nodes (which are excluded from the base nodes).

        """
        node_dict = {node.node_id: node for node in nodes}

        node_mappings = {}
        base_nodes = []

        # first map index nodes to their child nodes
        nonbase_node_ids = set()
        for node in nodes:
            if isinstance(node, IndexNode):
                node_mappings[node.index_id] = node_dict[node.index_id]
                nonbase_node_ids.add(node.index_id)
            else:
                pass

        # then add all nodes that are not children of index nodes
        for node in nodes:
            if node.node_id not in nonbase_node_ids:
                base_nodes.append(node)

        return base_nodes, node_mappings

    def get_nodes_and_objects(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[BaseNode], List[IndexNode]]:
        base_nodes, node_mappings = self.get_base_nodes_and_mappings(nodes)

        nodes = []
        objects = []
        for node in base_nodes:
            if isinstance(node, IndexNode):
                node.obj = node_mappings[node.index_id]
                objects.append(node)
            else:
                nodes.append(node)

        return nodes, objects

    def _get_nodes_from_buffer(
        self, buffer: List[str], node_parser: NodeParser
    ) -> List[BaseNode]:
        """Get nodes from buffer."""
        doc = Document(text="\n\n".join(list(buffer)))
        return node_parser.get_nodes_from_documents([doc])

    def get_nodes_from_elements(self, elements: List[Element]) -> List[BaseNode]:
        """Get nodes and mappings."""
        # from llama_index.core.node_parser import SentenceSplitter

        # node_parser = SentenceSplitter()
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        node_parser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=70)) #here we change the Sentence splitter into Recursive character splitter
        
        nodes = []
        cur_text_el_buffer: List[str] = []
        for element in elements:
            if element.type == "table" or element.type == "table_text":
                # flush text buffer for table
                if len(cur_text_el_buffer) > 0:
                    cur_text_nodes = self._get_nodes_from_buffer(
                        cur_text_el_buffer, node_parser
                    )
                    nodes.extend(cur_text_nodes)
                    cur_text_el_buffer = []

                table_output = cast(TableOutput, element.table_output)
                table_md = ""
                if element.type == "table":
                    table_df = cast(pd.DataFrame, element.table)
                    # We serialize the table as markdown as it allow better accuracy
                    # We do not use the table_df.to_markdown() method as it generate
                    # a table with a token hungry format.
                    table_md = "|"
                    for col_name, col in table_df.items():
                        table_md += f"{col_name}|"
                    table_md += "\n|"
                    for col_name, col in table_df.items():
                        table_md += f"---|"
                    table_md += "\n"
                    for row in table_df.itertuples():
                        table_md += "|"
                        for col in row[1:]:
                            table_md += f"{col}|"
                        table_md += "\n"
                elif element.type == "table_text":
                    # if the table is non-perfect table, we still want to keep the original text of table
                    table_md = str(element.element)
                table_id = element.id + "_table"
                table_ref_id = element.id + "_table_ref"

                col_schema = "\n\n".join([str(col) for col in table_output.columns])

                # We build a summary of the table containing the extracted summary, and a description of the columns
                table_summary = str(table_output.summary)
                if table_output.table_title:
                    table_summary += ",\nwith the following table title:\n"
                    table_summary += str(table_output.table_title)

                table_summary += ",\nwith the following columns:\n"

                for col in table_output.columns:
                    table_summary += f"- {col.col_name}: {col.summary}\n"

                index_node = IndexNode(
                    text=table_summary,
                    metadata={"col_schema": col_schema},
                    excluded_embed_metadata_keys=["col_schema"],
                    id_=table_ref_id,
                    index_id=table_id,
                )

                table_str = table_summary + "\n" + table_md

                text_node = TextNode(
                    text=table_str,
                    id_=table_id,
                    metadata={
                        # serialize the table as a dictionary string for dataframe of perfect table
                        "table_df": (
                            str(table_df.to_dict())
                            if element.type == "table"
                            else table_md
                        ),
                        # add table summary for retrieval purposes
                        "table_summary": table_summary,
                    },
                    excluded_embed_metadata_keys=["table_df", "table_summary"],
                    excluded_llm_metadata_keys=["table_df", "table_summary"],
                )
                nodes.extend([index_node, text_node])
            else:
                cur_text_el_buffer.append(str(element.element))
        # flush text buffer
        if len(cur_text_el_buffer) > 0:
            cur_text_nodes = self._get_nodes_from_buffer(
                cur_text_el_buffer, node_parser
            )
            nodes.extend(cur_text_nodes)
            cur_text_el_buffer = []

        # remove empty nodes
        return [node for node in nodes if len(node.text) > 0]
    






from io import StringIO
from typing import Any, Callable, List, Optional

import pandas as pd
from llama_index.core.node_parser.relational.base_element import (
#     BaseElementNodeParser,
    Element,
)
from llama_index.core.schema import BaseNode, TextNode


def md_to_df(md_str: str) -> pd.DataFrame:
    """Convert Markdown to dataframe."""
    # Replace " by "" in md_str
    md_str = md_str.replace('"', '""')

    # Replace markdown pipe tables with commas
    md_str = md_str.replace("|", '","')

    # Remove the second line (table header separator)
    lines = md_str.split("\n")
    md_str = "\n".join(lines[:1] + lines[2:])

    # Remove the first and last second char of the line (the pipes, transformed to ",")
    lines = md_str.split("\n")
    md_str = "\n".join([line[2:-2] for line in lines])

    # Check if the table is empty
    if len(md_str) == 0:
        return None

    # Use pandas to read the CSV string into a DataFrame
    return pd.read_csv(StringIO(md_str))


class MarkdownElementNodeParser(BaseElementNodeParser):
    """Markdown element node parser.

    Splits a markdown document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    @classmethod
    def class_name(cls) -> str:
        return "MarkdownElementNodeParser"

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(),
            table_filters=[self.filter_table],
            node_id=node.id_,
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        self.extract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return self.get_nodes_from_elements(elements)

    def extract_elements(
        self,
        text: str,
        node_id: Optional[str] = None,
        table_filters: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> List[Element]:
        # get node id for each node so that we can avoid using the same id for different nodes
        """Extract elements from text."""
        lines = text.split("\n")
        currentElement = None

        elements: List[Element] = []
        # Then parse the lines
        for line in lines:
            if line.startswith("```"):
                # check if this is the end of a code block
                if currentElement is not None and currentElement.type == "code":
                    elements.append(currentElement)
                    currentElement = None
                    # if there is some text after the ``` create a text element with it
                    if len(line) > 3:
                        elements.append(
                            Element(
                                id=f"id_{len(elements)}",
                                type="text",
                                element=line.lstrip("```"),
                            )
                        )

                elif line.count("```") == 2 and line[-3] != "`":
                    # check if inline code block (aka have a second ``` in line but not at the end)
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}",
                        type="code",
                        element=line.lstrip("```"),
                    )
                elif currentElement is not None and currentElement.type == "text":
                    currentElement.element += "\n" + line
                else:
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )

            elif currentElement is not None and currentElement.type == "code":
                currentElement.element += "\n" + line

            elif line.startswith("|"):
                if currentElement is not None and currentElement.type != "table":
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
            elif line.startswith("#"):
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}",
                    type="title",
                    element=line.lstrip("#"),
                    title_level=len(line) - len(line.lstrip("#")),
                )
            else:
                if currentElement is not None and currentElement.type != "text":
                    elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
        if currentElement is not None:
            elements.append(currentElement)

        for idx, element in enumerate(elements):
            if element.type == "table":
                should_keep = True
                perfect_table = True

                # verify that the table (markdown) have the same number of columns on each rows
                table_lines = element.element.split("\n")
                table_columns = [len(line.split("|")) for line in table_lines]
                if len(set(table_columns)) > 1:
                    # if the table have different number of columns on each rows, it's not a perfect table
                    # we will store the raw text for such tables instead of converting them to a dataframe
                    perfect_table = False

                # verify that the table (markdown) have at least 2 rows
                if len(table_lines) < 2:
                    should_keep = False

                # apply the table filter, now only filter empty tables
                if should_keep and perfect_table and table_filters is not None:
                    should_keep = all(tf(element) for tf in table_filters)

                # if the element is a table, convert it to a dataframe
                if should_keep:
                    if perfect_table:
                        table = md_to_df(element.element)

                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table",
                            element=element,
                            table=table,
                        )
                    else:
                        # for non-perfect tables, we will store the raw text
                        # and give it a different type to differentiate it from perfect tables
                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table_text",
                            element=element.element,
                            # table=table
                        )
                else:
                    elements[idx] = Element(
                        id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                        type="text",
                        element=element.element,
                    )
            else:
                # if the element is not a table, keep it as to text
                elements[idx] = Element(
                    id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                    type="text",
                    element=element.element,
                )

        # merge consecutive text elements together for now
        merged_elements: List[Element] = []
        for element in elements:
            if (
                len(merged_elements) > 0
                and element.type == "text"
                and merged_elements[-1].type == "text"
            ):
                merged_elements[-1].element += "\n" + element.element
            else:
                merged_elements.append(element)
        elements = merged_elements
        return merged_elements

    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        table_df = md_to_df(table_element.element)

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1
    





# from llama_index.core import Settings
Settings.llm = None

node_parser = MarkdownElementNodeParser(llm = None, num_workers=8)
nodes = node_parser.get_nodes_from_documents(documents, progress = True)

storage_context = StorageContext.from_defaults(
    vector_store=vectorstore
)
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    node_parser=node_parser,
    llm=None,
)

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
# vectorstore.add(nodes=base_nodes)
vector_index_chunk = VectorStoreIndex.from_vector_store(
    vector_store=vectorstore, service_context=service_context
)

vector_index_chunk.insert_nodes(nodes=base_nodes)
# recursive_index = VectorStoreIndex(nodes=base_nodes+objects,storage_context=storage_context)
# recursive_index = VectorStoreIndex(nodes=base_nodes,objects=objects,storage_context=storage_context)
# recursive_index = VectorStoreIndex(nodes=base_nodes,storage_context=storage_context)
# recursive_index = VectorStoreIndex(nodes=base_nodes, storage_context=storage_context)
# nodes=nodes, objects=[obj]
# raw_index = VectorStoreIndex.from_documents(documents,storage_context=storage_context)

print("Congrats all things working file")