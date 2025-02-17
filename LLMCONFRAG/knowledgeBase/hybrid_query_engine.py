import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever

from typing import List

from utils import get_query_engines_detail


class HybridRetriever(BaseRetriever):
    """
    HybridRetriever is a class that combines both vector-based and keyword-based retrieval mechanisms to fetch relevant nodes based on a query.
    Attributes:
        query_engine_name (str): The name of the query engine.
        query_engine_description (str): A description of the query engine.
        model_llm: The language model used for keyword-based retrieval.
        model_embd: The embedding model used for vector-based retrieval.
        _vector_retriever: An instance of VectorIndexRetriever for vector-based retrieval.
        _keyword_retriever: An instance of KeywordTableSimpleRetriever for keyword-based retrieval.
    Methods:
        __init__(model_llm, model_embd, query_engine_name, query_engine_description, k_semantic=16, k_keyword=6):
            Initializes the HybridRetriever with the given models, query engine details, and retrieval parameters.
        __load_vector_index_from_file():
            Loads the vector index from a file based on the query engine name.
        __load_keyword_index_from_file():
            Loads the keyword index from a file based on the query engine name.
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Retrieves relevant nodes based on the given query bundle using both vector and keyword retrieval mechanisms.
    """
    
   

    def __init__(self, model_llm, model_embd, query_engine_name, query_engine_description, k_semantic=16, k_keyword=6)-> None:
        """
        Initializes the HybridRetriever with the given models, query engine details, and retrieval parameters.
        """
        self.query_engine_name = query_engine_name
        self.query_engine_description = query_engine_description
        self.model_llm = model_llm
        self.model_embd = model_embd
        
        self.__vector_index_save_path='Data/query-engines/collections'
        self.__keyword_index_save_path='Data/query-engines/keyword-index'
        self.__query_engines_info_json='Data/query-engines/query_engines_list.json'

        # Load the vector index and keyword index
        vector_index = self.__load_vector_index_from_file()
        keyword_index = self.__load_keyword_index_from_file()

        self._vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=k_semantic)
        self._keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, num_chunks_per_query=k_keyword)
    
    def __load_vector_index_from_file(self):
        """
        Loads a vector index from a file based on the input name.
        This method retrieves the details of vector index and searches for the 
        vector index with the specified name. If found, it loads the corresponding 
        vector index from a persistent Chroma database.
        Returns:
            VectorStoreIndex: The vector store index associated with the query engine name.
            None: If the query engine name is not found in the details.
        """
        
        qe_details = get_query_engines_detail()
        
        loc = -1
        for idx, qe_i in enumerate(qe_details):
            if qe_i['name'] == self.query_engine_name:
                loc = idx
                break

        if loc == -1:
            return None

        # Load query engine from database
        chroma_client = chromadb.PersistentClient(path=self.__vector_index_save_path)
        chroma_collection = chroma_client.get_collection(name=self.query_engine_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.model_embd)
        return vector_store_index

    def __load_keyword_index_from_file(self):
        """
        Load the keyword index from a file.
        This method rebuilds the storage context using the default settings and 
        loads the keyword index from the specified storage directory.
        Returns:
            keyword_index: The loaded keyword index.
        """
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(
                persist_dir=os.path.join(self.__keyword_index_save_path, self.query_engine_name)
            )
        keyword_index = load_index_from_storage(storage_context=storage_context, index_id=None, llm=self.model_llm)
        return keyword_index


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes based on the given query bundle by combining results from
        vector and keyword retrievers.
        Args:
            query_bundle (QueryBundle): The query bundle containing the query information.
        Returns:
            List[NodeWithScore]: A list of nodes with scores that match the query,
                                 combining results from both vector and keyword retrieval.
        """
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        resulting_nodes = []
        node_ids_added = set()

        # Process all nodes from both lists
        for vector_node in vector_nodes:
            if vector_node.node.node_id not in node_ids_added:
                resulting_nodes.append(vector_node)
                node_ids_added.add(vector_node.node.node_id)

        for keyword_node in keyword_nodes:
            if keyword_node.node.node_id not in node_ids_added:
                resulting_nodes.append(keyword_node)
                node_ids_added.add(keyword_node.node.node_id)

        return resulting_nodes


def load_hybrid_query_engine(model_llm, model_embd, query_engine_name, query_engine_description, k_semantic=18, k_keyword=6):
    """
    Load and configure a hybrid query engine that combines semantic and keyword-based retrieval.
    Args:
        model_llm: The language model to be used for semantic understanding and response synthesis.
        model_embd: The embedding model to be used for keyword-based retrieval.
        query_engine_name (str): The name of the query engine.
        query_engine_description (str): A description of the query engine.
        k_semantic (int, optional): The number of top results to retrieve based on semantic similarity. Defaults to 16.
        k_keyword (int, optional): The number of top results to retrieve based on keyword matching. Defaults to 5.
    Returns:
        hybrid_query_engine: An instance of RetrieverQueryEngine configured with the specified models and parameters.
    """

    # Hybrid retriever to combine vector and keyword-based retrieval
    hybrid_retriever = HybridRetriever(
                            model_llm=model_llm,
                            model_embd=model_embd, 
                            query_engine_name=query_engine_name, 
                            query_engine_description=query_engine_description, 
                            k_semantic=k_semantic, 
                            k_keyword=k_keyword
                        )
    
    # Reranker to sort retrieved results according to relevance to query by using the language model
    k_total = k_semantic + k_keyword
    num_keep_nodes = max(1, k_total//2)
    rankGPT  = RankGPTRerank(top_n=num_keep_nodes, llm=model_llm, verbose=True)
    
    response_synthesizer = get_response_synthesizer(llm=model_llm)
    
    hybrid_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rankGPT]
    )

    return hybrid_query_engine