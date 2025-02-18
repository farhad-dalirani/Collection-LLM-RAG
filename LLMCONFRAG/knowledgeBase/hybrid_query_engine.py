from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever

from typing import List
from knowledgeBase.collection import CollectionManager

class HybridRetriever(BaseRetriever):
    """
    A retriever that combines vector-based and keyword-based retrieval methods to
    retrieve relevant nodes based on a given query.
    Attributes:
        query_engine_name (str): The name of the query engine.
        query_engine_description (str): A description of the query engine.
        model_llm: The language model used for keyword-based retrieval.
        model_embd: The embedding model used for vector-based retrieval.
        _vector_retriever (VectorIndexRetriever): The retriever for vector-based retrieval.
        _keyword_retriever (KeywordTableSimpleRetriever): The retriever for keyword-based retrieval.
    Methods:
        __init__(model_llm, model_embd, query_engine_name, query_engine_description, k_semantic=16, k_keyword=6):
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
    """
    
    def __init__(self, model_llm, model_embd, query_engine_name, query_engine_description, k_semantic=16, k_keyword=6)-> None:
        """
        Initializes the HybridRetriever with the given models, query engine details, and retrieval parameters.
        """
        self.query_engine_name = query_engine_name
        self.query_engine_description = query_engine_description
        self.model_llm = model_llm
        self.model_embd = model_embd
        
        collection_manager = CollectionManager()

        # Load the vector index and keyword index
        vector_index = collection_manager.load_vector_index_from_file(query_engine_name=query_engine_name, model_embd=model_embd)
        keyword_index = collection_manager.load_keyword_index_from_file(query_engine_name=query_engine_name, model_llm=model_llm)

        self._vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=k_semantic)
        self._keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, num_chunks_per_query=k_keyword)

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
    Load a hybrid query engine that combines vector-based and keyword-based retrieval methods.
    Args:
        model_llm (object): The language model to be used for semantic understanding and reranking.
        model_embd (object): The embedding model to be used for vector-based retrieval.
        query_engine_name (str): The name of the query engine.
        query_engine_description (str): A description of the query engine.
        k_semantic (int, optional): The number of top results to retrieve using semantic search. Defaults to 18.
        k_keyword (int, optional): The number of top results to retrieve using keyword search. Defaults to 6.
    Returns:
        object: An instance of the hybrid query engine.
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