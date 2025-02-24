import os
import shutil
import json
import logging
from openai import AuthenticationError

import chromadb
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import VectorStoreIndex

from knowledgeBase.text_extraction_webpages import scrape_articles, scrape_pdfs
from utils import format_collection_name


class CollectionManager:

    def __init__(self, scraped_data_path='Data/output-processed-sources', 
                 vector_index_save_path='Data/query-engines/collections', 
                 keyword_index_save_path='Data/query-engines/keyword-index/', 
                 query_engines_info_json='Data/query-engines/query_engines_list.json'):
        self.scraped_data_path = scraped_data_path
        self.vector_index_save_path = vector_index_save_path
        self.keyword_index_save_path = keyword_index_save_path
        self.query_engines_info_json = query_engines_info_json

    def create_new_collection(self, user_models, path_json_file, type_json):
        """
        Creates a new collection by processing the input JSON file and generating vector and keyword indices.
        Args:
            user_models (UserModels): The user models used for creating the collection.
            path_json_file (str): The path to the input JSON file containing the data.
            type_json (str): The type of JSON file, either 'Webpages' or 'PDFs'.
        Raises:
            ValueError: If the type_json is not 'Webpages' or 'PDFs'.
            FileNotFoundError: If the output file is not found.
            ValueError: If the output file contains invalid JSON format.
        Returns:
            None
        """
        
        file_name = os.path.basename(path_json_file)
        dot_location = file_name.find('.')
        file_name_no_exten = file_name[0:dot_location]

        file_name_no_exten = format_collection_name(name=file_name_no_exten)

        # Extract text content of each entities in input json file
        output_file = None
        if type_json == 'Webpages':
            try:
                output_file = scrape_articles(
                    json_file=path_json_file, 
                    output_file=os.path.join(self.scraped_data_path, file_name)
                )
            except Exception as e:
                logging.error("An error occured: {}".format(e))
                output_file = None
        elif type_json == 'PDFs':
            try:
                output_file = scrape_pdfs(
                    json_file=path_json_file, 
                    output_file=os.path.join(self.scraped_data_path, file_name)
                )
            except Exception as e:
                logging.error("An error occured: {}".format(e))
                output_file = None
        else:
            raise ValueError('Selected Type of JSON file is incorrect.')

        try:
            with open(output_file, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("The file was not found: {}.",format(output_file))  # Raising error here
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format: {}.".format(output_file))

        # Convert text to Document object
        documents = []
        for entity_i in data['data']:
            documents.append(Document(
                text=entity_i['Content'], 
                metadata={'Link': entity_i['Link'], 'Name': entity_i['Name']}, 
                excluded_llm_metadata_keys=[
                        "Name",
                        "Link",
                    ],
                excluded_embed_metadata_keys=[
                        "Link"                    
                    ],
                )
            ) 

        # Create vector index
        nodes = self.__create_vector_index(
                user_models=user_models, 
                documents=documents, 
                collection_name=file_name_no_exten
            )
        
        # Create keyword index
        self.__create_keyword_index(
                nodes=nodes, 
                collection_name=file_name_no_exten, 
                model_llm=user_models.model_llm
            )

        # Save the details of the created vector store
        self.__save_query_engine_info(
                user_models=user_models, 
                collection_name=file_name_no_exten, 
                collection_description=data['description']
            )

    def __create_vector_index(self, user_models, documents, collection_name):
        """
        Creates a vector index for the given documents using the specified user models and collection name.
        Args:
            user_models (object): An object containing user-defined models for embedding.
            documents (list): A list of documents to be indexed.
            collection_name (str): The name of the collection to be created in the vector database.
        Returns:
            list: A list of nodes resulting from the transformation pipeline.
        Raises:
            ValueError: If an authentication error occurs or any other unexpected error is encountered.
        """
        # Path to save collection
        collection_path = os.path.join(self.vector_index_save_path, collection_name)

        #Vector based database to store docs, their embeddings, ...
        logging.info(">    Creating {} Vector Index ...".format(collection_name))
        chroma_client = chromadb.PersistentClient(path=collection_path)
        chroma_collection = chroma_client.create_collection(name=collection_name)
        # Define a storage context object using the created vector database.
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    

        token_spliter = TokenTextSplitter(chunk_size=800, chunk_overlap=0, separator=" ")
        
        # Create the pipeline to apply the transformation on each document,
        # and store the transformed nodes in the vector store.
        pipeline = IngestionPipeline(
            transformations=[
                token_spliter, # Split documents to chunks
                user_models.model_embd, # Convert to embedding vector
            ],
            vector_store=vector_store
        )

        # Run the transformation pipeline.
        try:
            nodes = pipeline.run(documents=documents, show_progress=True)
        except AuthenticationError:
            raise ValueError("Authentication error: Incorrect API key provided.")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {e}")

        return nodes

    def __create_keyword_index(self, nodes, collection_name, model_llm):
        """
        Creates a keyword index for the given nodes and collection name.
        This method initializes a SimpleKeywordTableIndex with the provided nodes and LLM model,
        logs the creation process, and persists the index to a specified directory.
        Args:
            nodes (list): A list of nodes to be indexed.
            collection_name (str): The name of the collection for which the keyword index is being created.
            model_llm (object): The language model to be used for creating the keyword index.
        Returns:
            None
        """
        logging.info(">    Creating {} Keyword Index ...".format(collection_name))
        # Initialize the SimpleKeywordTableIndex with the service context
        keyword_index = SimpleKeywordTableIndex(nodes=nodes, llm=model_llm, show_progress=True)

        # Define the directory path
        os.makedirs(self.keyword_index_save_path, exist_ok=True)

        # Persist the index with a specific ID
        persist_directory = os.path.join(self.keyword_index_save_path, collection_name)
        keyword_index.storage_context.persist(persist_directory)

    def __save_query_engine_info(self, user_models, collection_name, collection_description):
        """
        Saves information about the query engine to a JSON file.
        This method adds details of the created vector store to a list of vector stores
        stored in a JSON file. If the JSON file does not exist, it creates an empty list
        and then appends the new entry.
        Args:
            user_models: An object containing user model information, specifically the embedding name.
            collection_name (str): The name of the collection to be saved.
            collection_description (str): A description of the collection to be saved.
        Raises:
            IOError: If there is an error reading or writing to the JSON file.
        """        
        # Add detail of created vector store to list of vector stores
        if not os.path.exists(self.query_engines_info_json):
            with open(self.query_engines_info_json, 'w') as file:
                json.dump([], file)
        vec_store_desc=[]
        with open(self.query_engines_info_json, 'r') as file:
            vec_store_desc = json.load(file)
            new_entry = {
                        "name": collection_name,
                        "description": collection_description,
                        "embedding_name": user_models.embedding_name
                    }
            vec_store_desc.append(new_entry)
        with open(self.query_engines_info_json, 'w') as file:
                json.dump(vec_store_desc, file)

    def delete_query_engine_by_name(self, name):
        """
        Deletes a query engine by its name.
        This method performs the following actions:
        1. Deletes the vector store associated with the query engine.
        2. Deletes the keyword index directory associated with the query engine.
        3. Updates the list of query engines by removing the entry with the specified name.
        Args:
            name (str): The name of the query engine to be deleted.
        Raises:
            FileNotFoundError: If the query engines info JSON file does not exist.
            json.JSONDecodeError: If the query engines info JSON file contains invalid JSON.
        """

        # Path to save collection
        collection_path = os.path.join(self.vector_index_save_path, name)
        
        # Delete the vector store
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
            print("The folder has been deleted successfully!")
        else:
            print("The folder does not exist.")

        # Delete the keyword index
        directory_path = self.keyword_index_save_path
        persist_directory = os.path.join(directory_path, name)
        os.system("rm -rf {}".format(persist_directory))

        # Update the list of query engines
        with open(self.query_engines_info_json, 'r') as file:
            vec_store_desc = json.load(file)
            vec_store_desc = [i for i in vec_store_desc if i['name'] != name]
        with open(self.query_engines_info_json, 'w') as file:
            json.dump(vec_store_desc, file)

    def load_vector_index_from_file(self, query_engine_name, model_embd):
        """
        Load a vector index from a file based on the query engine name and embedding model.
        Args:
            query_engine_name (str): The name of the query engine to load.
            model_embd: The embedding model to use for the vector store index.
        Returns:
            VectorStoreIndex: The loaded vector store index if the query engine is found, otherwise None.
        """
                
        qe_details = self.get_query_engines_detail()
        
        loc = -1
        for idx, qe_i in enumerate(qe_details):
            if qe_i['name'] == query_engine_name:
                loc = idx
                break

        if loc == -1:
            return None

        # Path to save collection
        collection_path = os.path.join(self.vector_index_save_path, query_engine_name)

        # Load query engine from database
        chroma_client = chromadb.PersistentClient(path=collection_path)
        chroma_collection = chroma_client.get_collection(name=query_engine_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=model_embd)
        return vector_store_index

    def load_keyword_index_from_file(self, query_engine_name, model_llm):
        """
        Load the keyword index from a file.
        This method rebuilds the storage context using the specified query engine name
        and loads the keyword index from the storage using the provided LLM model.
        Args:
            query_engine_name (str): The name of the query engine.
            model_llm (Any): The language model to be used for loading the index.
        Returns:
            keyword_index: The loaded keyword index.
        """

        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(
                persist_dir=os.path.join(self.keyword_index_save_path, query_engine_name)
            )
        keyword_index = load_index_from_storage(storage_context=storage_context, index_id=None, llm=model_llm)
        return keyword_index


    def get_query_engines_detail(self):
        """
        Retrieves the details of query engines from a JSON file.
        This method checks if the JSON file specified by `self.query_engines_info_json` exists.
        If the file does not exist, it returns an empty list. If the file exists, it reads the
        contents of the file and returns it as a list.
        Returns:
            list: A list containing the details of query engines. If the file does not exist,
                  an empty list is returned.
        """

        if not os.path.exists(self.query_engines_info_json):
            return []
        vec_store_desc=[]
        with open(self.query_engines_info_json, 'r') as file:
            vec_store_desc = json.load(file)

        return vec_store_desc

    def get_query_engines_detail_by_name(self, query_engine_names):
        """
        Retrieves detailed information about specific query engines by their names.
        Args:
            query_engine_names (list of str): A list of query engine names to filter the details.
        Returns:
            list of dict: A list of dictionaries containing the details of the query engines 
                          that match the provided names.
        """
        
        vec_store_desc = self.get_query_engines_detail()

        filtered_vec_store_desc = []
        for qe_i in vec_store_desc:
            if qe_i['name'] in query_engine_names:
                filtered_vec_store_desc.append(qe_i)

        return filtered_vec_store_desc

    def get_query_engines_name(self):
        """
        Retrieve the names of query engines.
        This method fetches the details of query engines and extracts their names.
        Returns:
            list: A list of names of the query engines.
        """
                
        vec_store_desc=self.get_query_engines_detail()
    
        return [vs_i['name'] for vs_i in vec_store_desc]