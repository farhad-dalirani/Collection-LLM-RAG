import os
import json
import logging
from openai import AuthenticationError

import chromadb
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleKeywordTableIndex

from knowledgeBase.text_extraction_webpages import scrape_articles


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
        Creates a new collection (query engine) by processing a JSON file containing entities, extracting their text content,
        converting the text to document objects, and storing the documents in a vector-based database (both vector index and keyword index).
        Args:
            user_models: A user-defined model object that includes an embedding model.
            path_json_file (str): The path to the input JSON file containing entities.
            type_json (str): The type of JSON file, either 'Webpages' or 'PDFs'.
            output_path (str, optional): The path to store the processed output files. Defaults to 'Data/output-processed-sources'.
        Raises:
            ValueError: If the type_json is not 'Webpages' or 'PDFs'.
            FileNotFoundError: If the output file is not found.
            ValueError: If the output file contains invalid JSON.
        Returns:
            None
        """
        file_name = os.path.basename(path_json_file)
        dot_location = file_name.find('.')
        file_name_no_exten = file_name[0:dot_location]

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
            pass
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
        Creates a vector index for a collection of documents using a specified user model for embeddings.
        Args:
            user_models (object): An object containing the user-defined models for embedding.
            documents (list): A list of documents to be indexed.
            collection_name (str): The name of the collection to be created.
            collection_description (str): A description of the collection.
        Returns:
            list: A list of nodes representing the transformed documents stored in the vector index.
        The function performs the following steps:
        1. Initializes a persistent client for the vector database.
        2. Creates a collection in the vector database with the specified name.
        3. Defines a storage context object using the created vector database.
        4. Sets up a token splitter to chunk the documents.
        5. Creates an ingestion pipeline to transform the documents and store the transformed nodes in the vector store.
        6. Runs the transformation pipeline on the provided documents.
        7. Updates a JSON file with details of the created vector store.
        8. Prints status messages indicating the progress and completion of the vector index creation.
        """

        #Vector based database to store docs, their embeddings, ...
        logging.info(">    Creating {} Vector Index ...".format(collection_name))
        chroma_client = chromadb.PersistentClient(path=self.vector_index_save_path)
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
        Creates a keyword index for a given collection of nodes using a specified language model.
        This function initializes a SimpleKeywordTableIndex with the provided nodes and language model,
        then saves the index to a specified directory.
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
        Deletes a query engine (collection) by its name.
        Args:
            name (str): The name of the query engine to be deleted.
        Returns:
            None
        """
        # Delete the vector store
        chroma_client = chromadb.PersistentClient(path=self.vector_index_save_path)
        chroma_client.delete_collection(name)

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