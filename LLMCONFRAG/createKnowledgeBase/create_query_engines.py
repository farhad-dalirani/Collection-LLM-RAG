import os
import json

import chromadb
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline

from createKnowledgeBase.text_extraction_webpages import scrape_articles


def create_new_query_engine(user_models, path_json_file, type_json, output_path='LLMCONFRAG/createKnowledgeBase/output-processed-sources'):
    """
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
                output_file=os.path.join(output_path, file_name)
            )
        except Exception as e:
            print("An error occured: {}".format(e))
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

    # Vector based database to store docs, their embeddings, ...
    print(file_name_no_exten)
    chroma_client = chromadb.PersistentClient(path='./query-engines/collections')
    chroma_collection = chroma_client.create_collection(name=file_name_no_exten)
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
    nodes = pipeline.run(documents=documents, show_progress=True);

    # Add detail of created vector store to list of vector stores
    file_path = "./query-engines/query_engines_list.json"
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump([], file)
    vec_store_desc=[]
    with open(file_path, 'r') as file:
        vec_store_desc = json.load(file)
        new_entry = {
                    "description": data['description'],
                    "embedding_name": user_models.embedding_name
                }
        vec_store_desc.append(new_entry)
    with open(file_path, 'w') as file:
            json.dump(vec_store_desc, file)
    
    print('>    Nodes were created.')


if __name__ == '__main__':
    # Example usage:
    create_new_query_engine("LLMCONFRAG/createKnowledgeBase/input-sources/test.json", 'Webpages')
