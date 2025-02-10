import os
import json
from llama_index.core import Document

from createKnowledgeBase.text_extraction_webpages import scrape_articles


def create_new_query_engine(path_json_file, type_json, output_path='LLMCONFRAG/createKnowledgeBase/output-processed-sources'):
    """
    """
    
    # Extract text content of each entities in input json file
    output_file = None
    if type_json == 'Webpages':
        try:
            output_file = scrape_articles(
                json_file=path_json_file, 
                output_file=os.path.join(output_path, os.path.basename(path_json_file))
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
        documents.append(Document(text=entity_i['Content'], metadata={'Link': entity_i['Link'], 'Name': entity_i['Name']})) 

    

if __name__ == '__main__':
    # Example usage:
    create_new_query_engine("LLMCONFRAG/createKnowledgeBase/input-sources/test.json", 'Webpages')
