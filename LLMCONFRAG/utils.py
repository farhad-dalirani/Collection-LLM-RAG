import os
import json


def get_query_engines_detail():
    """
    Function to get existing query engines detail like name, description, 
    embedding model that used
    """
    file_path = "./query-engines/query_engines_list.json"
    if not os.path.exists(file_path):
        return []
    vec_store_desc=[]
    with open(file_path, 'r') as file:
        vec_store_desc = json.load(file)

    return vec_store_desc


def get_query_engines_name():
    """
    Function to list name of existing query engines
    """
    vec_store_desc=get_query_engines_detail()
   
    return [vs_i['name'] for vs_i in vec_store_desc]
