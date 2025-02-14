import os
import json


def get_query_engines_detail():
    """
    Retrieves details of existing query engines from a JSON file.

    This function reads a JSON file containing details about query engines,
    such as their name, description, and the embedding model used. If the file
    does not exist, it returns an empty list.

    Returns:
        list: A list of dictionaries, each containing details of a query engine.
    """
    file_path = "./query-engines/query_engines_list.json"
    if not os.path.exists(file_path):
        return []
    vec_store_desc=[]
    with open(file_path, 'r') as file:
        vec_store_desc = json.load(file)

    return vec_store_desc


def get_query_engines_detail_by_name(query_engine_names):
    """
    Retrieve details of specific query engines by their names.

    This function filters and returns details of query engines such as name, 
    description, and the embedding model used, based on the provided list of 
    query engine names.

    Args:
        query_engine_names (list of str): A list of query engine names to filter by.

    Returns:
        list of dict: A list of dictionaries containing details of the query engines 
                      that match the provided names.
    """
    vec_store_desc = get_query_engines_detail()

    filtered_vec_store_desc = []
    for qe_i in vec_store_desc:
        if qe_i['name'] in query_engine_names:
            filtered_vec_store_desc.append(qe_i)

    return filtered_vec_store_desc


def get_query_engines_name():
    """
    Retrieves the names of existing query engines.
    This function calls `get_query_engines_detail()` to get detailed information
    about the query engines and extracts their names from the returned data.
    Returns:
        list: A list of names of the existing query engines.
    """
    vec_store_desc=get_query_engines_detail()
   
    return [vs_i['name'] for vs_i in vec_store_desc]


def remove_duplicates_pairs(pairs):
    """
    Remove duplicate pairs from a list of pairs.
    Args:
        pairs (list of tuple): A list of pairs (tuples) from which duplicates need to be removed.
    Returns:
        list of tuple: A list containing only unique pairs from the input list, preserving the original order.
    """
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    return unique_pairs