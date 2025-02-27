import re
from duckduckgo_search import DDGS
from knowledgeBase.text_extraction_webpages import extract_text_from_url
from types import SimpleNamespace

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

def sort_dict_by_values(my_dict):
    """
    Sorts a dictionary by its values and returns a list of tuples (key, value) in descending order of their corresponding values.
    Args:
        my_dict (dict): The dictionary to be sorted.
    Returns:
        list: A list of tuples (key, value) sorted based on their corresponding values in descending order.
    """
    # Sort the dictionary by its values and extract the keys and values as tuples
    sorted_items = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
    return sorted_items

def format_collection_name(name: str) -> str:
    """
    Formats a collection name by applying several transformations to ensure it is valid.
    The function performs the following steps:
    1. Removes leading and trailing spaces.
    2. Replaces invalid characters (anything other than alphanumeric, underscore, or hyphen) with an underscore.
    3. Ensures the name starts and ends with an alphanumeric character.
    4. Ensures there are no consecutive periods (though periods are replaced with underscores).
    5. Truncates the name to 63 characters if it is too long.
    6. Provides a fallback name "default_name" if the resulting name is empty.
    Args:
        name (str): The original collection name to be formatted.
    Returns:
        str: The formatted collection name.
    """
    
    # Remove leading/trailing spaces
    name = name.strip()

    # Replace invalid characters (anything other than alphanumeric, _, or -) with an underscore
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    # Ensure the name starts and ends with an alphanumeric character
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)  # Remove invalid leading characters
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)  # Remove invalid trailing characters

    # Ensure no consecutive periods (though periods were already replaced with _)
    name = re.sub(r'\.\.', '_', name)

    # Truncate to 63 characters if too long
    return name[:63] if name else "default_name"  # Provide a fallback name if empty


def internet_search(query: str) -> SimpleNamespace:
    """
    Perform an internet search using the DDGS (DuckDuckGo Search) API and return structured search results.

    Args:
        query (str): The search query string.

    Returns:
        SimpleNamespace: An object with attributes:
            - formatted_results: str
            - source_nodes: list of SimpleNamespace objects, each containing a 'node' (with metadata) and a 'score'.
    
    In case of an error during the search, returns a SimpleNamespace with an 'error' attribute describing the issue.
    """
    try:
        response = DDGS().text(query, max_results=5)
    except Exception as e:
        return SimpleNamespace(error=f"An error occurred while searching: {e}.")

    source_nodes = []
    formatted_results = ""
    for r in response:
        content = extract_text_from_url(url=r["href"])
        # Create the inner node with metadata.
        inner_node = SimpleNamespace(metadata={
            "Name": "Internet Search Tool-" + r["title"],
            "Link": r["href"]
        })
        # Wrap in an outer node that includes a score attribute.
        node_object = SimpleNamespace(node=inner_node, score=None)
        source_nodes.append(node_object)
        
        formatted_results += f"Title: {r['title']}\n"
        formatted_results += f"Content: {content}\n"
        formatted_results += "----\n\n"

    return SimpleNamespace(formatted_results=formatted_results, source_nodes=source_nodes)
