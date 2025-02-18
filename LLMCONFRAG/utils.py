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