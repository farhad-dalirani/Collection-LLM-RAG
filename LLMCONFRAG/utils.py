import os


def get_folders():
    """
    Function to list only folders inside 'query-engines'
    """
    base_path = "./query-engines"
    if not os.path.exists(base_path):
        return ["No folders found"]
    return [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
