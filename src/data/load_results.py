import json

def load_model_outputs_from_json(input_path):
    """
    Load the output labels from a JSON file.

    Args:
        input_path (str): The path to the JSON file containing the output labels.

    Returns:
        List[int]: A list of integers containing the entire output JSON.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return data