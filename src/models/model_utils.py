import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_from_hf(model_name, num_labels):
    """
    Loads a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.
        num_labels (int): The number of labels for the model.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    if ":" in model_name: # Handle model names with a slash in them for HF
        model_name = model_name.replace(":", "/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def query_ollama(model, prompt, temperature=0.1, seed=42):
    """
    Sends a chat request to the Ollama API with the given model and prompt.

    Args:
        model (str): The name of the model to use. For example, "llama3.2:1b".
        prompt (str): The prompt to send to the model.

    Returns:
        dict or None: The response from the API as a dictionary if the request was successful, None otherwise.
    """
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "temperature": temperature,
            "seed": seed
        },
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None