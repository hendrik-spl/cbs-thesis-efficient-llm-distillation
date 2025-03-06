import requests

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