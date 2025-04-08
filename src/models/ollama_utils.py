import os
import time
import ollama
from ollama import chat, ChatResponse

ollama_model_mapping = {
    "llama3.1:8b": "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:405b": "llama3.1:405b-instruct-q4_K_M",
    "llama3.2:1b": "llama3.2:1b-instruct-q4_K_M",
    "llama3.3:70b": "llama3.3:70b-instruct-q4_K_M",
}

def query_ollama_model(model_config, prompt, params):
    """
    Sends a chat request to the Ollama API with the given model and prompt using the Ollama SDK.

    Args:
        model (str): The name of the model to use. For example, "llama3.2:1b".
        prompt (str): The prompt to send to the model.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.1.
        seed (int, optional): The seed for reproducibility. Defaults to 42.
        max_retries (int, optional): Maximum number of retries on failure. Defaults to 3.
        retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.

    Returns:
        str or None: The response content if the request was successful, None otherwise.
    """
    model_name = ollama_model_mapping.get(model_config, model_config)
    print("Using ollama with model:", model_name)

    messages = [{"role": "user", "content": prompt}]
    options = {
        "temperature": params.get("temperature"),
        "seed": params.get("seed"),
        "top_p": params.get("top_p"),
        "top_k": params.get("top_k"),
        "num_predict": params.get("max_new_tokens"),
        }

    max_retries = params.get("custom_max_retries")
    retry_delay = params.get("custom_retry_delay")
    
    for attempt in range(max_retries):
        try:
            response: ChatResponse = chat(
                model=model_name,
                messages=messages,
                options=options
            )
            return response.message.content
        except Exception as e:
            print(f"Error in Ollama request (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get response from Ollama after multiple attempts.")
                return None
            time.sleep(retry_delay)

def use_ollama(model_name: str) -> bool:
    """
    Check if the model is an Ollama model.

    Args:
        model_name: The name of the model to check.

    Returns:
        bool: True if the model is an Ollama model, False otherwise.
    """
    if os.path.exists(model_name):
        return False
    else:
        return True

def check_if_ollama_model_exists(model_name):
    """
    Checks if a model exists in the Ollama API.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    model_name = ollama_model_mapping.get(model_name, model_name)
    try:
        list = ollama.list()
        if len(list.models) == 0:
            print("No ollama models available yet. Pulling...")
            pull_model_from_ollama(model_name)
            return False
        for model in list:
            if model[1][0].model == model_name:
                return True
            else:
                print(f"Model {model_name} not found in Ollama. Attempting to pull model.")
                pull_model_from_ollama(model_name)
                return False
    except Exception as e:
        print(f"Failed to check if model {model_name} exists: {e}")
        return False

def pull_model_from_ollama(model_name):
    """
    Pulls a model from the Ollama API.

    Args:
        model_name (str): The name of the model to pull.
    """
    try:
        ollama.pull(model_name)
        print(f"Model {model_name} pulled successfully.")
    except Exception as e:
        print(f"Failed to pull model {model_name}: {e}")