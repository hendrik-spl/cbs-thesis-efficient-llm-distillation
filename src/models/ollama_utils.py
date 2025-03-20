import time
import ollama
from ollama import chat, ChatResponse

from src.models.model_utils import clean_llm_output_sentiment, find_majority

def query_ollama_sc(model, prompt, shots=5):
    """
    Implement self-consistency on top of Ollama API.

    Args:
        model (str): The name of the model to use. For example, "llama3.2:1b".
        prompt (str): The prompt to send to the model.
        shots (int): The number of shots to use for self-consistency.
    """
    responses = []
    for i in range(shots):
        response = query_ollama(model, prompt)
        responses.append(clean_llm_output_sentiment(response))
    
    majority_vote = find_majority(responses)
    return majority_vote

def query_ollama(model, prompt, temperature=0.1, seed=42, max_retries=3, retry_delay=5):
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
    messages = [{"role": "user", "content": prompt}]
    options = {"temperature": temperature, "seed": seed}

    for attempt in range(max_retries):
        try:
            response: ChatResponse = chat(
                model=model,
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

def check_if_ollama_model_exists(model_name):
    """
    Checks if a model exists in the Ollama API.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
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