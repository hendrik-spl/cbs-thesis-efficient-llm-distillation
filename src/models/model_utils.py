import time
import ollama
from ollama import chat, ChatResponse
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
            if "not found" in str(e) and "model" in str(e):
                print("Model not found. Attempting to pull model from ollama.")
                pull_model_from_ollama(model)
            if attempt == max_retries - 1:
                print("Failed to get response from Ollama after multiple attempts.")
                return None
            time.sleep(retry_delay)

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