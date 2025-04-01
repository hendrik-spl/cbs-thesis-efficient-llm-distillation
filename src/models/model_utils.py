import re
import random
from collections import Counter

from src.models.ollama_utils import query_ollama_model
from src.models.hf_utils import query_hf_model

def query_with_sc(model, prompt, shots, use_ollama):
    query_func = query_ollama_model if use_ollama else query_hf_model

    query_params = {
        "temperature": 0.3,
        "seed": None, # Set to None for self-consistency
        "do_sample": True, # Enable sampling for diversity
        "top_p": 0.8,
        "top_k": 40,
        "max_new_tokens": 3,
        "custom_max_retries": 3,
        "custom_retry_delay": 5,
    }
    
    responses = []
    for i in range(shots):
        response = query_func(model_config=model, prompt=prompt, params=query_params)
        print(f"Response {i+1}:")
        print(f"Reponse: {response}")
        print(f"Cleaned response: {clean_llm_output_sentiment(response)}")
        print(f"------------")
        responses.append(clean_llm_output_sentiment(response))

    majority_vote = find_majority(responses)
    return majority_vote

def find_majority(responses):
    counter = Counter(responses)
    majority = counter.most_common(1)[0]
    if majority[1] > len(responses) / 2:
        return majority[0]
    else:
        return random.choice(responses)

def clean_llm_output_sentiment(text: str):
    """
    Cleans the output of a language model and extracts sentiment.

    Args:
        text (str): The text to clean.

    Returns:
        str: The sentiment label ("negative", "neutral", "positive")
    """
    sentiment_options = ["negative", "neutral", "positive"]

    if text is None or text == "":
        print("Received None text. Defaulting to neutral.")
        return "neutral"
    
    text = text.strip().lower()
    
    # look for exact matches
    if text in sentiment_options:
        return text
    
    words_found = []
    
    for word in sentiment_options:
        words_found.extend([word] * len(re.findall(word, text)))
    
    if not words_found:
        return "neutral"
        
    majority = find_majority(words_found)
    
    return majority