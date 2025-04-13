import re
import random
from collections import Counter

query_params_sentiment = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "max_new_tokens": 64,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
}

query_params_gold = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "max_new_tokens": 128,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
}

def get_query_params(dataset_name: str):
    if "sentiment" in dataset_name:
        return query_params_sentiment
    elif "gold" in dataset_name:
        return query_params_gold
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def find_majority(responses):
    counter = Counter(responses)
    majority = counter.most_common(1)[0]
    if majority[1] > len(responses) / 2:
        return majority[0]
    else:
        return random.choice(responses)

def clean_llm_output(dataset_name, text: str):
    if "sentiment" in dataset_name:
        return clean_llm_output_sentiment(text)

def clean_llm_output_gold(text: str):
    # TODO
    pass

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