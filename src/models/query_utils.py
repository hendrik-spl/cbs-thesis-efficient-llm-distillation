import re
import json
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

def find_majority(responses, dataset_name):
    if "sentiment" in dataset_name: # list of strings
        counter = Counter(responses)
        majority = counter.most_common(1)[0]
        if majority[1] > len(responses) / 2:
            return majority[0]
        else:
            return random.choice(responses)
    elif "gold" in dataset_name: # list of dicts
        return responses[0]  # Assuming we only use one shot / no self-consistency for gold

def clean_llm_output(dataset_name, text: str):
    if "sentiment" in dataset_name:
        return clean_llm_output_sentiment(text)
    elif "gold" in dataset_name:
        return clean_llm_output_gold(text)

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
        
    majority = find_majority(words_found, dataset="sentiment")
    
    return majority

def clean_llm_output_gold(input_data):
    """
    Process various input formats containing financial sentiment data and return a standardized dictionary.
    
    Args:
        input_data: Dictionary, string, or other format containing sentiment analysis results
        
    Returns:
        dict: Dictionary with all expected keys and validated values (0, 1, or -1 for errors)
    """
    # Define the expected keys
    expected_keys = [
        "price_or_not", 
        "price_up", 
        "price_const_stable", 
        "price_down", 
        "past_price_info", 
        "future_price_info", 
        "past_gen_info", 
        "future_gen_info", 
        "asset_comparison"
    ]
    
    # Initialize result dictionary with all keys set to -1
    result = {key: -1 for key in expected_keys}
    
    # Parse input to dictionary if it's not already
    parsed_data = {}
    
    if isinstance(input_data, dict):
        parsed_data = input_data
    elif isinstance(input_data, str):
        # Remove markdown code blocks if present
        clean_input = re.sub(r'```(?:json|python)?\s*|\s*```', '', input_data)
        
        # Try to find and extract JSON-like structure from text
        json_pattern = r'(?:\{|\[).*?(?:\}|\])'
        json_matches = re.findall(json_pattern, clean_input, re.DOTALL)
        
        if json_matches:
            # Try each potential JSON match
            for json_str in json_matches:
                try:
                    candidate = json.loads(json_str)
                    if isinstance(candidate, dict) and any(key in candidate for key in expected_keys):
                        parsed_data = candidate
                        break
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON was found, try direct parsing
        if not parsed_data:
            try:
                parsed_data = json.loads(clean_input)
            except json.JSONDecodeError:
                # Try to parse Python dict syntax
                try:
                    # Replace single quotes with double quotes for JSON parsing
                    clean_input = clean_input.replace("'", '"')
                    parsed_data = json.loads(clean_input)
                except json.JSONDecodeError:
                    # Try to extract key-value pairs using regex
                    pairs = re.findall(r'"?(\w+)"?\s*:\s*(-?\d+)', clean_input)
                    parsed_data = {key: int(value) for key, value in pairs}
    
    # Update result with valid values from parsed data
    for key in expected_keys:
        if key in parsed_data:
            # Ensure value is 0 or 1, otherwise set to -1
            if parsed_data[key] in [0, 1]:
                result[key] = parsed_data[key]
    
    return result