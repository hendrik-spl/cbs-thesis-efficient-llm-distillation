import re
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.models.ollama_utils import check_if_ollama_model_exists, use_ollama, query_ollama_model
# from src.models.hf_utils import HF_Manager

query_params_sentiment = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "max_new_tokens": 50,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
}

# def query_with_sc(model, prompt, shots, use_ollama):
#     query_func = query_ollama_model if use_ollama else HF_Manager.query_model

#     query_params = {
#         "temperature": 0.3,
#         "seed": None, # Set to None for self-consistency
#         "do_sample": True, # Enable sampling for diversity
#         "top_p": 0.8,
#         "top_k": 40,
#         "max_new_tokens": 50,
#         "custom_max_retries": 3,
#         "custom_retry_delay": 5,
#     }

#     responses = []
#     for i in range(shots):
#         response = query_func(model_config=model, prompt=prompt, params=query_params)
#         responses.append(clean_llm_output_sentiment(response))

#     majority_vote = find_majority(responses)
#     return majority_vote

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

# def get_model_config(model_name: str):
#     if use_ollama(model_name):
#         check_if_ollama_model_exists(model_name)
#         return model_name, True
#     else:
#         # Check if this is a PEFT adapter model path
#         if os.path.exists(os.path.join(model_name, "adapter_config.json")):
#             print(f"Loading as PEFT adapter model: {model_name}")
#             model, tokenizer = load_finetuned_adapter(model_name)
#             return (model, tokenizer), False
#         else:
#             # Regular HF model loading
#             print(f"Loading as standard model: {model_name}")
#             hf_model = AutoModelForCausalLM.from_pretrained(model_name)
#             hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
#             return (hf_model, hf_tokenizer), False
        
def load_finetuned_adapter(model_path):
    """
    Load a fine-tuned PEFT/LoRA adapter model from a local path
    """
    import json
    import os
    from peft import PeftModel
    import torch
    
    # Step 1: Load the tokenizer from the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Step 2: Get the base model name from adapter_config.json
    with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path")
    
    print(f"Loading base model: {base_model_name}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Step 3: Load base model with the EXACT config from the adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Step 4: CRITICAL - Resize token embeddings BEFORE loading adapter
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Step 5: Set pad token ID if it exists
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set pad_token_id to {tokenizer.pad_token_id}")
    
    # Step 6: Now load the adapter
    print(f"Loading adapter from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer