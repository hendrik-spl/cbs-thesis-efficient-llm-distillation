import os
import re
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.ollama_utils import check_if_ollama_model_exists, use_ollama, query_ollama_model
from src.models.hf_utils import query_hf_model

def query_with_sc(model, prompt, shots, use_ollama):
    query_func = query_ollama_model if use_ollama else query_hf_model

    query_params = {
        "temperature": 0.3,
        "seed": None, # Set to None for self-consistency
        "do_sample": True, # Enable sampling for diversity
        "top_p": 0.8,
        "top_k": 40,
        "max_new_tokens": 50,
        "custom_max_retries": 3,
        "custom_retry_delay": 5,
    }

    hf_model = AutoModelForCausalLM.from_pretrained("models/sentiment:50agree/llama3.2:1b/checkpoints/honest-serenity-73")
    hf_tokenizer = AutoTokenizer.from_pretrained("models/sentiment:50agree/llama3.2:1b/checkpoints/honest-serenity-73")
    hf_model_config = (hf_model, hf_tokenizer)
    
    responses = []
    for i in range(2):
        ollama_response = query_ollama_model("hf.co/hendrik-spl/test_honest-serenity-73-Q4_K_M-GGUF", prompt, query_params)
        hf_response = query_hf_model(hf_model_config, prompt, query_params)
        print(f"Response {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Ollama Reponse: {ollama_response}")
        print(f"HF Reponse: {hf_response}")

    return "positive"

    # responses = []
    # for i in range(shots):
        # response = query_func(model_config=model, prompt=prompt, params=query_params)
        # print(f"Response {i+1}:")
        # print(f"Reponse: {response}")
        # print(f"Cleaned response: {clean_llm_output_sentiment(response)}")
        # print(f"------------")
        # responses.append(clean_llm_output_sentiment(response))

    # majority_vote = find_majority(responses)
    # return majority_vote

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

def get_model_config(model_name: str):
    if use_ollama(model_name):
        check_if_ollama_model_exists(model_name)
        return model_name, True
    else:
        # Check if this is a path to a fine-tuned model
        if os.path.exists(model_name) and os.path.isdir(model_name):
            # First, load the tokenizer to get its vocabulary size
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Get the base model name (usually stored in config)
            import json
            with open(os.path.join(model_name, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            
            # Load the base model first
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Ensure tokenizer and model vocab sizes match
            if tokenizer.pad_token is None:
                special_tokens_dict = {'pad_token': '[PAD]'}
                tokenizer.add_special_tokens(special_tokens_dict)
                base_model.resize_token_embeddings(len(tokenizer))
                base_model.config.pad_token_id = tokenizer.pad_token_id
                
            # Then load the adapter/LoRA weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, model_name)
            
            return (model, tokenizer), False
        else:
            # Regular HF model loading
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            return (hf_model, hf_tokenizer), False