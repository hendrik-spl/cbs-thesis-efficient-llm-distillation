query_params_sentiment = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
    "max_new_tokens": 8,
    "max_context_length": 384
}

query_params_gold = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
    "max_new_tokens": 96,
    "max_context_length": 768
}

query_params_summary = {
    "temperature": 0.3,
    "seed": None, # Set to None for self-consistency
    "do_sample": True, # Enable sampling for diversity
    "top_p": 0.8,
    "top_k": 40,
    "custom_max_retries": 3,
    "custom_retry_delay": 5,
    "max_new_tokens": 256,
    "max_context_length": 6144
}