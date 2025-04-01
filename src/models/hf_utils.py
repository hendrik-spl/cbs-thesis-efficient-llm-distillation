import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.model_utils import clean_llm_output_sentiment, find_majority

def query_hf_model_sc(model_path, prompt, shots=5):
    """
    Implement self-consistency on top of Hugging Face model.
    Args:
        model_path (str): The path to the model.
        prompt (str): The prompt to send to the model.
        shots (int): The number of shots to use for self-consistency.
    """
    responses = []
    for i in range(shots):
        response = query_hf_model(model_path, prompt)
        responses.append(clean_llm_output_sentiment(response))

    majority_vote = find_majority(responses)
    return majority_vote

def query_hf_model(model_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Store the input length to know where the generated text starts
    input_length = inputs.input_ids.shape[1]

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 pad_token_id=tokenizer.eos_token_id,
                                 max_new_tokens=50
                                 )

    # Decode ONLY the generated tokens (exclude the input prompt tokens)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response

def load_model_from_hf(model_name: str, peft: bool):
    """
    Loads a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (PreTrainedModel): The model loaded from the model hub.
        tokenizer (PreTrainedTokenizer): The tokenizer loaded from the model hub.
        peft_config (Optional[LoraConfig]): The PEFT configuration for the model, if it exists.
    """
    model_hf_mapping = {
        "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.1:405b": "meta-llama/Llama-3.1-405B-Instruct",
        "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
        "smollm2:135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    }
    if model_name in model_hf_mapping:
        model_name = model_hf_mapping[model_name]
    else:
        raise ValueError(f"Model name {model_name} not recognized. Predefined models are: {model_hf_mapping.keys()}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None: # Set the pad token to the eos token if it doesn't exist
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            lora_dropout=0,
            modules_to_save=["lm_head", "embed_token"],
        )
    #     Example from https://www.kaggle.com/code/gpreda/fine-tune-llama3-with-qlora-for-sentiment-analysis
    #     peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0,
    #     r=64,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",],
    # )
    
        print(f"Loaded model {model_name} with PEFT configuration.")
        return model, tokenizer, peft_config
    
    return model, tokenizer, None