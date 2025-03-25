from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_from_hf(model_name: str, peft: bool):
    """
    Loads a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    if "llama3.1:8b" in model_name:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif "llama3.1:405b" in model_name:
        model_name = "meta-llama/Llama-3.1-405B-Instruct"
    elif "llama3.2:1b" in model_name:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif "llama3.2:3b" in model_name:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif "llama3.3:70b" in model_name:
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
    else:
        print(f"Model name {model_name} not recognized.")
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
        print(f"Loaded model {model_name} with PEFT configuration.")
        return model, tokenizer, peft_config
    
    return model, tokenizer