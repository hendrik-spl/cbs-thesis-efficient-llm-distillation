from transformers import AutoTokenizer, AutoModel

def load_model_from_hf(model_name, num_labels):
    """
    Loads a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.
        num_labels (int): The number of labels for the model.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    # if ":" in model_name: # Handle model names with a slash in them for HF
    #     model_name = model_name.replace(":", "/")
    if "llama3.2:1b" in model_name:
        model_name = "meta-llama/Llama-3.2-1B"
    else:
        print(f"Model name {model_name} not recognized.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer