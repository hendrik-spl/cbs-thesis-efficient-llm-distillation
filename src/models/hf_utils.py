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