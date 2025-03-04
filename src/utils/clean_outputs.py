import re

def clean_llm_output_to_int(text):
    """
    Cleans the output of a language model to an integer.

    Args:
        text (str): The text to clean.

    Returns:
        int: The integer extracted from the text.
    """
    # Extract the integer from the text
    match = re.search(r"\d+", text)

    # If an integer was found, return it
    if match:
        return int(match.group())
    else:
        print(f"Could not extract an integer from the text: {text}")
        return None