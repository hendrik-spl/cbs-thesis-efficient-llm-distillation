import re

def clean_llm_output_to_int(text, extract_sentiment=False):
    """
    Cleans the output of a language model to an integer and optionally extracts sentiment.

    Args:
        text (str): The text to clean.
        extract_sentiment (bool): Whether to extract sentiment information.

    Returns:
        int: If extract_sentiment is False, returns the extracted integer or -1 if none found.
             If extract_sentiment is True:
                - 2 for positive sentiment
                - 1 for neutral sentiment
                - 0 for negative sentiment
                - -1 if no sentiment or multiple sentiments found
    """
    if text is None:
        print("Received None text. Defaulting to -1.")
        return -1

    # Extract the integer from the text
    match = re.search(r"\d+", text)

    # If an integer was found, store it
    if match:
        int_value = int(match.group())
    else:
        int_value = -1
    
    # If not extracting sentiment, just return the int
    if not extract_sentiment:
        return int_value
        
    # Check for sentiment words - only extract if exactly one is found
    sentiment_words = ["negative", "neutral", "positive"]
    found_sentiments = []
    
    for sentiment in sentiment_words:
        if re.search(r'\b' + sentiment + r'\b', text.lower()):
            found_sentiments.append(sentiment)
    
    # If exactly one sentiment word is found, return corresponding integer
    if len(found_sentiments) == 1:
        sentiment = found_sentiments[0]
        if sentiment == "positive":
            return 2
        elif sentiment == "neutral":
            return 1
        elif sentiment == "negative":
            return 0
    
    # If no sentiment or multiple sentiments found
    return -1