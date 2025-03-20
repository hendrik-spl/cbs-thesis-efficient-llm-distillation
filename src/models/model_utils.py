import re
import random
from collections import Counter

def find_majority(responses):
    counter = Counter(responses)
    majority = counter.most_common(1)[0]
    if majority[1] > len(responses) / 2:
        return majority[0]
    else:
        return random.choice(responses)

def clean_llm_output_sentiment(text: str):
    """
    Cleans the output of a language model to an integer and extracts sentiment.

    Args:
        text (str): The text to clean.

    Returns:
        int: The cleaned integer value
    """
    sentiment_options = ["negative", "neutral", "positive"] # integer labels: 0, 1, 2

    if text is None or text == "":
        print("Received None text. Defaulting to -1.")
        return -1
    
    text = text.strip().lower()
    
    # look for exact matches
    if text in sentiment_options:
        return sentiment_options.index(text)
    
    words_found = []
    
    for word in sentiment_options:
        words_found.extend([word] * len(re.findall(word, text)))

    majority = find_majority(words_found)
    
    return sentiment_options.index(majority)