def get_sentiment_prompt(text):
    return f"""
            Please classify the sentiment of the following financial text:
            {text}
            Sentiment Options: negative, neutral, positive.
            Respond with the integer corresponding to the sentiment.
            0: negative
            1: neutral
            2: positive
            Do not include any other information in your response.
            """