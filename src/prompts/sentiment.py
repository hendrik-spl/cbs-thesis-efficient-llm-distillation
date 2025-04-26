response_template = "Final Label:"

def get_sentiment_prompt(text):
    return f"""\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to analyze the sentiment in the TEXT below from an investor perspective and label it with only one the three labels:
positive, negative, or neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company. 

Do not provide any explanations and only respond with one of the labels as one word: negative, positive, or neutral

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
Label: positive
Text: The company generated net sales of 11.3 million euro this year.
Label: neutral
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
Label: negative

Your TEXT to analyse:
TEXT: {text}
Final Label: """