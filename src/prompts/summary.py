def get_summmary_prompt(text):
    return f"""\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to generate a bullet-point summary of the TEXT provided below. Extract and present the most relevant and significant insights in the TEXT as 3–5 concise bullet points or short sentences. Follow these guidelines:

Respond only with bullet points or short sentences.

Each bullet point should capture one key idea in a brief, clear manner.

Base your summary solely on the TEXT provided and do not add any external context or information.

Your TEXT to analyze:
TEXT: {text}

BULLET-POINT SUMMARY: """