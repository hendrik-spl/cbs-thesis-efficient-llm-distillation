def get_summmary_prompt(text, version="reversed"):
    if version == "short":
        return short_summary_prompt(text)
    elif version == "long":
        return long_summary_prompt(text)
    elif version == "reversed":
        return reversed_summary_prompt(text)

def short_summary_prompt(text):
    return f"""\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to generate a bullet-point summary of the TEXT provided below. Extract and present the most relevant and significant insights in the TEXT as 3–5 concise bullet points or short sentences. Follow these guidelines:

Respond only with bullet points or short sentences.

Each bullet point should capture one key idea in a brief, clear manner.

Base your summary solely on the TEXT provided and do not add any external context or information.

Your TEXT to analyze:
TEXT: {text}

FINAL SUMMARY OF YOUR TEXT: """

def long_summary_prompt(text):
    return f"""\
    You are a specialist in extracting the most critical facts from an earnings‑call transcript.  

Your job is to produce **5 very short bullet points** that capture the most important fact per bullet. Be as concise as possible, and do **not** add any information that isn’t in the text.

Examples:
TEXT: “Company X reported Q1 adjusted EPS of $2.18, up from $1.98 a year ago. Revenue rose 9.7% to $52.5 B, and full‑year revenue guidance was raised from mid‑single digits to high‑single digits growth. All other previously issued guidance remains unchanged.”
SUMMARY: Q1 adjusted EPS $2.18 vs. $1.98 a year ago. Q1 revenue $52.5 B, +9.7% YoY. Raised FY revenue growth guide to high‑single digits. Other FY guidance unchanged.

TEXT: “CompName reported Q3 adjusted EPS of $0.66 and revenue of $1.02 B, up 9% year‑over‑year. It now expects full‑year revenue between $3.96 B and $4.00 B, and 2021 adjusted EPS of $2.40–$2.43.”
SUMMARY: Q3 adjusted EPS $0.66. Q3 revenue $1.02 B, +9% YoY. FY revenue guide $3.96–4.00 B. 2021 EPS guide $2.40–2.43.

TEXT: “During Q1, sales climbed 10% to $1.151 B, and EPS hit $0.64. The company raised its 2021 sales guide to $4.8–5.0 B and EPS guide to $2.55–2.75, and increased its Q2 dividend by $0.02 to $0.42 per share.”
SUMMARY: Q1 sales $1.151 B, +10% YoY. Q1 EPS $0.64. Raised 2021 sales guide to $4.8–5.0 B. Raised 2021 EPS guide to $2.55–2.75. Q2 dividend up $0.02 to $0.42.

Now your task:

TEXT: {text}

FINAL SUMMARY OF YOUR TEXT: """

def reversed_summary_prompt(text):
    return f"""\
You are a specialist in extracting the most critical facts from an earnings‑call transcript.

Here is the text from the earnings call transcript that you need to summarize:
TEXT: {text}

Your job is to produce **5 very short bullet points** that capture the most important fact per bullet. Be as concise as possible, and do **not** add any information that isn’t in the text.

Examples:
TEXT: “Company X reported Q1 adjusted EPS of $2.18, up from $1.98 a year ago. Revenue rose 9.7% to $52.5 B, and full‑year revenue guidance was raised from mid‑single digits to high‑single digits growth. All other previously issued guidance remains unchanged.”
SUMMARY: Q1 adjusted EPS $2.18 vs. $1.98 a year ago. Q1 revenue $52.5 B, +9.7% YoY. Raised FY revenue growth guide to high‑single digits. Other FY guidance unchanged.

TEXT: “CompName reported Q3 adjusted EPS of $0.66 and revenue of $1.02 B, up 9% year‑over‑year. It now expects full‑year revenue between $3.96 B and $4.00 B, and 2021 adjusted EPS of $2.40–$2.43.”
SUMMARY: Q3 adjusted EPS $0.66. Q3 revenue $1.02 B, +9% YoY. FY revenue guide $3.96–4.00 B. 2021 EPS guide $2.40–2.43.

TEXT: “During Q1, sales climbed 10% to $1.151 B, and EPS hit $0.64. The company raised its 2021 sales guide to $4.8–5.0 B and EPS guide to $2.55–2.75, and increased its Q2 dividend by $0.02 to $0.42 per share.”
SUMMARY: Q1 sales $1.151 B, +10% YoY. Q1 EPS $0.64. Raised 2021 sales guide to $4.8–5.0 B. Raised 2021 EPS guide to $2.55–2.75. Q2 dividend up $0.02 to $0.42.

FINAL SUMMARY OF YOUR TEXT: """