response_template = "FINAL CLASSIFICATION:"

def get_gold_classification_prompt(text):
    return f"""\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to analyze gold commodity news headlines based exclusively on the TEXT provided, classifying them into nine binary dimensions. Your classification must strictly follow the categories defined below. Respond ONLY in JSON format as illustrated. Do NOT provide explanations or additional information.

Categories:
price_or_not: 1 if headline refers to gold prices, 0 otherwise
price_up: 1 if headline indicates prices going up, 0 otherwise
price_const_stable: 1 if headline indicates prices remaining constant/stable, 0 otherwise
price_down: 1 if headline indicates prices going down, 0 otherwise
past_price_info: 1 if headline refers to past gold price information, 0 otherwise
future_price_info: 1 if headline refers to future gold price predictions or expectations, 0 otherwise
past_gen_info: 1 if headline provides past general information unrelated to prices (e.g., production, imports, exports), 0 otherwise
future_gen_info: 1 if headline provides future general information unrelated to prices (e.g., policy announcements, future events), 0 otherwise
asset_comparison: 1 if headline compares gold price with another asset, 0 otherwise

Respond ONLY with the following JSON format:
{{
"price_or_not": 0 or 1,
"price_up": 0 or 1,
"price_const_stable": 0 or 1,
"price_down": 0 or 1,
"past_price_info": 0 or 1,
"future_price_info": 0 or 1,
"past_gen_info": 0 or 1,
"future_gen_info": 0 or 1,
"asset_comparison": 0 or 1
}}

Your response should be a JSON object with the keys and values as specified above. Do not include any other text or explanations.
Your response should not contain any backtickes such as ` or ``, and should not be formatted in any code block.

Examples:

TEXT: "Gold futures fall as traders await key Fed announcement."
CLASSIFICATION:
{{
"price_or_not": 1,
"price_up": 0,
"price_const_stable": 0,
"price_down": 1,
"past_price_info": 0,
"future_price_info": 1,
"past_gen_info": 0,
"future_gen_info": 0,
"asset_comparison": 0
}}

TEXT: "India's gold imports rise by 12% in March compared to February."
CLASSIFICATION:
{{
"price_or_not": 0,
"price_up": 0,
"price_const_stable": 0,
"price_down": 0,
"past_price_info": 0,
"future_price_info": 0,
"past_gen_info": 1,
"future_gen_info": 0,
"asset_comparison": 0
}}

TEXT: "Gold likely to remain steady ahead of the U.S. jobs report."
CLASSIFICATION:
{{
"price_or_not": 1,
"price_up": 0,
"price_const_stable": 1,
"price_down": 0,
"past_price_info": 0,
"future_price_info": 1,
"past_gen_info": 0,
"future_gen_info": 0,
"asset_comparison": 0
}}

Your TEXT to analyze:
TEXT: {text}
FINAL CLASSIFICATION: """