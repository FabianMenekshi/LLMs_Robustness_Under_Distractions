# from dataclasses import dataclass
# from typing import Dict, Any

# @dataclass
# class CandidateExample:
#     ...

# SINGLE_LABEL_SET = [...]
# MULTI_LABEL_SET = [...]
# IE_SCHEMA_KEYS = [...]
# RULE_SET = [...]

# people = [...]
# locations = [...]
# dates = [...]
# products = [...]
# companies = [...]

# single_label_templates = [...]
# single_label_value_map = {...}

# multi_label_templates = [...]
# topic_pool = {...}
# multi_label_combinations = [...]

# ie_templates = [...]

# transformation_input_templates = [...]
# rule_instructions = {...}

# qa_templates = [...]

##########################################################33333

from dataclasses import dataclass, field, asdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional
import json
import random
import re
from itertools import product

@dataclass
class CandidateExample:
    example_id: str
    task_name: str
    template_name: str
    instruction: str
    input_data: Dict[str, Any]
    gold_output: Dict[str, Any]
    metadata: Dict[str, Any]
    review_status: str = "pending"
    review_note: str = ""

# We now choose concrete template families for each task

# Chosen task variants
# 1. Single-label classification (positive, negative, neutral)
# 2. Multi-label classification (politics, tech, health, sports, finance)
# 3. Information extraction {"person": "...", "date": "...", "location": "..."}
# 4. Rule-based transformation (we choose deterministic rules)
# 5. Extractive QA (We create short passages and questions where the answer is an exact, unique span in the passage)

SINGLE_LABEL_SET = ["positive", "negative", "neutral"]
MULTI_LABEL_SET = ["finance", "health", "politics", "sports", "tech"]
IE_SCHEMA_KEYS = ["person", "date", "location"]

RULE_SET = [
"lowercase",
"remove_punctuation",
"replace_numbers_with_NUM",
"remove_words_longer_than_6"
]

# Now we create reusable value pools

people = [
"Alice Smith", "Bruno Rossi", "Carla Gomez", "David Lee", "Elena Marino",
"Farah Khan", "Giulia Conti", "Hassan Ali", "Irene Novak", "Jonas Weber"
]

locations = [
"Rome", "Milan", "Paris", "Berlin", "Madrid",
"Lisbon", "Vienna", "Prague", "Athens", "Dublin"
]

dates = [
"2024-01-15", "2024-03-21", "2024-06-10", "2024-09-05", "2025-02-14",
"2025-04-30", "2025-07-12", "2025-10-03", "2026-01-08", "2026-02-19"
]

products = [
"medical software", "robot vacuum", "budget phone", "fitness watch", "solar battery"
]

companies = [
"Northwind Labs", "BlueRiver Health", "Atlas Finance", "Civic Data Group", "Urban Sports Media"
]

sports = ["football", "tennis", "basketball", "cycling", "swimming"]
tech_items = ["AI system", "cloud platform", "mobile app", "data pipeline", "robotics tool"]
finance_items = ["bank loan", "tax policy", "stock market", "budget plan", "investment fund"]
health_items = ["hospital policy", "vaccine program", "mental health service", "medical device", "nutrition study"]
politics_items = ["election reform", "ministerial decision", "government regulation", "parliament debate", "public policy"]

# Now we write single-label classification templates

single_label_templates = [
{
"template_name": "product_review",
"pattern": "The {product} was {adjective} and worked {ending}.",
"instruction": "Classify the sentiment of the text using exactly one label from {positive, negative, neutral}.",
},
{
"template_name": "service_feedback",
"pattern": "The staff at the store were {adjective} and the service was {ending}.",
"instruction": "Classify the sentiment of the text using exactly one label from {positive, negative, neutral}.",
},
{
"template_name": "event_reaction",
"pattern": "The event was {adjective}; overall it felt {ending}.",
"instruction": "Classify the sentiment of the text using exactly one label from {positive, negative, neutral}.",
},
{
"template_name": "delivery_comment",
"pattern": "The package arrived {adjective} and the whole experience was {ending}.",
"instruction": "Classify the sentiment of the text using exactly one label from {positive, negative, neutral}.",
},
]

single_label_value_map = {
"positive": [
{"adjective": "excellent", "ending": "smooth and satisfying"},
{"adjective": "great", "ending": "pleasant from start to finish"},
{"adjective": "reliable", "ending": "better than expected"},
],
"negative": [
{"adjective": "terrible", "ending": "frustrating and disappointing"},
{"adjective": "awful", "ending": "worse than expected"},
{"adjective": "poor", "ending": "annoying from start to finish"},
],
"neutral": [
{"adjective": "ordinary", "ending": "acceptable but unremarkable"},
{"adjective": "fine", "ending": "neither good nor bad"},
{"adjective": "average", "ending": "mostly standard"},
],
}