from dataclasses import dataclass
from typing import Dict, Any

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

sports_items = ["football", "tennis", "basketball", "cycling", "swimming"]
tech_items = ["AI system", "cloud platform", "mobile app", "data pipeline", "robotics tool"]
finance_items = ["bank loan", "tax policy", "stock market", "budget plan", "investment fund"]
health_items = ["hospital policy", "vaccine program", "mental health service", "medical device", "nutrition study"]
politics_items = ["election reform", "ministerial decision", "government regulation", "parliament debate", "public policy"]

# Now we write single-label classification templates. We choose sentiment classification as the task, with labels {positive, negative, neutral}. We create templates that can be filled with different products, companies, and experiences to generate a variety of examples.
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

# Now we implement the single-label classification templates. We use sentiment classification with labels: {positive, negative, neutral}
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

# Now we do the same for multi-label classification templates. We use topic classification with labels: {finance, health, politics, sports, tech}
multi_label_templates = [
{
"template_name": "institutional_announcement",
"pattern": "The {actor} announced a new {topic_item} related to {secondary_item}.",
"instruction": "Assign all applicable topic labels from {finance, health, politics, sports, tech}. Return them alphabetically."
},
{
"template_name": "company_news",
"pattern": "{company} launched a {topic_item} while discussing its impact on {secondary_item}.",
"instruction": "Assign all applicable topic labels from {finance, health, politics, sports, tech}. Return them alphabetically."
},
{
"template_name": "policy_summary",
"pattern": "A report described how {topic_item} could affect {secondary_item}.",
"instruction": "Assign all applicable topic labels from {finance, health, politics, sports, tech}. Return them alphabetically."
},
{
"template_name": "event_report",
"pattern": "During the meeting, speakers focused on {topic_item} and {secondary_item}.",
"instruction": "Assign all applicable topic labels from {finance, health, politics, sports, tech}. Return them alphabetically."
}
]

# For the multi-label templates, we will fill the {topic_item} and {secondary_item} slots with items from the topic_pool. Each item in the topic_pool is associated with one of the five topics, so we can determine which labels apply based on which items are used in the template.
topic_pool = {
    "politics": politics_items,
    "tech": tech_items,
    "health": health_items,
    "sports": sports_items,
    "finance": finance_items
}

# We create a list of all possible multi-label combinations for the 5 topics, excluding the empty set. This will be used to generate examples with multiple labels.
multi_label_combinations = [
["politics"],
["tech"],
["health"],
["sports"],
["finance"],
["politics", "tech"],
["tech", "health"],
["politics", "finance"],
["sports", "health"],
["finance", "tech"],
["politics", "tech", "health"],
["finance", "politics", "tech"],
]

# Finally, we create templates for information extraction. We choose a schema with three fields: person, date, and location. We create templates that describe events where these three fields are mentioned together, so that the expected output is a JSON object with those three fields.
ie_templates = [
{
"template_name": "meeting_announcement",
"pattern": "{person} will speak in {location} on {date}.",
"instruction": "Extract the fields person, date, and location from the text."
},
{
"template_name": "conference_attendance",
"pattern": "On {date}, {person} attended a conference in {location}.",
"instruction": "Extract the fields person, date, and location from the text."
},
{
"template_name": "appointment_notice",
"pattern": "{person} was appointed in {location} on {date}.",
"instruction": "Extract the fields person, date, and location from the text."
},
{
"template_name": "event_registration",
"pattern": "The registration confirms that {person} is expected in {location} on {date}.",
"instruction": "Extract the fields person, date, and location from the text."
}
]

# Rule-based transformation templates We create input templates that can be deterministically transformed by the rules we defined in RULE_SET. For example, the "lowercase" rule should transform all letters to lowercase, so we create input patterns with a mix of uppercase and lowercase letters to test this.
transformation_input_templates = [
{
"template_name": "mixed_case_statement",
"pattern": "Today {person} Visited {location} With {n} Friends!",
},
{
"template_name": "punctuated_note",
"pattern": "Reminder: bring {n} tickets, {n2} pens, and a map to {location}.",
},
{
"template_name": "short_report",
"pattern": "{person} bought {n} books and {n2} snacks in {location} yesterday.",
},
{
"template_name": "word_length_test",
"pattern": "Tiny words vanish slowly when elephantine expressions dominate paragraphs.",
}
]

# For the rule-based transformations, we will fill the {person}, {location}, {n}, and {n2} slots with values from our reusable pools (people, locations) and some random numbers. This way we can generate a variety of inputs to test the deterministic rules.
rule_instructions = {
"lowercase": "Convert the text to lowercase.",
"remove_punctuation": "Remove all punctuation from the text.",
"replace_numbers_with_NUM": "Replace every number in the text with <NUM>.",
"remove_words_longer_than_6": "Remove all words longer than 6 characters from the text."
}

# Finally, we create templates for extractive QA. Each template consists of a passage with placeholders for person, location, date, company, and product. The question is designed so that the answer is an exact span from the passage corresponding to one of those fields.
qa_templates = [
{
"template_name": "schedule_passage",
"passage": "{person} arrived in {location} on {date} for a training session.",
"question": "Where did {person} arrive?",
"answer_field": "location",
"instruction": "Answer the question using an exact span from the passage."
},
{
"template_name": "announcement_passage",
"passage": "The report says that {company} released the {product} on {date}.",
"question": "What did {company} release?",
"answer_field": "product",
"instruction": "Answer the question using an exact span from the passage."
},
{
"template_name": "travel_note",
"passage": "On {date}, {person} left Rome and reached {location} that evening.",
"question": "Who reached {location} that evening?",
"answer_field": "person",
"instruction": "Answer the question using an exact span from the passage."
},
{
"template_name": "project_summary",
"passage": "{person} said the meeting in {location} would start on {date}.",
"question": "On what date would the meeting start?",
"answer_field": "date",
"instruction": "Answer the question using an exact span from the passage."
}
]