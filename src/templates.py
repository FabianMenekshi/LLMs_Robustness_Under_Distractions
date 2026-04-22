'''
This file is the foundational content library of the benchmark. It defines:
    - the task types
    - the allowed labels / schemas / rules
    - reusable value pools like people, dates, locations, companies
    - paraphrased instruction pools
    - the actual text templates for each task

So this file does not generate examples by itself. It is more like a structured inventory of ingredients.
'''

from dataclasses import dataclass
from typing import Dict, Any, List


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


# -------------------------------------------------------------------
# Task sets and schema definitions
# -------------------------------------------------------------------

SINGLE_LABEL_SET = ["positive", "negative", "neutral"]
MULTI_LABEL_SET = ["finance", "health", "politics", "sports", "tech"]
IE_SCHEMA_KEYS = ["person", "date", "location"]

RULE_SET = [
    "lowercase",
    "remove_punctuation",
    "replace_numbers_with_NUM",
    "remove_words_longer_than_6",
]


# -------------------------------------------------------------------
# Reusable value pools
# -------------------------------------------------------------------

people = [
    "Alice Smith",
    "Bruno Rossi",
    "Carla Gomez",
    "David Lee",
    "Elena Marino",
    "Farah Khan",
    "Giulia Conti",
    "Hassan Ali",
    "Irene Novak",
    "Jonas Weber",
]

locations = [
    "Rome",
    "Milan",
    "Paris",
    "Berlin",
    "Madrid",
    "Lisbon",
    "Vienna",
    "Prague",
    "Athens",
    "Dublin",
]

dates = [
    "2024-01-15",
    "2024-03-21",
    "2024-06-10",
    "2024-09-05",
    "2025-02-14",
    "2025-04-30",
    "2025-07-12",
    "2025-10-03",
    "2026-01-08",
    "2026-02-19",
]

products = [
    "medical software",
    "robot vacuum",
    "budget phone",
    "fitness watch",
    "solar battery",
    "analytics dashboard",
    "smart speaker",
    "security platform",
]

companies = [
    "Northwind Labs",
    "BlueRiver Health",
    "Atlas Finance",
    "Civic Data Group",
    "Urban Sports Media",
    "Harbor Analytics",
    "Summit Retail Systems",
    "Meridian HealthWorks",
]

sports_items = [
    "football tournament",
    "tennis training app",
    "basketball analytics tool",
    "cycling event",
    "swimming program",
]

tech_items = [
    "AI system",
    "cloud platform",
    "mobile app",
    "data pipeline",
    "robotics tool",
    "software update",
    "machine learning model",
]

finance_items = [
    "bank loan",
    "tax policy",
    "stock market update",
    "budget plan",
    "investment fund",
    "procurement budget",
]

health_items = [
    "hospital policy",
    "vaccine program",
    "mental health service",
    "medical device",
    "nutrition study",
    "clinical software",
]

politics_items = [
    "election reform",
    "ministerial decision",
    "government regulation",
    "parliament debate",
    "public policy",
    "municipal vote",
]


# -------------------------------------------------------------------
# Instruction paraphrase pools
# -------------------------------------------------------------------

single_label_instructions = [
    "Classify the sentiment of the text using exactly one label from {positive, negative, neutral}.",
    "Return exactly one sentiment label from {positive, negative, neutral}.",
    "Choose one sentiment label only from {positive, negative, neutral}.",
    "Assign a single sentiment label from the allowed set {positive, negative, neutral}.",
    "Label the text with exactly one of {positive, negative, neutral}.",
    "Return one sentiment label only: {positive, negative, neutral}.",
]

multi_label_instructions = [
    "Assign all applicable topic labels from {finance, health, politics, sports, tech}. Return them alphabetically.",
    "Return every relevant topic label from {finance, health, politics, sports, tech}, sorted alphabetically.",
    "Choose all labels that apply from {finance, health, politics, sports, tech}. Return them in alphabetical order.",
    "Assign the applicable topic labels using only {finance, health, politics, sports, tech}. Sort the labels alphabetically.",
    "Return all matching topic labels from the allowed set {finance, health, politics, sports, tech}, in alphabetical order.",
    "Label the text with every applicable topic from {finance, health, politics, sports, tech}. Output the labels alphabetically.",
]

ie_instructions = [
    "Extract the fields person, date, and location from the text.",
    "Return the person, date, and location fields from the text.",
    "Identify and output person, date, and location from the text.",
    "Extract person, date, and location and return them as structured fields.",
    "Find the person, date, and location mentioned in the text.",
    "Return the values for person, date, and location from the text.",
]

rule_instructions = {
    "lowercase": [
        "Convert the text to lowercase.",
        "Rewrite the text in lowercase.",
        "Lowercase all letters in the text.",
        "Return the text with all alphabetic characters in lowercase.",
    ],
    "remove_punctuation": [
        "Remove all punctuation from the text.",
        "Return the text with punctuation removed.",
        "Delete every punctuation mark from the text.",
        "Strip punctuation characters from the text.",
    ],
    "replace_numbers_with_NUM": [
        "Replace every number in the text with <NUM>.",
        "Substitute each number in the text with <NUM>.",
        "Return the text with all numbers replaced by <NUM>.",
        "Replace every numeric sequence with <NUM>.",
    ],
    "remove_words_longer_than_6": [
        "Remove all words longer than 6 characters from the text.",
        "Delete every word whose length is greater than 6 characters.",
        "Return the text after removing words longer than 6 characters.",
        "Drop all words with more than 6 characters.",
    ],
}

qa_instructions = [
    "Answer the question using an exact span from the passage.",
    "Return the exact words from the passage that answer the question.",
    "Use a verbatim span from the passage.",
    "Copy the answer directly from the passage.",
    "Answer with the precise text appearing in the passage.",
    "Return only the exact answer span from the passage.",
    "Use the passage wording exactly for the answer.",
    "Extract the answer directly from the passage without paraphrasing.",
]


# -------------------------------------------------------------------
# Single-label classification templates
# These are slightly more natural and less keyword-obvious than before.
# -------------------------------------------------------------------

single_label_templates = [
    {
        "template_name": "product_review_balanced",
        "pattern": "I expected more from the {product}. It was {descriptor_1}, but the overall experience felt {descriptor_2}.",
        "instruction_pool": single_label_instructions,
    },
    {
        "template_name": "service_feedback_natural",
        "pattern": "The staff were {descriptor_1} and the process was {descriptor_2}. I would describe the experience as {descriptor_3}.",
        "instruction_pool": single_label_instructions,
    },
    {
        "template_name": "delivery_comment_mixed",
        "pattern": "The package arrived {descriptor_1}. Nothing was seriously wrong, but the whole experience felt {descriptor_2}.",
        "instruction_pool": single_label_instructions,
    },
    {
        "template_name": "event_reaction_short_review",
        "pattern": "The event was {descriptor_1}. By the end, it felt {descriptor_2} rather than memorable.",
        "instruction_pool": single_label_instructions,
    },
    {
        "template_name": "app_experience_note",
        "pattern": "The app looked {descriptor_1}, and most features worked, but using it felt {descriptor_2}.",
        "instruction_pool": single_label_instructions,
    },
    {
        "template_name": "store_visit_note",
        "pattern": "My visit to the store was {descriptor_1}. The staff were {descriptor_2}, and the result felt {descriptor_3}.",
        "instruction_pool": single_label_instructions,
    },
]

single_label_value_map = {
    "positive": [
        {
            "descriptor_1": "smooth",
            "descriptor_2": "pleasant",
            "descriptor_3": "worth repeating",
        },
        {
            "descriptor_1": "reliable",
            "descriptor_2": "easy",
            "descriptor_3": "better than expected",
        },
        {
            "descriptor_1": "well organized",
            "descriptor_2": "helpful",
            "descriptor_3": "genuinely satisfying",
        },
        {
            "descriptor_1": "impressive",
            "descriptor_2": "straightforward",
            "descriptor_3": "very positive",
        },
    ],
    "negative": [
        {
            "descriptor_1": "confusing",
            "descriptor_2": "frustrating",
            "descriptor_3": "not worth repeating",
        },
        {
            "descriptor_1": "disappointing",
            "descriptor_2": "slow",
            "descriptor_3": "worse than expected",
        },
        {
            "descriptor_1": "poorly handled",
            "descriptor_2": "unhelpful",
            "descriptor_3": "quite negative",
        },
        {
            "descriptor_1": "rough",
            "descriptor_2": "annoying",
            "descriptor_3": "hard to recommend",
        },
    ],
    "neutral": [
        {
            "descriptor_1": "ordinary",
            "descriptor_2": "fine",
            "descriptor_3": "acceptable overall",
        },
        {
            "descriptor_1": "standard",
            "descriptor_2": "routine",
            "descriptor_3": "neither especially good nor bad",
        },
        {
            "descriptor_1": "average",
            "descriptor_2": "mostly typical",
            "descriptor_3": "fairly neutral",
        },
        {
            "descriptor_1": "unremarkable",
            "descriptor_2": "manageable",
            "descriptor_3": "about what I expected",
        },
    ],
}


# -------------------------------------------------------------------
# Multi-label classification templates
# More realistic topic mixing and phrasing.
# -------------------------------------------------------------------

multi_label_templates = [
    {
        "template_name": "institutional_announcement",
        "pattern": (
            "The {actor} announced a new initiative involving {topic_item}. "
            "The statement also discussed implications for {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "company_news",
        "pattern": (
            "{company} introduced a new effort around {topic_item} while explaining its connection to {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "policy_summary",
        "pattern": (
            "A policy summary described how {topic_item} could affect {secondary_item} over the next year."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "event_report",
        "pattern": (
            "During the event recap, speakers focused on {topic_item} together with concerns about {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "press_briefing",
        "pattern": (
            "In a press briefing, the {actor} discussed {topic_item} and answered questions about {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "project_update",
        "pattern": (
            "{company} published a project update covering {topic_item} and its relation to {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "newsletter_excerpt",
        "pattern": (
            "The newsletter excerpt linked {topic_item} with recent developments involving {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
    {
        "template_name": "panel_discussion_note",
        "pattern": (
            "Panel notes highlighted debate around {topic_item}, with repeated references to {secondary_item}."
        ),
        "instruction_pool": multi_label_instructions,
    },
]

topic_pool = {
    "politics": politics_items,
    "tech": tech_items,
    "health": health_items,
    "sports": sports_items,
    "finance": finance_items,
}

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
    ["sports", "tech"],
    ["health", "finance"],
]


# -------------------------------------------------------------------
# Information extraction templates
# Slightly richer passages with distractors but still unique target fields.
# -------------------------------------------------------------------

ie_templates = [
    {
        "template_name": "meeting_announcement_rich",
        "pattern": (
            "The agenda lists a remote check-in on {other_date}, but confirms that {person} will speak in {location} on {date}. "
            "A second venue was considered earlier and later removed."
        ),
        "instruction_pool": ie_instructions,
    },
    {
        "template_name": "conference_attendance_rich",
        "pattern": (
            "On {date}, {person} attended a conference in {location}. Several attendees arrived the day before, "
            "and one side session was postponed to {other_date}."
        ),
        "instruction_pool": ie_instructions,
    },
    {
        "template_name": "appointment_notice_rich",
        "pattern": (
            "{person} was appointed in {location} on {date}. Another internal memo mentioned a review meeting on {other_date}, "
            "but no change to the appointment record."
        ),
        "instruction_pool": ie_instructions,
    },
    {
        "template_name": "event_registration_rich",
        "pattern": (
            "The registration confirms that {person} is expected in {location} on {date}. An earlier draft used a placeholder name "
            "and listed {other_location} before the final update."
        ),
        "instruction_pool": ie_instructions,
    },
    {
        "template_name": "travel_schedule_notice",
        "pattern": (
            "{person} is scheduled to arrive in {location} on {date} for the workshop. The return trip begins later from "
            "{other_location}, and a planning call is set for {other_date}."
        ),
        "instruction_pool": ie_instructions,
    },
    {
        "template_name": "speaker_log_entry",
        "pattern": (
            "The log entry states that {person} checked in for the session in {location} on {date}. A rehearsal note refers to "
            "{other_date}, but that date belongs to the setup period."
        ),
        "instruction_pool": ie_instructions,
    },
]


# -------------------------------------------------------------------
# Rule-based transformation templates
# More varied punctuation, casing, and token structure.
# -------------------------------------------------------------------

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
        "template_name": "quote_and_symbol_note",
        "pattern": "\"{person}\" said: pack {n} cables, {n2} adapters, and re-check {location}!",
    },
    {
        "template_name": "slash_and_dash_update",
        "pattern": "Status update - {person}/{location}: {n} forms filed, {n2} pending.",
    },
    {
        "template_name": "mixed_caps_checklist",
        "pattern": "CHECKLIST for {person}: Bring {n} IDs, {n2} copies, and call {location}.",
    },
    {
        "template_name": "word_length_test",
        "pattern": "Tiny words vanish slowly when elephantine expressions dominate paragraphs.",
    },
]


# -------------------------------------------------------------------
# Extractive QA templates
# Increased difficulty: richer passages, multiple candidate spans,
# and less repetitive schemas while keeping exact-match scoring.
# -------------------------------------------------------------------

qa_templates = [
    {
        "template_name": "schedule_passage_rich",
        "passage": (
            "{person} left Rome on {other_date}, arrived in {location} on {date} for a training session, "
            "and later continued to {other_location} for a short meeting."
        ),
        "question": "Where did {person} arrive on {date}?",
        "answer_field": "location",
        "instruction_pool": qa_instructions,
        "required_fields": ["person", "location", "date", "other_date", "other_location"],
    },
    {
        "template_name": "release_comparison_passage",
        "passage": (
            "Earlier drafts mentioned a prototype from {other_company}, but the final report says that {company} released "
            "the {product} on {date}. Analysts compared it with a separate launch planned for {other_date}."
        ),
        "question": "What did {company} release?",
        "answer_field": "product",
        "instruction_pool": qa_instructions,
        "required_fields": ["company", "product", "date", "other_company", "other_date"],
    },
    {
        "template_name": "travel_note_rich",
        "passage": (
            "On {date}, {person} left {other_location} in the morning and reached {location} that evening. "
            "A second traveler arrived in Rome the same day."
        ),
        "question": "Who reached {location} that evening?",
        "answer_field": "person",
        "instruction_pool": qa_instructions,
        "required_fields": ["person", "location", "date", "other_location"],
    },
    {
        "template_name": "project_summary_rich",
        "passage": (
            "{person} said the meeting in {location} would start on {date}, even though an earlier draft listed {other_date}. "
            "The venue itself did not change."
        ),
        "question": "Who said the meeting in {location} would start on {date}?",
        "answer_field": "person",
        "instruction_pool": qa_instructions,
        "required_fields": ["person", "location", "date", "other_date"],
    },
    {
        "template_name": "hosted_event_passage",
        "passage": (
            "{company} hosted a workshop in {location} on {date}. The invitation also referred to a later review in "
            "{other_location}, but that follow-up had no confirmed venue."
        ),
        "question": "Which company hosted the workshop in {location}?",
        "answer_field": "company",
        "instruction_pool": qa_instructions,
        "required_fields": ["company", "location", "date", "other_location"],
    },
    {
        "template_name": "arrival_record_with_multiple_dates",
        "passage": (
            "The travel record notes that {person} was expected on {other_date}, but the confirmed arrival in {location} "
            "occurred on {date}. A final check-in was logged the following morning."
        ),
        "question": "Who had the confirmed arrival in {location}?",
        "answer_field": "person",
        "instruction_pool": qa_instructions,
        "required_fields": ["person", "location", "date", "other_date"],
    },
    {
        "template_name": "product_brief_with_competitor",
        "passage": (
            "{other_company} previewed a headset earlier in the month, but the briefing says {company} officially introduced "
            "the {product} on {date} after the press session in {location}."
        ),
        "question": "Which product did {company} officially introduce?",
        "answer_field": "product",
        "instruction_pool": qa_instructions,
        "required_fields": ["other_company", "company", "product", "date", "location"],
    },
    {
        "template_name": "meeting_city_resolution",
        "passage": (
            "{person} first suggested meeting in {other_location}, but the final note states that the session would be held in "
            "{location} on {date}. Another draft changed only the start time."
        ),
        "question": "Where would the session be held on {date}?",
        "answer_field": "location",
        "instruction_pool": qa_instructions,
        "required_fields": ["person", "other_location", "location", "date"],
    },
    {
        "template_name": "speaker_schedule_resolution",
        "passage": (
            "Although the draft agenda listed {other_person} for the opening remarks, the final schedule says that {person} "
            "will speak in {location} on {date}."
        ),
        "question": "Who will speak in {location} on {date}?",
        "answer_field": "person",
        "instruction_pool": qa_instructions,
        "required_fields": ["other_person", "person", "location", "date"],
    },
    {
        "template_name": "deployment_update_passage",
        "passage": (
            "The deployment note says {company} will roll out the {product} on {date} after testing in {location}. "
            "A separate memo about {other_company} concerned a different tool."
        ),
        "question": "Which company will roll out the {product} on {date}?",
        "answer_field": "company",
        "instruction_pool": qa_instructions,
        "required_fields": ["company", "product", "date", "location", "other_company"],
    },
    {
        "template_name": "venue_confirmation_note",
        "passage": (
            "The draft first mentioned {other_location}, but the final confirmation says that {company} will present "
            "the {product} in {location} on {date}."
        ),
        "question": "Where will {company} present the {product} on {date}?",
        "answer_field": "location",
        "instruction_pool": qa_instructions,
        "required_fields": ["company", "product", "location", "other_location", "date"],
    },
    {
        "template_name": "product_owner_note",
        "passage": (
            "The comparison memo says the {product} belongs to {company}, while a separate note about {other_company} "
            "concerns a different release planned for {other_date}."
        ),
        "question": "Which company does the {product} belong to?",
        "answer_field": "company",
        "instruction_pool": qa_instructions,
        "required_fields": ["product", "company", "other_company", "other_date"],
    },
]


# -------------------------------------------------------------------
# Helper pools used by richer IE/QA templates
# -------------------------------------------------------------------

other_locations = [
    "Naples",
    "Turin",
    "Lyon",
    "Munich",
    "Seville",
    "Porto",
    "Salzburg",
    "Brno",
    "Patras",
    "Cork",
]

other_people = [
    "Laura Bianchi",
    "Marco Silva",
    "Nina Petrova",
    "Oscar Klein",
    "Paula Mendes",
    "Rami Haddad",
    "Sara Novak",
    "Tomas Greco",
    "Valeria Costa",
    "Yusuf Demir",
]

other_companies = [
    "BrightCore Systems",
    "Nova Health Labs",
    "Peak Ledger Group",
    "GreenBridge Analytics",
    "Vertex Retail Tech",
    "Cobalt Media Works",
]

other_dates = [
    "2024-01-22",
    "2024-04-02",
    "2024-06-18",
    "2024-09-14",
    "2025-02-28",
    "2025-05-07",
    "2025-07-29",
    "2025-10-15",
    "2026-01-21",
    "2026-03-03",
]