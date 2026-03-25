import json
import random
import re
from dataclasses import asdict
from typing import List, Dict, Any

from src.templates import (
    CandidateExample,
    SINGLE_LABEL_SET,
    MULTI_LABEL_SET,
    IE_SCHEMA_KEYS,
    RULE_SET,
    people,
    locations,
    dates,
    products,
    companies,
    single_label_templates,
    single_label_value_map,
    multi_label_templates,
    topic_pool,
    multi_label_combinations,
    ie_templates,
    transformation_input_templates,
    rule_instructions,
    qa_templates,
)

# Reproducibility: generation will be deterministic across runs
random.seed(42)