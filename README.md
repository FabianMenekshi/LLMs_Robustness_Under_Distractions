# <h1 align="center">*<ins>LLM Prompt Robustness Benchmark</ins>*</h1>

A thesis-oriented benchmark pipeline for studying how LLMs handle **noisy, misleading, and realistic prompts**.

This project builds a **balanced base dataset** across several task families, then expands each base example into a **prompt-robustness benchmark** with bounded and unbounded prompt regimes plus multiple distraction conditions.

---

## Overview

The benchmark evaluates whether models can still follow the correct task when prompts include:

- irrelevant text (before/after)
- buried instructions
- conflicting instructions
- negation cues
- stylistic pressure
- long distracting context

The pipeline has two major stages:

1. **Base dataset construction**
   - Generate balanced task examples with gold outputs.
   - Validate structure, diversity, and distribution.

2. **Prompt instance generation**
   - Wrap each base example in multiple prompt regimes and distraction conditions.
   - Validate prompt-level balance and metadata coverage.

### Task Families

Each with 50 examples:

1. **Single-label classification** -> one label from a fixed set  
2. **Multi-label classification** -> multiple labels (sorted)  
3. **Information extraction** -> structured fields (person, date, location)  
4. **Rule-based transformation** -> deterministic text modification  
5. **Extractive QA** -> exact answer span from passage  

### Prompt Regimes

Each base example is rendered in two forms:

- **Bounded**  
  Structured prompts with explicit sections (task + input)

- **Unbounded**  
  Naturalistic prompts (messages, notes, emails, etc.)


### Pipeline

1. **Base dataset**
   - 5 task families, 50 examples each (250 total base examples)
   - balanced, validated and automatically scorable

2. **Prompt benchmark**
   - 2 regimes x 8 distraction types = 16 prompt instances for each base example (250 x 16 = 4000 prompt instances)
   - fully annotated for analysis

---

## Research Goal

The benchmark measures **robustness to prompt interference**, including:
   - sensitivity to irrelevant or conflicting instructions  
   - differences between **bounded vs unbounded prompts**  
   - failure modes by distraction type  
   - format/style deviations  
   - extractive faithfulness under noise  

---

## Repository Structure

```text
.
├── .gitignore
├── README.md
├── data/
│   ├── base/
│   │   ├── base_examples.json
│   │   ├── base_examples.jsonl
│   │   ├── dataset_summary.json
│   │   └── validation_report.json
│   ├── candidates/
│   │   └── candidates.jsonl
│   ├── prompts/
│   │   ├── prompt_instances.json
│   │   ├── prompt_instances.jsonl
│   │   ├── prompt_instance_summary.json
│   │   ├── prompt_instance_validation.json
│   │   └── prompt_preview_samples.json
│   ├── reviewed/
│   │   └── selected_base_examples.jsonl
│   └── specs/
│       ├── benchmark_spec.json
│       └── prompt_design_spec.json
├── notebooks/
│   ├── build_base_dataset.ipynb
│   ├── design_template.ipynb
│   ├── generate_prompt_instances.ipynb
│   ├── lock_design.ipynb
│   └── prompt_design.ipynb
└── src/
    ├── base_dataset.py
    ├── generation.py
    ├── prompt_builder.py
    ├── prompt_instance_generation.py
    ├── prompt_instance_validation.py
    ├── prompt_templates.py
    ├── templates.py
    └── validation.py
```

---

## Pipeline Usage

### 1. Build base dataset
Run:
```bash
notebooks/build_base_dataset.ipynb
```

Check:
- `data/base/dataset_summary.json`
- `data/base/validation_report.json`



### 2. Generate prompt benchmark
Run:
```bash
notebooks/generate_prompt_instances.ipynb
```

Check:
- `data/prompts/prompt_instance_summary.json`
- `data/prompts/prompt_instance_validation.json`
- `data/prompts/prompt_preview_samples.json`

### The benchmark:
```bash
data/prompts/prompt_instances.jsonl
```