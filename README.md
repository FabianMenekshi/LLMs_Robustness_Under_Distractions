# <h1 align="center">*<ins>NLP Prompt Robustness Benchmark</ins>*</h1>

A thesis-oriented benchmark pipeline for studying how NLP task performance changes under prompt wrappers, distractors, and instruction conflicts.

This project builds a **balanced base dataset** across several task families, then expands each base example into a **prompt-robustness benchmark** with bounded and unbounded prompt regimes plus multiple distraction conditions.



## Overview
  
The benchmark is designed to test whether models can still follow the true task instruction when prompts include:

- irrelevant text before or after the task
- buried instructions
- conflicting instructions
- negation-based distractions
- style pressure
- long irrelevant context

The pipeline has two major stages:

1. **Base dataset construction**
   - Generate balanced task examples with gold outputs.
   - Validate structure, diversity, and distribution.

2. **Prompt instance generation**
   - Wrap each base example in multiple prompt regimes and distraction conditions.
   - Validate prompt-level balance and metadata coverage.




## Research Goal

The objective is not only to measure raw task performance, but also to analyze:

- robustness to conflicting or irrelevant prompt content
- performance differences between bounded and unbounded prompts
- failure patterns by distraction subtype
- sensitivity to prompt wording and layout
- extractive faithfulness under interference

The benchmark is especially useful for studying whether a model:

- follows the true task instruction,
- gets distracted by nearby irrelevant content,
- obeys later contradictory instructions,
- switches to the wrong format or style,
- or paraphrases when exact extraction is required.



## Task Families

The base dataset contains 5 task families with **50 examples each**.

### 1. Single-label classification
Models assign exactly one label from a fixed label set.

### 2. Multi-label classification
Models assign all applicable labels from a fixed label set and return them in alphabetical order.

### 3. Information extraction
Models extract structured fields such as:
- person
- date
- location

### 4. Rule-based transformation
Models transform text according to an explicit deterministic rule.

### 5. Extractive QA
Models answer a question with an exact span from the passage.



## Prompt Regimes

Each base example is expanded into two prompt regimes.

### Bounded
Bounded prompts preserve explicit structure using visible tags such as:

- `<TASK>`
- `<INPUT>`

### Unbounded
Unbounded prompts do **not** simply remove tags from bounded prompts.
Instead, they are rendered as distinct naturalistic surfaces such as:

- direct messages
- work notes
- pasted requests
- email-like prompts
- chat-like prompts
- workflow-like notes



## Distraction Conditions

Each base example is rendered under 8 conditions:

1. `clean`
2. `irrelevant_prefix`
3. `irrelevant_suffix`
4. `instruction_in_the_middle`
5. `conflicting_instruction`
6. `negation_distraction`
7. `style_distraction`
8. `length_stress`

### Clean
A realistic but non-distracted prompt.

### Irrelevant prefix / suffix
A short unrelated block appears before or after the true task.

### Instruction in the middle
The real task is buried between unrelated blocks.

### Conflicting instruction
An additional instruction attempts to redirect the model toward the wrong task, format, or output mode.

### Negation distraction
A misleading negation or softened reversal tries to weaken the real task instruction.

### Style distraction
A strong style request creates tension with the correct task output.

### Length stress
A long irrelevant block increases context pressure and tests whether the model still follows the real task.



## Repository Structure

```text
.
├── .gitignore
├── README.md
├── structure.txt
├── test.ipynb
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


<!-- 

## Source Modules

### `src/templates.py`
Defines the task content inventory, including:
- label sets
- entity pools
- task templates
- instruction pools
- QA templates and answer fields
- transformation rules

This file controls the semantic content used to generate candidate examples.

### `src/generation.py`
Generates candidate examples and selects balanced final examples.

Key responsibilities:
- generate candidate pools for each task
- apply deterministic transformation rules
- construct QA passages and questions
- enforce stronger balancing during selection
- support review status and final exact selection

### `src/base_dataset.py`
Converts selected examples into final base dataset records and builds dataset summaries.

### `src/validation.py`
Validates the base dataset.

Checks include:
- schema correctness
- task counts
- unique example ids
- duplicate rendered inputs
- instruction diversity
- template diversity
- template concentration
- QA answer-field diversity
- QA answer-field concentration

### `src/prompt_templates.py`
Defines prompt regimes, layouts, openers, unbounded surfaces, and distraction inventories.

This includes:
- bounded openers
- bounded layouts
- unbounded prompt surfaces
- short-noise blocks
- long-noise blocks
- style distractions
- conflict variants
- negation variants

### `src/prompt_instance_generation.py`
Builds prompt instances from the base dataset.

Key responsibilities:
- create one prompt instance for every base example / regime / distraction combination
- assign balanced variant indices
- attach metadata for surfaces, layouts, and distraction variants
- generate prompt-level summaries
- create preview samples

### `src/prompt_instance_validation.py`
Validates prompt instances.

Checks include:
- required metadata fields
- regime/distraction integrity
- prompt id uniqueness
- balanced counts across conditions
- subtype availability and metadata consistency

### `src/prompt_builder.py`
Used for rendering and previewing prompts in a structured way. It mirrors the current prompt-generation logic closely enough for inspection and notebook previews.



## Notebooks

### `build_base_dataset.ipynb`
Builds and validates the base dataset.

Typical outputs:
- `data/base/base_examples.json`
- `data/base/base_examples.jsonl`
- `data/base/dataset_summary.json`
- `data/base/validation_report.json`

### `generate_prompt_instances.ipynb`
Generates prompt instances from the base dataset.

Typical outputs:
- `data/prompts/prompt_instances.json`
- `data/prompts/prompt_instances.jsonl`
- `data/prompts/prompt_instance_summary.json`
- `data/prompts/prompt_instance_validation.json`
- `data/prompts/prompt_preview_samples.json`

### `prompt_design.ipynb`
Used to inspect prompt regimes, layouts, and wrapper logic in detail.

### `design_template.ipynb`
Supports inspection and refinement of prompt-design components.

### `lock_design.ipynb`
Used to lock or export design specifications.



## Data Outputs

### Base dataset outputs

#### `base_examples.json` / `base_examples.jsonl`
The finalized base dataset used for expansion into prompt instances.

#### `dataset_summary.json`
Summarizes the base dataset, including:
- counts by task
- counts by template
- counts by instruction
- counts by task and template
- instruction diversity by task
- template diversity by task
- QA answer-field summaries

#### `validation_report.json`
Validation report for the base dataset.

### Prompt dataset outputs

#### `prompt_instances.json` / `prompt_instances.jsonl`
The full prompt-robustness benchmark.

Each base example becomes multiple prompt instances across regimes and distraction types.

#### `prompt_instance_summary.json`
Summarizes prompt-level distributions, including:
- counts by task
- counts by regime
- counts by distraction type
- counts by subtype
- counts by prompt surface type
- counts by layout
- counts by surface id
- counts by opener id
- counts by conflict / negation / style variants
- counts by noise block ids
- counts by long-noise block ids

#### `prompt_instance_validation.json`
Validation report for prompt instances.

#### `prompt_preview_samples.json`
A human-readable preview artifact used for qualitative inspection.

The preview sampler is stratified so it is more representative than simply taking the first row from each condition. -->



## End-to-End Pipeline

### 1: Build the base dataset
Run `build_base_dataset.ipynb`.

This will:
1. generate or load candidate examples
2. validate them
3. select exactly 50 per task
4. export the final base dataset
5. export summary and validation files

### 2: Inspect base outputs
Check:
- `dataset_summary.json`
- `validation_report.json`

These tell you whether task counts, template diversity, and QA answer-field balance look correct.

### 3: Generate prompt instances
Run `generate_prompt_instances.ipynb`.

This will:
1. load the base dataset
2. generate all regime/distraction combinations
3. validate the prompt dataset
4. export prompt summaries and preview samples

### 4: Inspect prompt outputs
Check:
- `prompt_instance_summary.json`
- `prompt_instance_validation.json`
- `prompt_preview_samples.json`

These tell you whether prompt-level balance and qualitative diversity are acceptable.



## Expected Dataset Sizes

### Base dataset
- 5 task families
- 50 examples per task
- **250 total base examples**

### Prompt dataset
For each base example:
- 2 regimes
- 8 distraction conditions

So:
- 250 × 2 × 8 = **4000 prompt instances**



## Quick Start

### Build the base dataset
Run:
- `notebooks/build_base_dataset.ipynb`

Then inspect:
- `data/base/dataset_summary.json`
- `data/base/validation_report.json`

### Build the prompt dataset
Run:
- `notebooks/generate_prompt_instances.ipynb`

Then inspect:
- `data/prompts/prompt_instance_summary.json`
- `data/prompts/prompt_instance_validation.json`
- `data/prompts/prompt_preview_samples.json`