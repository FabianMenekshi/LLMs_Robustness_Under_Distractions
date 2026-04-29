# Evaluation freeze: Phase 0

This directory freezes the evaluation contract for the Instruction Distraction Robustness Benchmark before any model runs begin.

## Frozen benchmark input

The benchmark input is the prompt-instance dataset:

- `data/prompts/prompt_instances.jsonl`

This file is the canonical evaluation dataset. It must not be edited, filtered, reordered for convenience, or regenerated once model evaluation starts.

## Frozen benchmark size assumptions

The evaluation pipeline assumes the following fixed benchmark structure:

- 250 base examples total
- 50 base examples per task
- 4000 prompt instances total
- 16 prompt instances per base example
- 500 clean prompt instances
- 3500 distracted prompt instances
- 2000 bounded prompt instances
- 2000 unbounded prompt instances
- 500 prompt instances for each distraction type

## Frozen task set

The benchmark contains exactly 5 tasks:

1. `single_label_classification`
2. `multi_label_classification`
3. `information_extraction`
4. `rule_based_transformation`
5. `extractive_qa`

## Frozen prompt condition set

The benchmark contains exactly 2 regimes:

- `bounded`
- `unbounded`

The benchmark contains exactly 8 evaluation conditions:

- `clean`
- `irrelevant_prefix`
- `irrelevant_suffix`
- `instruction_in_the_middle`
- `conflicting_instruction`
- `negation_distraction`
- `style_distraction`
- `length_stress`

## Frozen scoring contract

The benchmark is scored at the prompt-instance level.

Primary metric:

- exact-match accuracy

Parsing and scoring policy:

- model outputs are parsed strictly
- parse failures count as incorrect
- schema failures count as incorrect
- raw model outputs are stored for auditability
- parsed outputs are stored for auditability
- per-example correctness is stored explicitly

Task-level scoring rules:

- single-label classification: exact match on canonical JSON
- multi-label classification: exact match on canonical JSON after label deduplication and sorting
- information extraction: exact match on canonical JSON with exact required keys
- rule-based transformation: exact match on canonical JSON
- extractive QA: exact match on canonical JSON with exact gold span match

## Prompting policy for models

The benchmark's `prompt_text` field is the exact task input sent to the model.

Evaluation policy:

- use the benchmark `prompt_text` as the user-facing content
- do not rewrite or simplify benchmark prompts
- do not add extra task hints
- do not add benchmark-specific rescue instructions
- do not add chain-of-thought requests
- use each instruct model's normal chat/template interface when needed
- keep decoding deterministic

Default decoding policy:

- `temperature = 0.0`
- `do_sample = false`
- `top_p = 1.0`

## Frozen main model suite

Main 4-model comparison:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`

Optional extension model:

- `Qwen/Qwen2.5-14B-Instruct`

## Output policy

Each model run should produce a raw prediction file with at least:

- `prompt_id`
- `base_example_id`
- `model_name`
- `raw_output`
- `parsed_output`
- `parse_status`
- `is_correct`
- `task_name`
- `regime`
- `distraction_type`
- `gold_output`

## Change-control rule

Once Phase 0 is accepted, the following items are frozen unless a documented benchmark-version bump is created:

- benchmark JSONL input
- scoring rules
- parsing strictness
- model suite for the main comparison
- decoding defaults
- result file schema
