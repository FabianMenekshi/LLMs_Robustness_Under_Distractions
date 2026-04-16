- File Structure:

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