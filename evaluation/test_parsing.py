from parsing import parse_prediction
from scoring import score_prediction

# Example from the benchmark
raw_output = '{"answer": "Urban Sports Media"}'

# Extra test cases
# raw_output = "Urban Sports Media"
# raw_output = '{"answer": "Urban Sports Media", "extra": "oops"}'
# raw_output = '{"answer": "Wrong Company"}'

gold = {"answer": "Urban Sports Media"}

task_name = "extractive_qa"

parsed = parse_prediction(raw_output, task_name)
score = score_prediction(parsed, gold, task_name)

print("Parsed:", parsed)
print("Score:", score)