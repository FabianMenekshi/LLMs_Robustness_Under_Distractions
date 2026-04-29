from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from parsing import ParseResult


@dataclass
class ScoreResult:
    success: bool
    task_name: str
    is_correct: bool
    score: int
    error_code: Optional[str]
    error_detail: Optional[str]
    parsed_output: Optional[Dict[str, Any]]
    gold_output: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_name": self.task_name,
            "is_correct": self.is_correct,
            "score": self.score,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "parsed_output": self.parsed_output,
            "gold_output": self.gold_output,
        }


def _canonical_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_gold_for_task(task_name: str, gold_output: Dict[str, Any]) -> Dict[str, Any]:
    if task_name == "multi_label_classification":
        labels = gold_output.get("labels", [])
        if isinstance(labels, list):
            return {"labels": sorted(set(str(label).strip() for label in labels))}
    return gold_output


def _score_exact_match(task_name: str, parsed_output: Dict[str, Any], gold_output: Dict[str, Any]) -> ScoreResult:
    normalized_gold = _normalize_gold_for_task(task_name, gold_output)
    is_correct = _canonical_json(parsed_output) == _canonical_json(normalized_gold)

    return ScoreResult(
        success=True,
        task_name=task_name,
        is_correct=is_correct,
        score=1 if is_correct else 0,
        error_code=None if is_correct else "wrong_answer",
        error_detail=None if is_correct else "Parsed output does not match gold output under canonical exact match.",
        parsed_output=parsed_output,
        gold_output=normalized_gold,
    )


def score_prediction(parsed_result: ParseResult, gold_output: Dict[str, Any], task_name: str) -> ScoreResult:
    if not parsed_result.success:
        return ScoreResult(
            success=False,
            task_name=task_name,
            is_correct=False,
            score=0,
            error_code=parsed_result.error_code,
            error_detail=parsed_result.error_detail,
            parsed_output=None,
            gold_output=_normalize_gold_for_task(task_name, gold_output),
        )

    parsed_output = parsed_result.parsed_output
    if parsed_output is None:
        return ScoreResult(
            success=False,
            task_name=task_name,
            is_correct=False,
            score=0,
            error_code="missing_parsed_output",
            error_detail="Parser reported success but returned no parsed output.",
            parsed_output=None,
            gold_output=_normalize_gold_for_task(task_name, gold_output),
        )

    return _score_exact_match(task_name, parsed_output, gold_output)
