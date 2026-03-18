from __future__ import annotations

from typing import Iterable

from config import FAILED_REPORT_PATH, SUCCESS_REPORT_PATH
from io_utils import load_json


def _normalize_case(case: dict, source: str) -> dict:
    return {
        "source": source,
        "is_original_correct": source == "success",
        "dataset_idx": case.get("dataset_idx"),
        "unique_id": case.get("unique_id"),
        "problem": case.get("problem", ""),
        "solution": case.get("solution", ""),
        "gold_answer": case.get("gold_answer", ""),
        "pred_boxed": case.get("pred_boxed", ""),
        "reason": case.get("reason", ""),
        "model_output": case.get("model_output", ""),
        "retry_result": case.get("retry_result"),
        "retry_successful": case.get("retry_successful"),
    }


def _load_cases_from_report(path, source: str) -> list[dict]:
    payload = load_json(path)
    cases = payload.get("cases", [])
    return [_normalize_case(case, source) for case in cases if isinstance(case, dict)]


def load_success_cases() -> list[dict]:
    return _load_cases_from_report(SUCCESS_REPORT_PATH, "success")


def load_failed_cases() -> list[dict]:
    return _load_cases_from_report(FAILED_REPORT_PATH, "failed")


def load_cases(sources: Iterable[str]) -> list[dict]:
    source_set = {item.strip().lower() for item in sources}
    cases: list[dict] = []
    if "success" in source_set:
        cases.extend(load_success_cases())
    if "failed" in source_set:
        cases.extend(load_failed_cases())
    cases.sort(key=lambda item: (str(item["source"]), int(item["dataset_idx"])))
    return cases
