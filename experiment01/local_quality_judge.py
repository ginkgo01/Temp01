from __future__ import annotations

import json
import time

from config import LOCAL_JUDGE_PROMPT_PATH, LOCAL_JUDGE_STAGE
from io_utils import (
    build_payload,
    call_chat_completion,
    extract_first_json_object,
    load_text,
    render_prompt,
)


METRIC_KEYS = [
    "truthfulness",
    "coherence",
    "correctness",
    "progress",
    "non_leakage",
]

VALID_LABELS = ("T", "M", "F")

MAX_LOCAL_JUDGE_RETRIES = 10
LOCAL_JUDGE_RETRY_SLEEP_SECONDS = 2.0


class LocalJudgeUnavailableError(RuntimeError):
    pass


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().upper()
    if normalized in VALID_LABELS:
        return normalized

    alias_map = {
        "TRUE": "T",
        "FALSE": "F",
        "NEUTRAL": "M",
        "MIDDLE": "M",
        "MIXED": "M",
        "UNCERTAIN": "M",
        "AMBIGUOUS": "M",
        "模糊": "M",
        "中性": "M",
    }
    return alias_map.get(normalized, "F")


def _normalize_metric_block(data: dict | None) -> dict:
    block = data if isinstance(data, dict) else {}
    return {
        "label": _normalize_label(block.get("label", "F")),
        "reason": str(block.get("reason", "")).strip(),
    }


def judge_local_quality(case: dict, prefix_text: str, original_suffix_text: str, repaired_suffix_text: str, l_value: int) -> dict:
    template = load_text(LOCAL_JUDGE_PROMPT_PATH)
    prompt = render_prompt(
        template,
        {
            "question": case["problem"],
            "prefix_thought": prefix_text,
            "original_suffix": original_suffix_text,
            "repaired_suffix": repaired_suffix_text,
            "answer": case["gold_answer"],
            "solution": case["solution"],
            "L": l_value,
        },
    )
    payload = build_payload(
        model=LOCAL_JUDGE_STAGE.endpoint.model,
        prompt=prompt,
        temperature=LOCAL_JUDGE_STAGE.temperature,
        max_tokens=LOCAL_JUDGE_STAGE.max_tokens,
    )
    last_error_message = ""

    for attempt in range(1, MAX_LOCAL_JUDGE_RETRIES + 1):
        try:
            raw_output = call_chat_completion(
                base_url=LOCAL_JUDGE_STAGE.endpoint.base_url,
                api_key=LOCAL_JUDGE_STAGE.endpoint.api_key,
                payload=payload,
                timeout=LOCAL_JUDGE_STAGE.timeout,
            )
        except Exception as exc:
            last_error_message = f"attempt={attempt}; exception={exc}"
            if attempt < MAX_LOCAL_JUDGE_RETRIES:
                time.sleep(LOCAL_JUDGE_RETRY_SLEEP_SECONDS)
                continue
            raise LocalJudgeUnavailableError(last_error_message) from exc

        json_text = extract_first_json_object(raw_output)
        parsed_ok = False
        parsed_data: dict = {}
        if json_text:
            try:
                parsed_data = json.loads(json_text)
                parsed_ok = True
            except json.JSONDecodeError as exc:
                last_error_message = f"attempt={attempt}; json_decode_error={exc}; raw_output={raw_output[:500]}"
        else:
            last_error_message = f"attempt={attempt}; no_json_found; raw_output={raw_output[:500]}"

        if parsed_ok:
            normalized = {key: _normalize_metric_block(parsed_data.get(key)) for key in METRIC_KEYS}
            return {
                "parsed_ok": True,
                "raw_output": raw_output,
                "metrics": normalized,
                "attempts_used": attempt,
            }

        if attempt < MAX_LOCAL_JUDGE_RETRIES:
            time.sleep(LOCAL_JUDGE_RETRY_SLEEP_SECONDS)

    raise LocalJudgeUnavailableError(last_error_message or "local judge failed after retries")
