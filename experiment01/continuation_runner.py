from __future__ import annotations

import sys
from pathlib import Path

from config import CONTINUATION_STAGE, CONTINUE_SOLVE_PROMPT_PATH, EQUIV_JUDGE_ENDPOINT, EQUIV_JUDGE_MAX_TOKENS
from cot_utils import count_thought_units, extract_reasoning_text, join_thought_units
from io_utils import build_payload, call_chat_completion, load_text, render_prompt


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from answer_judge import is_correct, llm_fallback_equivalence_judge  # noqa: E402


def run_continuation(case: dict, prefix_units: list[str], seed_units: list[str], seed_type: str) -> dict:
    template = load_text(CONTINUE_SOLVE_PROMPT_PATH)
    prompt = render_prompt(
        template,
        {
            "question": case["problem"],
            "existing_thought": join_thought_units(prefix_units + seed_units),
            "seed_type": seed_type,
        },
    )
    payload = build_payload(
        model=CONTINUATION_STAGE.endpoint.model,
        prompt=prompt,
        temperature=CONTINUATION_STAGE.temperature,
        max_tokens=CONTINUATION_STAGE.max_tokens,
    )
    raw_output = call_chat_completion(
        base_url=CONTINUATION_STAGE.endpoint.base_url,
        api_key=CONTINUATION_STAGE.endpoint.api_key,
        payload=payload,
        timeout=CONTINUATION_STAGE.timeout,
    )

    ok, reason, pred_boxed = is_correct(raw_output, case["gold_answer"])
    if (not ok) and reason.startswith("不匹配:") and pred_boxed:
        try:
            judge_ok, judge_raw = llm_fallback_equivalence_judge(
                base_url=EQUIV_JUDGE_ENDPOINT.base_url,
                api_key=EQUIV_JUDGE_ENDPOINT.api_key,
                model=EQUIV_JUDGE_ENDPOINT.model,
                pred_boxed=pred_boxed,
                gold_answer=case["gold_answer"],
                call_chat_completion_fn=lambda base_url, api_key, payload: call_chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    payload=payload,
                    timeout=CONTINUATION_STAGE.timeout,
                ),
                judge_max_tokens=EQUIV_JUDGE_MAX_TOKENS,
            )
            if judge_ok:
                ok = True
                reason = "LLM 兜底判定等价"
            else:
                reason = f"{reason}; LLM兜底={judge_raw}"
        except Exception as exc:
            reason = f"{reason}; LLM兜底异常={exc}"

    continuation_reasoning = extract_reasoning_text(raw_output)
    continuation_unit_count = count_thought_units(continuation_reasoning)
    return {
        "seed_type": seed_type,
        "raw_output": raw_output,
        "continuation_reasoning": continuation_reasoning,
        "continuation_unit_count": continuation_unit_count,
        "pred_boxed": pred_boxed,
        "correct": ok,
        "reason": reason,
    }
