from __future__ import annotations

import json

from config import REPAIR_PROMPT_PATH, REPAIR_STAGE
from cot_utils import extract_visible_output_text, join_thought_units, split_thought_units
from io_utils import build_payload, call_chat_completion, extract_first_json_object, load_text, render_prompt


def generate_repaired_suffix(case: dict, prefix_units: list[str], l_value: int) -> dict:
    template = load_text(REPAIR_PROMPT_PATH)
    prefix_text = join_thought_units(prefix_units)
    prompt = render_prompt(
        template,
        {
            "question": case["problem"],
            "prefix_thought": prefix_text,
            "answer": case["gold_answer"],
            "solution": case["solution"],
            "L": l_value,
        },
    )
    payload = build_payload(
        model=REPAIR_STAGE.endpoint.model,
        prompt=prompt,
        temperature=REPAIR_STAGE.temperature,
        max_tokens=REPAIR_STAGE.max_tokens,
    )
    raw_output = call_chat_completion(
        base_url=REPAIR_STAGE.endpoint.base_url,
        api_key=REPAIR_STAGE.endpoint.api_key,
        payload=payload,
        timeout=REPAIR_STAGE.timeout,
    )

    visible_text = extract_visible_output_text(raw_output)
    parsed_text = visible_text
    parsed_units: list[str] = []
    parsed_mode = "plain_text"

    json_text = extract_first_json_object(visible_text)
    if json_text:
        try:
            json_data = json.loads(json_text)
            thought_fragments = json_data.get("thought_fragments")
            if isinstance(thought_fragments, list):
                parsed_units = [str(item).strip() for item in thought_fragments if str(item).strip()]
                parsed_text = join_thought_units(parsed_units)
                parsed_mode = "json"
        except json.JSONDecodeError:
            parsed_units = []

    if not parsed_units:
        parsed_units = split_thought_units(parsed_text)

    exact_length_ok = len(parsed_units) == l_value
    truncated = False
    if len(parsed_units) > l_value:
        parsed_units = parsed_units[:l_value]
        truncated = True
    repaired_text = join_thought_units(parsed_units)

    return {
        "raw_output": raw_output,
        "parsed_text": parsed_text,
        "parsed_mode": parsed_mode,
        "repaired_units": parsed_units,
        "repaired_text": repaired_text,
        "repaired_unit_count": len(parsed_units),
        "exact_length_ok": exact_length_ok,
        "truncated_to_l": truncated,
    }
