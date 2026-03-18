from __future__ import annotations

import re


THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{", re.DOTALL)


def extract_reasoning_text(text: str) -> str:
    if not text:
        return ""

    match = THINK_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        return text[: boxed_match.start()].strip()
    return text.strip()


def extract_visible_output_text(text: str) -> str:
    if not text:
        return ""

    lower_text = text.lower()
    closing_tag = "</think>"
    end_idx = lower_text.find(closing_tag)
    if end_idx != -1:
        visible = text[end_idx + len(closing_tag) :]
        return visible.lstrip("\n").strip()

    return text.strip()


def split_thought_units(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def join_thought_units(units: list[str]) -> str:
    return "\n\n".join(unit.strip() for unit in units if unit.strip())


def count_thought_units(text: str) -> int:
    return len(split_thought_units(text))


def get_valid_cut_points(units: list[str], l_value: int) -> list[int]:
    if len(units) < 2 * l_value:
        return []
    return list(range(l_value, len(units) - l_value + 1, l_value))


def build_slice(units: list[str], cut_idx: int, l_value: int) -> dict:
    prefix_units = units[:cut_idx]
    original_suffix_units = units[cut_idx : cut_idx + l_value]
    return {
        "cut_idx": cut_idx,
        "prefix_units": prefix_units,
        "prefix_text": join_thought_units(prefix_units),
        "original_suffix_units": original_suffix_units,
        "original_suffix_text": join_thought_units(original_suffix_units),
    }
