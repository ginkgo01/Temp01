from __future__ import annotations

from statistics import mean

from local_quality_judge import METRIC_KEYS

LABEL_KEYS = ("T", "M", "F")


def _init_group() -> dict:
    return {
        "slice_count": 0,
        "local_judge_count": 0,
        "continuation_pair_count": 0,
        "both_correct_count": 0,
        "metric_totals": {key: {label: 0 for label in LABEL_KEYS} for key in METRIC_KEYS},
        "original_correct_count": 0,
        "repaired_correct_count": 0,
        "shortening_percentages": [],
    }


def _group_keys(record: dict) -> list[str]:
    source = record["case"]["source"]
    l_value = record["slice"]["l_value"]
    return ["overall", f"source:{source}", f"L:{l_value}", f"source:{source}|L:{l_value}"]


def build_summary(records: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for record in records:
        keys = _group_keys(record)
        for key in keys:
            groups.setdefault(key, _init_group())
            group = groups[key]
            group["slice_count"] += 1

            local_judge = record.get("local_judge")
            if local_judge and local_judge.get("parsed_ok"):
                group["local_judge_count"] += 1
                for metric_key in METRIC_KEYS:
                    label = local_judge["metrics"][metric_key]["label"]
                    group["metric_totals"][metric_key][label] += 1

            original = record.get("original_continuation")
            repaired = record.get("repaired_continuation")
            if original and repaired:
                group["continuation_pair_count"] += 1
                if original.get("correct"):
                    group["original_correct_count"] += 1
                if repaired.get("correct"):
                    group["repaired_correct_count"] += 1
                shortening_ratio = record.get("shortening_ratio")
                if shortening_ratio is not None:
                    group["both_correct_count"] += 1
                    group["shortening_percentages"].append(shortening_ratio * 100.0)

    summary_groups = {}
    for key, group in groups.items():
        metric_rates = {}
        for metric_key in METRIC_KEYS:
            total = sum(group["metric_totals"][metric_key][label] for label in LABEL_KEYS)
            metric_rates[metric_key] = {
                "T": group["metric_totals"][metric_key]["T"],
                "M": group["metric_totals"][metric_key]["M"],
                "F": group["metric_totals"][metric_key]["F"],
                "t_rate": (group["metric_totals"][metric_key]["T"] / total) if total else None,
                "m_rate": (group["metric_totals"][metric_key]["M"] / total) if total else None,
                "f_rate": (group["metric_totals"][metric_key]["F"] / total) if total else None,
            }

        continuation_pair_count = group["continuation_pair_count"]
        summary_groups[key] = {
            "slice_count": group["slice_count"],
            "local_judge_count": group["local_judge_count"],
            "continuation_pair_count": continuation_pair_count,
            "metric_rates": metric_rates,
            "original_correct_rate": (
                group["original_correct_count"] / continuation_pair_count if continuation_pair_count else None
            ),
            "repaired_correct_rate": (
                group["repaired_correct_count"] / continuation_pair_count if continuation_pair_count else None
            ),
            "both_correct_count": group["both_correct_count"],
            "avg_shortening_percentage": (
                mean(group["shortening_percentages"]) if group["shortening_percentages"] else None
            ),
        }
    return {"groups": summary_groups}


def summary_to_markdown(summary: dict) -> str:
    lines = ["# Experiment Summary", ""]
    groups = summary.get("groups", {})
    for group_key in sorted(groups):
        group = groups[group_key]
        lines.append(f"## {group_key}")
        lines.append(f"- slice_count: {group['slice_count']}")
        lines.append(f"- local_judge_count: {group['local_judge_count']}")
        lines.append(f"- continuation_pair_count: {group['continuation_pair_count']}")
        lines.append(f"- original_correct_rate: {group['original_correct_rate']}")
        lines.append(f"- repaired_correct_rate: {group['repaired_correct_rate']}")
        lines.append(f"- both_correct_count: {group['both_correct_count']}")
        lines.append(f"- avg_shortening_percentage: {group['avg_shortening_percentage']}")
        lines.append("- local_metric_rates:")
        for metric_key in METRIC_KEYS:
            rates = group["metric_rates"][metric_key]
            lines.append(
                f"  - {metric_key}: T={rates['T']} ({rates['t_rate']}), "
                f"M={rates['M']} ({rates['m_rate']}), F={rates['F']} ({rates['f_rate']})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
