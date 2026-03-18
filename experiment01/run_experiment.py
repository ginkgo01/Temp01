#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DEFAULT_RANDOM_SEED,
    JUDGED_OUTPUT_DIR,
    L_VALUES,
    REPAIRED_OUTPUT_DIR,
    RUNS_DIR,
    SUMMARIES_DIR,
)
from continuation_runner import run_continuation
from cot_utils import build_slice, extract_reasoning_text, get_valid_cut_points, split_thought_units
from data_loader import load_cases
from io_utils import ensure_dir, save_json
from local_quality_judge import LocalJudgeUnavailableError, judge_local_quality
from repair_generator import generate_repaired_suffix
from report_builder import build_summary, summary_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment01 local CoT repair experiment.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["success", "failed"],
        default=["success", "failed"],
        help="Which report sources to include.",
    )
    parser.add_argument(
        "--l-values",
        nargs="+",
        type=int,
        default=list(L_VALUES),
        help="Thought unit lengths to test.",
    )
    parser.add_argument("--max-cases-per-source", type=int, default=None, help="Optional cap per source.")
    parser.add_argument("--max-cutpoints-per-case", type=int, default=None, help="Optional cutpoint cap per case.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle cases before selection.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--skip-continuation",
        action="store_true",
        help="Run repair and local judge only, skipping continuation stage.",
    )
    return parser.parse_args()


def shorten_ratio(original_units: int, repaired_units: int) -> float | None:
    if original_units <= 0:
        return None
    return (original_units - repaired_units) / original_units


def select_cases(args: argparse.Namespace) -> list[dict]:
    cases = load_cases(args.sources)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(cases)

    if args.max_cases_per_source is None:
        return cases

    picked: list[dict] = []
    counts = {source: 0 for source in args.sources}
    for case in cases:
        source = case["source"]
        if counts[source] >= args.max_cases_per_source:
            continue
        picked.append(case)
        counts[source] += 1
    return picked


def main() -> int:
    args = parse_args()
    ensure_dir(RUNS_DIR)
    ensure_dir(REPAIRED_OUTPUT_DIR)
    ensure_dir(JUDGED_OUTPUT_DIR)
    ensure_dir(SUMMARIES_DIR)

    cases = select_cases(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict] = []
    skipped_cases: list[dict] = []

    for case_order, case in enumerate(cases, start=1):
        reasoning_text = extract_reasoning_text(case["model_output"])
        reasoning_units = split_thought_units(reasoning_text)
        if not reasoning_units:
            skipped_cases.append(
                {
                    "dataset_idx": case["dataset_idx"],
                    "unique_id": case["unique_id"],
                    "source": case["source"],
                    "reason": "no_reasoning_units",
                }
            )
            continue

        print(
            f"[{case_order}/{len(cases)}] source={case['source']} idx={case['dataset_idx']} "
            f"uid={case['unique_id']}"
        )

        case_records: list[dict] = []
        case_failed = False
        case_failure_reason = ""

        for l_value in args.l_values:
            cut_points = get_valid_cut_points(reasoning_units, l_value)
            if args.max_cutpoints_per_case is not None:
                cut_points = cut_points[: args.max_cutpoints_per_case]

            for cut_idx in cut_points:
                sliced = build_slice(reasoning_units, cut_idx, l_value)
                repaired = generate_repaired_suffix(case, sliced["prefix_units"], l_value)

                repaired_record = {
                    "case_uid": case["unique_id"],
                    "dataset_idx": case["dataset_idx"],
                    "source": case["source"],
                    "l_value": l_value,
                    "cut_idx": cut_idx,
                    "original_prefix_text": sliced["prefix_text"],
                    "original_suffix_text": sliced["original_suffix_text"],
                    "original_prefix_units": sliced["prefix_units"],
                    "original_suffix_units": sliced["original_suffix_units"],
                    "repaired": repaired,
                }
                save_json(
                    REPAIRED_OUTPUT_DIR / f"{run_id}_{case['dataset_idx']}_{l_value}_{cut_idx}.json",
                    repaired_record,
                )

                try:
                    local_judge = judge_local_quality(
                        case=case,
                        prefix_text=sliced["prefix_text"],
                        original_suffix_text=sliced["original_suffix_text"],
                        repaired_suffix_text=repaired["repaired_text"],
                        l_value=l_value,
                    )
                except LocalJudgeUnavailableError as exc:
                    case_failed = True
                    case_failure_reason = f"local_judge_failed_after_retries: {exc}"
                    print(f"  跳过该题: {case_failure_reason}")
                    break

                save_json(
                    JUDGED_OUTPUT_DIR / f"{run_id}_{case['dataset_idx']}_{l_value}_{cut_idx}.json",
                    {
                        "case_uid": case["unique_id"],
                        "dataset_idx": case["dataset_idx"],
                        "source": case["source"],
                        "l_value": l_value,
                        "cut_idx": cut_idx,
                        "judge": local_judge,
                    },
                )

                original_continuation = None
                repaired_continuation = None
                ratio = None

                if not args.skip_continuation:
                    original_continuation = run_continuation(
                        case=case,
                        prefix_units=sliced["prefix_units"],
                        seed_units=sliced["original_suffix_units"],
                        seed_type="original",
                    )

                    if repaired["repaired_unit_count"] == l_value:
                        repaired_continuation = run_continuation(
                            case=case,
                            prefix_units=sliced["prefix_units"],
                            seed_units=repaired["repaired_units"],
                            seed_type="repaired",
                        )

                    if (
                        repaired_continuation is not None
                        and original_continuation["correct"]
                        and repaired_continuation["correct"]
                    ):
                        ratio = shorten_ratio(
                            original_continuation["continuation_unit_count"],
                            repaired_continuation["continuation_unit_count"],
                        )

                case_records.append(
                    {
                        "case": {
                            "source": case["source"],
                            "is_original_correct": case["is_original_correct"],
                            "dataset_idx": case["dataset_idx"],
                            "unique_id": case["unique_id"],
                            "problem": case["problem"],
                            "gold_answer": case["gold_answer"],
                            "original_reason": case["reason"],
                        },
                        "slice": {
                            "l_value": l_value,
                            "cut_idx": cut_idx,
                            "reasoning_unit_count": len(reasoning_units),
                            "prefix_unit_count": len(sliced["prefix_units"]),
                            "original_suffix_unit_count": len(sliced["original_suffix_units"]),
                            "prefix_text": sliced["prefix_text"],
                            "original_suffix_text": sliced["original_suffix_text"],
                        },
                        "repair": repaired,
                        "local_judge": local_judge,
                        "original_continuation": original_continuation,
                        "repaired_continuation": repaired_continuation,
                        "shortening_ratio": ratio,
                    }
                )

            if case_failed:
                break

        if case_failed:
            skipped_cases.append(
                {
                    "dataset_idx": case["dataset_idx"],
                    "unique_id": case["unique_id"],
                    "source": case["source"],
                    "reason": case_failure_reason,
                }
            )
            continue

        records.extend(case_records)

    summary = build_summary(records)
    run_report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            "sources": args.sources,
            "l_values": args.l_values,
            "max_cases_per_source": args.max_cases_per_source,
            "max_cutpoints_per_case": args.max_cutpoints_per_case,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "skip_continuation": args.skip_continuation,
        },
        "record_count": len(records),
        "skipped_cases": skipped_cases,
        "records": records,
        "summary": summary,
    }

    run_path = RUNS_DIR / f"experiment_run_{run_id}.json"
    save_json(run_path, run_report)
    summary_path = SUMMARIES_DIR / f"experiment_run_{run_id}.md"
    summary_path.write_text(summary_to_markdown(summary), encoding="utf-8")

    print(f"Saved detailed run report to: {run_path}")
    print(f"Saved markdown summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
