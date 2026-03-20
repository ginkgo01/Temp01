import argparse
import json
import os
from urllib import error

from datasets import load_from_disk

from testing02 import (
    API_KEY,
    BASE_URL,
    DATASET_PATH,
    JUDGE_MAX_TOKENS,
    MAX_TOKENS,
    MODEL,
    PROMPT_PATH,
    call_chat_completion,
)
from answer_judge import is_correct, llm_fallback_equivalence_judge


SOURCE_REPORT_PATH = "/mnt/common/lx/Temp01/testing02_report.json"
OUTPUT_PATH = "/mnt/common/lx/Temp01/testing02_wrong_batch_report.json"
REPEAT_N = 5


def build_temperature_schedule(repeat_n: int) -> list[float]:
    if repeat_n <= 1:
        return [0.2]
    step = (1.0 - 0.2) / (repeat_n - 1)
    return [round(0.2 + i * step, 4) for i in range(repeat_n)]


def run_retry_attempts(
    *,
    prompt: str,
    gold_answer: str,
    repeat_n: int,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = 1.1,
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
) -> dict:
    temperature_schedule = build_temperature_schedule(repeat_n)
    attempts = []
    success_count = 0

    for k in range(1, repeat_n + 1):
        temperature = temperature_schedule[k - 1]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "repetition_penalty": repetition_penalty,
        }
        try:
            output = call_chat_completion(base_url, api_key, payload)
            ok, reason, pred_boxed = is_correct(output, gold_answer)
            if (not ok) and reason.startswith("不匹配:") and pred_boxed:
                try:
                    judge_ok, judge_raw = llm_fallback_equivalence_judge(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        pred_boxed=pred_boxed,
                        gold_answer=gold_answer,
                        call_chat_completion_fn=call_chat_completion,
                        judge_max_tokens=judge_max_tokens,
                    )
                    if judge_ok:
                        ok = True
                        reason = "LLM 兜底判定等价"
                    else:
                        reason = f"{reason}; LLM兜底={judge_raw}"
                except Exception as judge_exc:
                    reason = f"{reason}; LLM兜底异常={judge_exc}"
            if ok:
                success_count += 1
            attempts.append(
                {
                    "attempt": k,
                    "temperature": temperature,
                    "ok": ok,
                    "reason": reason,
                    "pred_boxed": pred_boxed,
                    "model_output": output,
                }
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            msg = f"HTTP 错误 {exc.code}: {detail}"
            attempts.append(
                {
                    "attempt": k,
                    "temperature": temperature,
                    "ok": False,
                    "reason": msg,
                    "pred_boxed": "",
                    "model_output": "",
                }
            )
        except error.URLError as exc:
            msg = f"网络错误: {exc.reason}"
            attempts.append(
                {
                    "attempt": k,
                    "temperature": temperature,
                    "ok": False,
                    "reason": msg,
                    "pred_boxed": "",
                    "model_output": "",
                }
            )
        except Exception as exc:
            msg = f"未知错误: {exc}"
            attempts.append(
                {
                    "attempt": k,
                    "temperature": temperature,
                    "ok": False,
                    "reason": msg,
                    "pred_boxed": "",
                    "model_output": "",
                }
            )

    return {
        "repeat_n": repeat_n,
        "temperature_schedule": temperature_schedule,
        "success_count": success_count,
        "solved_in_batch": success_count > 0,
        "attempts": attempts,
    }


def build_output_path(base_path: str, question_idx: int | None) -> str:
    if question_idx is None:
        return base_path
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".json"
    return f"{root}_{question_idx}{ext}"


def load_wrong_cases(report_path: str) -> list[dict]:
    if not os.path.exists(report_path):
        return []
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs: list[dict] = []
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        runs = [x for x in data["runs"] if isinstance(x, dict)]
    elif isinstance(data, dict):
        runs = [data]
    elif isinstance(data, list):
        runs = [x for x in data if isinstance(x, dict)]

    merged: list[dict] = []
    seen: set[str] = set()
    for run in runs:
        for case in run.get("wrong_cases", []):
            if not isinstance(case, dict):
                continue
            key = str(case.get("unique_id") or f"idx:{case.get('dataset_idx')}")
            if key in seen:
                continue
            seen.add(key)
            merged.append(case)
    return merged


def find_question_row(dataset, case: dict) -> tuple[int, dict] | tuple[None, None]:
    unique_id = case.get("unique_id")
    dataset_idx = case.get("dataset_idx")

    if isinstance(dataset_idx, int) and 0 <= dataset_idx < len(dataset):
        row = dataset[dataset_idx]
        if unique_id is None or row.get("unique_id") == unique_id:
            return dataset_idx, row

    if unique_id is not None:
        for i in range(len(dataset)):
            row = dataset[i]
            if row.get("unique_id") == unique_id:
                return i, row

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="复测历史错题（支持按题号过滤）")
    parser.add_argument(
        "-n",
        "--repeat-n",
        type=int,
        default=REPEAT_N,
        help=f"每题重复测试次数（默认 {REPEAT_N}）",
    )
    parser.add_argument(
        "-q",
        "--question-idx",
        type=int,
        default=None,
        help="仅复测指定 dataset_idx 题号",
    )
    args = parser.parse_args()
    if args.repeat_n < 1:
        raise ValueError("--repeat-n 必须 >= 1")

    print("加载错题来源报告...")
    wrong_cases = load_wrong_cases(SOURCE_REPORT_PATH)
    if not wrong_cases:
        print("未找到错题，结束。")
        return

    if args.question_idx is not None:
        wrong_cases = [c for c in wrong_cases if c.get("dataset_idx") == args.question_idx]
        if not wrong_cases:
            print(f"在错题列表中未找到 dataset_idx={args.question_idx}，结束。")
            return

    repeat_n = args.repeat_n
    output_path = build_output_path(OUTPUT_PATH, args.question_idx)
    temperature_schedule = build_temperature_schedule(repeat_n)
    print(f"错题数: {len(wrong_cases)}，每题重复测试次数: {repeat_n}")
    print(f"温度序列: {temperature_schedule}")
    print(f"输出路径: {output_path}")
    print("加载数据集与提示词...")
    ds = load_from_disk(DATASET_PATH)
    dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    question_reports = []
    total_attempts = 0
    total_success = 0

    for i, case in enumerate(wrong_cases, start=1):
        idx, row = find_question_row(dataset, case)
        if row is None:
            question_reports.append(
                {
                    "dataset_idx": case.get("dataset_idx"),
                    "unique_id": case.get("unique_id"),
                    "problem": "",
                    "solution": "",
                    "gold_answer": case.get("gold_answer", ""),
                    "error": "无法在数据集中定位该题",
                    "attempts": [],
                    "success_count": 0,
                }
            )
            continue

        problem = row["problem"]
        gold_answer = row["answer"]
        unique_id = row.get("unique_id")
        prompt = prompt_template.replace("{{problem}}", problem)
        print(f"\n[{i}/{len(wrong_cases)}] 复测 idx={idx}, unique_id={unique_id}")

        attempts = []
        success_count = 0
        retry_result = run_retry_attempts(prompt=prompt, gold_answer=gold_answer, repeat_n=repeat_n)
        attempts = retry_result["attempts"]
        success_count = retry_result["success_count"]
        for item in attempts:
            total_attempts += 1
            if item.get("ok"):
                total_success += 1
            print(
                f"  - 第{item['attempt']}次(temperature={item['temperature']}): "
                f"{'正确' if item.get('ok') else '错误'} ({item.get('reason')})"
            )

        question_reports.append(
            {
                "dataset_idx": idx,
                "unique_id": unique_id,
                "problem": problem,
                "solution": row.get("solution", ""),
                "gold_answer": gold_answer,
                "repeat_n": repeat_n,
                "success_count": success_count,
                "solved_in_batch": success_count > 0,
                "attempts": attempts,
            }
        )

    output_payload = {
        "source_report_path": SOURCE_REPORT_PATH,
        "repeat_n": repeat_n,
        "temperature_schedule": temperature_schedule,
        "question_idx_filter": args.question_idx,
        "question_count": len(question_reports),
        "total_attempts": total_attempts,
        "total_success": total_success,
        "overall_accuracy": (total_success / total_attempts) if total_attempts else 0.0,
        "question_reports": question_reports,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    solved_questions = sum(1 for q in question_reports if q.get("solved_in_batch"))
    print("\n===== 批量复测汇总 =====")
    print(f"错题数: {len(question_reports)}")
    print(f"每题重复次数: {repeat_n}")
    print(f"总尝试次数: {total_attempts}")
    print(f"总答对次数: {total_success}")
    print(f"总体准确率: {output_payload['overall_accuracy']:.2%}")
    print(f"至少做对过1次的题数: {solved_questions}")
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
