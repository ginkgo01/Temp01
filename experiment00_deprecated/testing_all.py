import json
from datetime import datetime, timezone
from urllib import error

from datasets import load_from_disk

from answer_judge import is_correct, llm_fallback_equivalence_judge
from testing02_wrong_batch import run_retry_attempts
from testing02 import (
    API_KEY,
    BASE_URL,
    DATASET_PATH,
    JUDGE_MAX_TOKENS,
    MAX_TOKENS,
    MODEL,
    PROMPT_PATH,
    TESTED_PATH,
    call_chat_completion,
)


SUCCESS_REPORT_PATH = "/mnt/common/lx/Temp01/report_success.json"
FAILED_REPORT_PATH = "/mnt/common/lx/Temp01/report_failed.json"
TARGET_COUNT = 500
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.1
RETRY_REPEAT_N = 5


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clear_tested(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": []}, f, ensure_ascii=False, indent=2)


def save_tested_records(path: str, records: dict[str, dict]) -> None:
    payload = {"records": [records[k] for k in sorted(records)]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_question_key(dataset_idx: int, unique_id: str | None) -> str:
    if unique_id is not None and str(unique_id).strip():
        return f"uid:{unique_id}"
    return f"idx:{dataset_idx}"


def main() -> None:
    started_at = now_iso()
    print("开始全量测试，先清空 tested...")
    clear_tested(TESTED_PATH)

    print("加载数据集与提示词...")
    ds = load_from_disk(DATASET_PATH)
    dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
    total_in_dataset = len(dataset)
    target_count = min(TARGET_COUNT, total_in_dataset)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    metadata = {
        "started_at": started_at,
        "base_url": BASE_URL,
        "model": MODEL,
        "dataset_path": DATASET_PATH,
        "prompt_path": PROMPT_PATH,
        "target_count": target_count,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "repetition_penalty": REPETITION_PENALTY,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "retry_repeat_n": RETRY_REPEAT_N,
    }

    success_cases: list[dict] = []
    failed_cases: list[dict] = []
    tested_records: dict[str, dict] = {}

    for idx in range(target_count):
        row = dataset[idx]
        unique_id = row.get("unique_id")
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        gold_answer = row.get("answer", "")
        question_key = build_question_key(idx, unique_id)

        prompt = prompt_template.replace("{{problem}}", problem)
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "repetition_penalty": REPETITION_PENALTY,
        }

        print(f"[{idx + 1}/{target_count}] idx={idx}, unique_id={unique_id}")
        status = "failed"
        pred_boxed = ""
        reason = ""
        output = ""
        try:
            output = call_chat_completion(base_url=BASE_URL, api_key=API_KEY, payload=payload)
            ok, reason, pred_boxed = is_correct(output, gold_answer)
            if (not ok) and reason.startswith("不匹配:") and pred_boxed:
                try:
                    judge_ok, judge_raw = llm_fallback_equivalence_judge(
                        base_url=BASE_URL,
                        api_key=API_KEY,
                        model=MODEL,
                        pred_boxed=pred_boxed,
                        gold_answer=gold_answer,
                        call_chat_completion_fn=call_chat_completion,
                        judge_max_tokens=JUDGE_MAX_TOKENS,
                    )
                    if judge_ok:
                        ok = True
                        reason = "LLM 兜底判定等价"
                    else:
                        reason = f"{reason}; LLM兜底={judge_raw}"
                except Exception as judge_exc:
                    reason = f"{reason}; LLM兜底异常={judge_exc}"

            case_payload = {
                "dataset_idx": idx,
                "unique_id": unique_id,
                "problem": problem,
                "solution": solution,
                "gold_answer": gold_answer,
                "pred_boxed": pred_boxed,
                "reason": reason,
                "model_output": output,
            }
            if ok:
                status = "success"
                success_cases.append(case_payload)
                print(f"  结果: 正确 ({reason})")
            else:
                retry_result = run_retry_attempts(
                    prompt=prompt,
                    gold_answer=gold_answer,
                    repeat_n=RETRY_REPEAT_N,
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    repetition_penalty=REPETITION_PENALTY,
                    judge_max_tokens=JUDGE_MAX_TOKENS,
                )
                retry_successful = (
                    retry_result["success_count"] / retry_result["repeat_n"] >= 0.7
                    if retry_result["repeat_n"]
                    else False
                )
                case_payload["retry_result"] = retry_result
                case_payload["retry_successful"] = retry_successful
                failed_cases.append(case_payload)
                print(f"  结果: 错误 ({reason})")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            reason = f"HTTP 错误 {exc.code}: {detail}"
            retry_result = run_retry_attempts(
                prompt=prompt,
                gold_answer=gold_answer,
                repeat_n=RETRY_REPEAT_N,
                base_url=BASE_URL,
                api_key=API_KEY,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                judge_max_tokens=JUDGE_MAX_TOKENS,
            )
            retry_successful = (
                retry_result["success_count"] / retry_result["repeat_n"] >= 0.7
                if retry_result["repeat_n"]
                else False
            )
            failed_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": unique_id,
                    "problem": problem,
                    "solution": solution,
                    "gold_answer": gold_answer,
                    "pred_boxed": "",
                    "reason": reason,
                    "model_output": "",
                    "retry_result": retry_result,
                    "retry_successful": retry_successful,
                }
            )
            print(f"  结果: 错误 ({reason})")
        except error.URLError as exc:
            reason = f"网络错误: {exc.reason}"
            retry_result = run_retry_attempts(
                prompt=prompt,
                gold_answer=gold_answer,
                repeat_n=RETRY_REPEAT_N,
                base_url=BASE_URL,
                api_key=API_KEY,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                judge_max_tokens=JUDGE_MAX_TOKENS,
            )
            retry_successful = (
                retry_result["success_count"] / retry_result["repeat_n"] >= 0.7
                if retry_result["repeat_n"]
                else False
            )
            failed_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": unique_id,
                    "problem": problem,
                    "solution": solution,
                    "gold_answer": gold_answer,
                    "pred_boxed": "",
                    "reason": reason,
                    "model_output": "",
                    "retry_result": retry_result,
                    "retry_successful": retry_successful,
                }
            )
            print(f"  结果: 错误 ({reason})")
        except Exception as exc:
            reason = f"未知错误: {exc}"
            retry_result = run_retry_attempts(
                prompt=prompt,
                gold_answer=gold_answer,
                repeat_n=RETRY_REPEAT_N,
                base_url=BASE_URL,
                api_key=API_KEY,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                judge_max_tokens=JUDGE_MAX_TOKENS,
            )
            retry_successful = (
                retry_result["success_count"] / retry_result["repeat_n"] >= 0.7
                if retry_result["repeat_n"]
                else False
            )
            failed_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": unique_id,
                    "problem": problem,
                    "solution": solution,
                    "gold_answer": gold_answer,
                    "pred_boxed": "",
                    "reason": reason,
                    "model_output": "",
                    "retry_result": retry_result,
                    "retry_successful": retry_successful,
                }
            )
            print(f"  结果: 错误 ({reason})")
        finally:
            tested_records[question_key] = {"key": question_key, "status": status}
            save_tested_records(TESTED_PATH, tested_records)

    ended_at = now_iso()
    metadata.update(
        {
            "ended_at": ended_at,
            "total_in_dataset": total_in_dataset,
            "tested_count": target_count,
            "success_count": len(success_cases),
            "failed_count": len(failed_cases),
            "accuracy": (len(success_cases) / target_count) if target_count else 0.0,
        }
    )

    success_report = {"metadata": metadata, "cases": success_cases}
    failed_report = {"metadata": metadata, "cases": failed_cases}

    with open(SUCCESS_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(success_report, f, ensure_ascii=False, indent=2)
    with open(FAILED_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(failed_report, f, ensure_ascii=False, indent=2)

    print("\n===== 全量测试完成 =====")
    print(f"测试题数: {target_count}")
    print(f"成功数: {len(success_cases)}")
    print(f"失败数: {len(failed_cases)}")
    print(f"准确率: {metadata['accuracy']:.2%}")
    print(f"成功报告: {SUCCESS_REPORT_PATH}")
    print(f"失败报告: {FAILED_REPORT_PATH}")
    print(f"已测记录: {TESTED_PATH}")


if __name__ == "__main__":
    main()
