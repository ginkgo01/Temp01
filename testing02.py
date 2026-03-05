import json
import os
import random
from urllib import error, request

from datasets import load_from_disk
from answer_judge import is_correct, llm_fallback_equivalence_judge


BASE_URL = "http://localhost:6002/v1"
API_KEY = "1e174CY6rKs28HcNjxhv"
MODEL = "model/Qwen3-8B"
DATASET_PATH = "/mnt/common/lx/Temp01/math500_hf"
PROMPT_PATH = "/mnt/common/lx/Temp01/solver_prompt.txt"
SAMPLE_SIZE = 10
SEED = 43
MAX_TOKENS = 12000
TESTED_PATH = "/mnt/common/lx/Temp01/testing02_tested.json"
REPORT_PATH = "/mnt/common/lx/Temp01/testing02_report.json"
JUDGE_MAX_TOKENS = 2560


def call_chat_completion(base_url: str, api_key: str, payload: dict, timeout: float = 600.0) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def build_question_key(dataset_idx: int, unique_id: str | None) -> str:
    if unique_id is not None and str(unique_id).strip():
        return f"uid:{unique_id}"
    return f"idx:{dataset_idx}"


def load_tested_records(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records: dict[str, dict] = {}
        # 兼容旧格式：["uid:xxx", "idx:1", ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    key = item.strip()
                    records[key] = {"key": key, "status": "failed"}
            return records

        # 新格式：{"records": [{"key": "...", "status": "success|failed"}, ...]}
        if isinstance(data, dict):
            items = data.get("records", [])
            if not isinstance(items, list):
                return {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                status = str(item.get("status", "")).strip().lower()
                if not key:
                    continue
                if status not in {"success", "failed"}:
                    status = "failed"
                records[key] = {"key": key, "status": status}
            return records
        return {}
    except Exception:
        return {}


def save_tested_records(path: str, records: dict[str, dict]) -> None:
    payload = {"records": [records[k] for k in sorted(records)]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_report_runs(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("runs"), list):
            return [item for item in data["runs"] if isinstance(item, dict)]
        # 兼容旧格式：单次运行报告对象
        if "sample_size" in data and "wrong_cases" in data:
            return [data]
    return []


def append_report_run(path: str, run_report: dict) -> None:
    runs = load_report_runs(path)
    runs.append(run_report)
    payload = {"total_runs": len(runs), "runs": runs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("初始化 LLM...")
    print(f"BASE_URL={BASE_URL}")
    print(f"MODEL={MODEL}")
    print(f"随机测试题数={SAMPLE_SIZE}, SEED={SEED}")

    print("加载数据集...")
    ds = load_from_disk(DATASET_PATH)
    dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
    tested_records = load_tested_records(TESTED_PATH)
    tested_keys = set(tested_records.keys())
    if not os.path.exists(TESTED_PATH):
        save_tested_records(TESTED_PATH, tested_records)

    untested_indices = []
    for i in range(len(dataset)):
        row = dataset[i]
        key = build_question_key(i, row.get("unique_id"))
        if key not in tested_keys:
            untested_indices.append(i)

    sample_size = min(SAMPLE_SIZE, len(untested_indices))
    print(f"历史已测题数={len(tested_keys)}")
    print(f"本次可测未测题数={len(untested_indices)}")
    print(f"本次实际测试题数={sample_size}")
    if sample_size == 0:
        print("没有可测试的新题，结束。")
        return

    rng = random.Random(SEED)
    sampled_indices = rng.sample(untested_indices, k=sample_size)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    wrong_cases = []
    correct_count = 0

    for order, idx in enumerate(sampled_indices, start=1):
        row = dataset[idx]
        problem = row["problem"]
        gold_answer = row["answer"]
        question_key = build_question_key(idx, row.get("unique_id"))
        prompt = prompt_template.replace("{{problem}}", problem)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "repetition_penalty": 1.1,
        }

        print(f"\n[{order}/{sample_size}] 测试题 idx={idx}, unique_id={row.get('unique_id')}")
        tested_status = "failed"
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
            if ok:
                tested_status = "success"
                correct_count += 1
                print(f"  结果: 正确 ({reason})")
            else:
                print(f"  结果: 错误 ({reason})")
                wrong_cases.append(
                    {
                        "dataset_idx": idx,
                        "unique_id": row.get("unique_id"),
                        "pred_boxed": pred_boxed,
                        "gold_answer": gold_answer,
                        "reason": reason,
                        "model_output": output,
                    }
                )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            msg = f"HTTP 错误 {exc.code}: {detail}"
            print(f"  结果: 错误 ({msg})")
            wrong_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": row.get("unique_id"),
                    "pred_boxed": "",
                    "gold_answer": gold_answer,
                    "reason": msg,
                    "model_output": "",
                }
            )
        except error.URLError as exc:
            msg = f"网络错误: {exc.reason}"
            print(f"  结果: 错误 ({msg})")
            wrong_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": row.get("unique_id"),
                    "pred_boxed": "",
                    "gold_answer": gold_answer,
                    "reason": msg,
                    "model_output": "",
                }
            )
        except Exception as exc:
            msg = f"未知错误: {exc}"
            print(f"  结果: 错误 ({msg})")
            wrong_cases.append(
                {
                    "dataset_idx": idx,
                    "unique_id": row.get("unique_id"),
                    "pred_boxed": "",
                    "gold_answer": gold_answer,
                    "reason": msg,
                    "model_output": "",
                }
            )
        finally:
            # 无论对错/异常，题目都记为已测试，避免重复测试
            tested_records[question_key] = {"key": question_key, "status": tested_status}
            tested_keys.add(question_key)
            save_tested_records(TESTED_PATH, tested_records)

    print("\n===== 测试汇总 =====")
    print(f"总题数: {sample_size}")
    print(f"答对: {correct_count}")
    print(f"答错: {len(wrong_cases)}")
    print(f"准确率: {correct_count / sample_size:.2%}")

    if wrong_cases:
        print("\n错误题号与做错情况:")
        for i, case in enumerate(wrong_cases, start=1):
            print(f"\n#{i} dataset_idx={case['dataset_idx']}, unique_id={case['unique_id']}")
            print(f"  预测boxed: {case['pred_boxed']}")
            print(f"  标准答案: {case['gold_answer']}")
            print(f"  原因: {case['reason']}")

    run_report = {
        "sample_size": sample_size,
        "seed": SEED,
        "correct": correct_count,
        "wrong": len(wrong_cases),
        "wrong_cases": wrong_cases,
    }
    append_report_run(REPORT_PATH, run_report)
    print(f"\n详细结果已保存(追加): {REPORT_PATH}")
    print(f"已测题列表已更新: {TESTED_PATH}")


if __name__ == "__main__":
    main()
