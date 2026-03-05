import argparse
import json
import re
from urllib import error, request

from datasets import load_from_disk


BASE_URL = "http://localhost:6002/v1"
API_KEY = "1e174CY6rKs28HcNjxhv"
MODEL = "model/Qwen3-8B"

INPUT_REPORT_PATH = "/mnt/common/lx/Temp01/testing02_report.json"
PROMPT_PATH_ANSWER = "/mnt/common/lx/Temp01/verify_with_answer_prompt.txt"
PROMPT_PATH_REASONING = "/mnt/common/lx/Temp01/verify_with_reasoning_prompt.txt"
OUTPUT_PATH_ANSWER = "/mnt/common/lx/Temp01/testing06_single_output_answer.json"
OUTPUT_PATH_REASONING = "/mnt/common/lx/Temp01/testing06_single_output_reasoning.json"
DATASET_PATH = "/mnt/common/lx/Temp01/math500_hf"

FAILED_OUTPUT_TOKEN_LIMIT = 6000
MAX_TOKENS = 20000


def call_chat_completion(base_url: str, api_key: str, payload: dict, timeout: float = 300.0) -> str:
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


def truncate_with_tiktoken(text: str, limit: int) -> tuple[str, bool, int | None]:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return text, False, None

    try:
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= limit:
            return text, False, len(toks)
        truncated = enc.decode(toks[:limit])
        return truncated, True, len(toks)
    except Exception:
        return text, False, None


def truncate_with_regex_tokens(text: str, limit: int) -> tuple[str, bool, int]:
    pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)
    spans = list(pattern.finditer(text))
    total = len(spans)
    if total <= limit:
        return text, False, total

    cut_pos = spans[limit - 1].end()
    return text[:cut_pos], True, total


def truncate_failed_output(text: str, limit: int) -> tuple[str, bool, str, int | None]:
    truncated, is_cut, total = truncate_with_tiktoken(text, limit)
    if total is not None:
        return truncated, is_cut, "tiktoken", total

    truncated, is_cut, total = truncate_with_regex_tokens(text, limit)
    return truncated, is_cut, "regex_approx", total


def find_case(wrong_cases: list[dict], dataset_idx: int | None, unique_id: str | None) -> dict:
    if dataset_idx is not None:
        for case in wrong_cases:
            if case.get("dataset_idx") == dataset_idx:
                return case
        raise ValueError(f"在 report 中未找到 dataset_idx={dataset_idx} 的 wrong_case")

    if unique_id is not None:
        for case in wrong_cases:
            if case.get("unique_id") == unique_id:
                return case
        raise ValueError(f"在 report 中未找到 unique_id={unique_id} 的 wrong_case")

    raise ValueError("必须提供 dataset_idx 或 unique_id")


def find_dataset_row(dataset, dataset_idx: int | None, unique_id: str | None):
    if dataset_idx is not None and 0 <= dataset_idx < len(dataset):
        return dataset[dataset_idx], dataset_idx

    if unique_id is not None:
        for i in range(len(dataset)):
            row = dataset[i]
            if row.get("unique_id") == unique_id:
                return row, i
        raise ValueError(f"在 dataset 中未找到 unique_id={unique_id}")

    raise ValueError("无法定位 dataset 行")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单题测试：题目 + 失败解 + 标准答案 -> 定位错误起点并给出正确解"
    )
    parser.add_argument("--dataset-idx", type=int, default=None, help="按 dataset_idx 指定题目")
    parser.add_argument("--unique-id", type=str, default=None, help="按 unique_id 指定题目")
    parser.add_argument(
        "--verify-mode",
        type=str,
        choices=["answer", "reasoning"],
        default="answer",
        help="选择验证模式：answer(仅答案) / reasoning(答案+标准解法)",
    )
    parser.add_argument("--report-path", type=str, default=INPUT_REPORT_PATH, help="错误报告 JSON 路径")
    parser.add_argument("--prompt-path", type=str, default=None, help="Prompt 模板路径（可覆盖 verify-mode）")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="HF 数据集路径")
    parser.add_argument("--output-path", type=str, default=None, help="输出 JSON 路径（可覆盖 verify-mode）")
    parser.add_argument("--failed-limit", type=int, default=FAILED_OUTPUT_TOKEN_LIMIT, help="失败解 token 截断上限")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="LLM max_tokens")
    args = parser.parse_args()

    if (args.dataset_idx is None) == (args.unique_id is None):
        raise ValueError("请且仅请提供一个定位参数：--dataset-idx 或 --unique-id")

    prompt_path_default = PROMPT_PATH_REASONING if args.verify_mode == "reasoning" else PROMPT_PATH_ANSWER
    output_path_default = OUTPUT_PATH_REASONING if args.verify_mode == "reasoning" else OUTPUT_PATH_ANSWER
    prompt_path = args.prompt_path or prompt_path_default
    output_path = args.output_path or output_path_default

    with open(args.report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    wrong_cases = report.get("wrong_cases", [])
    case = find_case(wrong_cases, args.dataset_idx, args.unique_id)

    failed_output = case.get("model_output", "")
    clipped_output, is_cut, trunc_method, total_tokens = truncate_failed_output(failed_output, args.failed_limit)

    ds = load_from_disk(args.dataset_path)
    dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
    row, resolved_idx = find_dataset_row(dataset, case.get("dataset_idx"), case.get("unique_id"))
    problem = row.get("problem", "")
    gold_answer = row.get("answer", "")
    gold_solution = row.get("solution", "")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = (
        prompt_template.replace("{{problem}}", str(problem))
        .replace("{{gold_answer}}", str(gold_answer))
        .replace("{{gold_solution}}", str(gold_solution))
        .replace("{{failed_attempt_output}}", clipped_output)
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
        "stream": False,
    }

    print(f"开始单题 verify: dataset_idx={resolved_idx}, unique_id={case.get('unique_id')}")
    try:
        verify_output = call_chat_completion(BASE_URL, API_KEY, payload)
        out = {
            "report_path": args.report_path,
            "verify_mode": args.verify_mode,
            "prompt_path": prompt_path,
            "dataset_path": args.dataset_path,
            "dataset_idx": resolved_idx,
            "unique_id": case.get("unique_id"),
            "problem": problem,
            "gold_answer_from_dataset": gold_answer,
            "gold_solution_from_dataset": gold_solution,
            "gold_answer_from_report": case.get("gold_answer"),
            "pred_boxed_prev": case.get("pred_boxed"),
            "truncate_method": trunc_method,
            "failed_output_tokens_before_truncate": total_tokens,
            "failed_output_truncated": is_cut,
            "failed_output_token_limit": args.failed_limit,
            "model": MODEL,
            "max_tokens": args.max_tokens,
            "verify_output": verify_output,
        }
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        out = {
            "dataset_idx": resolved_idx,
            "unique_id": case.get("unique_id"),
            "error": f"HTTP 错误 {exc.code}: {detail}",
        }
    except error.URLError as exc:
        out = {
            "dataset_idx": resolved_idx,
            "unique_id": case.get("unique_id"),
            "error": f"网络错误: {exc.reason}",
        }
    except Exception as exc:
        out = {
            "dataset_idx": resolved_idx,
            "unique_id": case.get("unique_id"),
            "error": f"未知错误: {exc}",
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已写入: {output_path}")


if __name__ == "__main__":
    main()
