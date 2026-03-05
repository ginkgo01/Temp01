import argparse
import json
import re
from urllib import error, request


BASE_URL = "http://localhost:6002/v1"
API_KEY = "1e174CY6rKs28HcNjxhv"
MODEL = "model/Qwen3-8B"

INPUT_REPORT_PATH = "/mnt/common/lx/Temp01/report_failed.json"
PROMPT_PATH = "/mnt/common/lx/Temp01/verifier_prompt.txt"
OUTPUT_REPORT_DIR = "/mnt/common/lx/Temp01"
FAILED_OUTPUT_TOKEN_LIMIT = 5000
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


def main() -> None:
    parser = argparse.ArgumentParser(description="对 report_failed 中指定题号做单题复盘")
    parser.add_argument("dataset_idx", type=int, help="题号（dataset_idx）")
    args = parser.parse_args()
    target_dataset_idx = args.dataset_idx
    output_report_path = f"{OUTPUT_REPORT_DIR}/verify_result_{target_dataset_idx}.json"

    with open(INPUT_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    wrong_cases = report.get("cases", [])
    target_case = None
    for case in wrong_cases:
        if case.get("dataset_idx") == target_dataset_idx:
            target_case = case
            break

    if target_case is None:
        out = {
            "input_report": INPUT_REPORT_PATH,
            "target_dataset_idx": target_dataset_idx,
            "error": "在 report_failed.cases 中未找到目标 dataset_idx",
        }
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"未找到 dataset_idx={target_dataset_idx}，已写入: {output_report_path}")
        return

    problem = target_case.get("problem", "")
    if not problem:
        problem = f"[unique_id={target_case.get('unique_id')}]"

    # 优先使用主流程失败时的第一次回答；若为空则回退到 retry 的第一次尝试
    failed_output = target_case.get("model_output", "")
    if not failed_output:
        retry_attempts = target_case.get("retry_result", {}).get("attempts", [])
        if isinstance(retry_attempts, list) and retry_attempts:
            failed_output = str(retry_attempts[0].get("model_output", ""))

    clipped_output, is_cut, trunc_method, total_tokens = truncate_failed_output(
        failed_output, FAILED_OUTPUT_TOKEN_LIMIT
    )

    prompt = (
        prompt_template.replace("{{problem}}", str(problem)).replace("{{failed_attempt_output}}", clipped_output)
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    print("\n========== PROMPT BEGIN ==========")
    print(prompt)
    print("=========== PROMPT END ===========\n")

    result = {
        "input_report": INPUT_REPORT_PATH,
        "prompt_path": PROMPT_PATH,
        "target_dataset_idx": target_dataset_idx,
        "failed_output_token_limit": FAILED_OUTPUT_TOKEN_LIMIT,
        "dataset_idx": target_case.get("dataset_idx"),
        "unique_id": target_case.get("unique_id"),
        "gold_answer": target_case.get("gold_answer"),
        "pred_boxed": target_case.get("pred_boxed"),
        "truncate_method": trunc_method,
        "failed_output_tokens_before_truncate": total_tokens,
        "failed_output_truncated": is_cut,
    }

    print(f"开始复盘 dataset_idx={target_dataset_idx}, unique_id={target_case.get('unique_id')}")
    try:
        verifier_output = call_chat_completion(BASE_URL, API_KEY, payload)
        result["verifier_output"] = verifier_output
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTP 错误 {exc.code}: {detail}"
    except error.URLError as exc:
        result["error"] = f"网络错误: {exc.reason}"
    except Exception as exc:
        result["error"] = f"未知错误: {exc}"

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已写入: {output_report_path}")


if __name__ == "__main__":
    main()
