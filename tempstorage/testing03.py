import json
import re
from urllib import error, request


BASE_URL = "http://localhost:6002/v1"
API_KEY = "1e174CY6rKs28HcNjxhv"
MODEL = "model/Qwen3-8B"

INPUT_REPORT_PATH = "/mnt/common/lx/Temp01/testing02_report.json"
PROMPT_PATH = "/mnt/common/lx/Temp01/verifier_prompt.txt"
OUTPUT_REPORT_PATH = "/mnt/common/lx/Temp01/testing03_report.json"
DATASET_PATH = "/mnt/common/lx/Temp01/math500_hf"

FAILED_OUTPUT_TOKEN_LIMIT = 5000
MAX_TOKENS = 10000


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
    # 近似 token：英文词块 / 数字块 / 单个 CJK 字符 / 单个符号
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
    with open(INPUT_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    wrong_cases = report.get("wrong_cases", [])
    problem_map: dict[int, str] = {}
    try:
        from datasets import load_from_disk

        ds = load_from_disk(DATASET_PATH)
        dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
        for case in wrong_cases:
            idx = case.get("dataset_idx")
            if isinstance(idx, int) and 0 <= idx < len(dataset):
                problem_map[idx] = dataset[idx].get("problem", "")
    except Exception:
        pass
    verifier_results = []

    print(f"读取错题数: {len(wrong_cases)}")
    for i, case in enumerate(wrong_cases, start=1):
        idx = case.get("dataset_idx")
        problem = case.get("problem") or problem_map.get(idx) or f"[unique_id={case.get('unique_id')}]"
        failed_output = case.get("model_output", "")

        clipped_output, is_cut, trunc_method, total_tokens = truncate_failed_output(
            failed_output, FAILED_OUTPUT_TOKEN_LIMIT
        )

        prompt = (
            prompt_template.replace("{{problem}}", str(problem))
            .replace("{{failed_attempt_output}}", clipped_output)
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

        print(f"[{i}/{len(wrong_cases)}] 复盘 unique_id={case.get('unique_id')}")
        try:
            verifier_output = call_chat_completion(BASE_URL, API_KEY, payload)
            verifier_results.append(
                {
                    "dataset_idx": case.get("dataset_idx"),
                    "unique_id": case.get("unique_id"),
                    "problem_loaded": bool(problem_map.get(case.get("dataset_idx"))),
                    "gold_answer": case.get("gold_answer"),
                    "pred_boxed": case.get("pred_boxed"),
                    "truncate_method": trunc_method,
                    "failed_output_tokens_before_truncate": total_tokens,
                    "failed_output_truncated": is_cut,
                    "verifier_output": verifier_output,
                }
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            verifier_results.append(
                {
                    "dataset_idx": case.get("dataset_idx"),
                    "unique_id": case.get("unique_id"),
                    "problem_loaded": bool(problem_map.get(case.get("dataset_idx"))),
                    "truncate_method": trunc_method,
                    "failed_output_tokens_before_truncate": total_tokens,
                    "failed_output_truncated": is_cut,
                    "error": f"HTTP 错误 {exc.code}: {detail}",
                }
            )
        except error.URLError as exc:
            verifier_results.append(
                {
                    "dataset_idx": case.get("dataset_idx"),
                    "unique_id": case.get("unique_id"),
                    "problem_loaded": bool(problem_map.get(case.get("dataset_idx"))),
                    "truncate_method": trunc_method,
                    "failed_output_tokens_before_truncate": total_tokens,
                    "failed_output_truncated": is_cut,
                    "error": f"网络错误: {exc.reason}",
                }
            )
        except Exception as exc:
            verifier_results.append(
                {
                    "dataset_idx": case.get("dataset_idx"),
                    "unique_id": case.get("unique_id"),
                    "problem_loaded": bool(problem_map.get(case.get("dataset_idx"))),
                    "truncate_method": trunc_method,
                    "failed_output_tokens_before_truncate": total_tokens,
                    "failed_output_truncated": is_cut,
                    "error": f"未知错误: {exc}",
                }
            )

    out = {
        "input_report": INPUT_REPORT_PATH,
        "prompt_path": PROMPT_PATH,
        "failed_output_token_limit": FAILED_OUTPUT_TOKEN_LIMIT,
        "count": len(verifier_results),
        "results": verifier_results,
    }
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"已写入: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
