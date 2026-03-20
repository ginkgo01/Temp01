#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib import error, request

DEFAULT_BASE_URL = "http://localhost:6002/v1"
DEFAULT_API_KEY = "1e174CY6rKs28HcNjxhv"
DEFAULT_MODEL = "model/Qwen3-8B"


def load_settings_from_file(settings_path: Path) -> tuple[str, str, str]:
    base_url = DEFAULT_BASE_URL
    api_key = DEFAULT_API_KEY
    model = DEFAULT_MODEL

    if not settings_path.exists():
        return base_url, api_key, model

    try:
        spec = importlib.util.spec_from_file_location("llm_settings", str(settings_path))
        if spec is None or spec.loader is None:
            return base_url, api_key, model
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        settings_cls = getattr(module, "Settings", None)
        if settings_cls is not None:
            base_url = getattr(settings_cls, "LLM_LIGHT_BASE_URL", base_url)
            api_key = getattr(settings_cls, "LLM_LIGHT_API_KEY", api_key)
            model = getattr(settings_cls, "LLM_LIGHT_MODEL_NAME", model)
    except Exception:
        pass

    return base_url, api_key, model


def http_json(method: str, url: str, api_key: str, data: dict | None, timeout: float) -> tuple[int, dict]:
    raw = None
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw else {}
            return resp.status, parsed
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"raw": raw}
        return exc.code, parsed


def get_models_metadata(base_url: str, api_key: str, timeout: float) -> dict | None:
    status, data = http_json("GET", f"{base_url.rstrip('/')}/models", api_key=api_key, data=None, timeout=timeout)
    if status != 200:
        return None
    return data


def build_probe_prompt(n_words: int) -> str:
    # 英文短词通常接近 1 词 ~ 1 token，便于二分查找上下文上限。
    return "hello " * n_words


def chat_once(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[int, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return http_json("POST", f"{base_url.rstrip('/')}/chat/completions", api_key=api_key, data=payload, timeout=timeout)


def is_context_too_long(err_obj: dict) -> bool:
    text = json.dumps(err_obj, ensure_ascii=False).lower()
    keywords = [
        "maximum context length",
        "context length",
        "too many tokens",
        "token limit",
        "prompt is too long",
    ]
    return any(k in text for k in keywords)


def extract_usage_prompt_tokens(resp_obj: dict) -> int | None:
    usage = resp_obj.get("usage", {})
    val = usage.get("prompt_tokens")
    return val if isinstance(val, int) else None


def extract_context_hint(err_obj: dict) -> int | None:
    text = json.dumps(err_obj, ensure_ascii=False)
    # 常见报错里会包含类似 "maximum context length is 32768 tokens"
    m = re.search(r"(?i)maximum context length is\s+(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?i)context length[^0-9]{0,20}(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def probe_max_input_tokens(base_url: str, api_key: str, model: str, timeout: float, high_words: int) -> tuple[int | None, int | None]:
    lo, hi = 1, high_words
    best_prompt_tokens = None
    context_hint = None

    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = build_probe_prompt(mid)
        status, data = chat_once(base_url, api_key, model, prompt=prompt, max_tokens=1, timeout=timeout)
        if status == 200:
            pt = extract_usage_prompt_tokens(data)
            if isinstance(pt, int):
                best_prompt_tokens = max(best_prompt_tokens or 0, pt)
            lo = mid + 1
        else:
            if is_context_too_long(data):
                hint = extract_context_hint(data)
                if isinstance(hint, int):
                    context_hint = hint
                hi = mid - 1
            else:
                # 非上下文超限错误，直接结束探测
                break

    return best_prompt_tokens, context_hint


def probe_small_prompt_tokens(base_url: str, api_key: str, model: str, timeout: float) -> int | None:
    status, data = chat_once(base_url, api_key, model, prompt="ping", max_tokens=8, timeout=timeout)
    if status != 200:
        return None
    return extract_usage_prompt_tokens(data)


def pick_context_from_models(models_meta: dict, model_name: str) -> int | None:
    data = models_meta.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", ""))
        if model_name in item_id or item_id in model_name:
            for key in ("max_model_len", "context_length", "max_context_length", "max_input_tokens"):
                v = item.get(key)
                if isinstance(v, int):
                    return v
            # 有些服务会把字段塞在 extra 中
            extra = item.get("extra")
            if isinstance(extra, dict):
                for key in ("max_model_len", "context_length", "max_context_length", "max_input_tokens"):
                    v = extra.get(key)
                    if isinstance(v, int):
                        return v
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="探测 6002 端口模型的最大输入/输出 token")
    parser.add_argument("--settings", default="llm_settings.py", help="配置文件路径")
    parser.add_argument("--base-url", default=None, help="覆盖 base url")
    parser.add_argument("--api-key", default=None, help="覆盖 API key")
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--timeout", type=float, default=20.0, help="单次请求超时秒数")
    parser.add_argument("--high-words", type=int, default=70000, help="二分探测上界（词数）")
    args = parser.parse_args()

    cfg_base_url, cfg_api_key, cfg_model = load_settings_from_file(Path(args.settings))
    base_url = args.base_url or cfg_base_url
    api_key = args.api_key or cfg_api_key
    model = args.model or cfg_model

    print("=== Token 上限探测 ===")
    print(f"Base URL: {base_url}")
    print(f"Model   : {model}")

    models_meta = get_models_metadata(base_url, api_key, args.timeout)
    context_from_models = None
    if models_meta is not None:
        context_from_models = pick_context_from_models(models_meta, model)
        if context_from_models is not None:
            print(f"[models] 发现上下文上限: {context_from_models}")
        else:
            print("[models] 未返回可用的上下文上限字段")
    else:
        print("[models] 查询失败或服务未实现该接口")

    max_input_probe, context_hint = probe_max_input_tokens(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        high_words=args.high_words,
    )
    small_prompt_tokens = probe_small_prompt_tokens(base_url, api_key, model, args.timeout)

    context_final = context_from_models or context_hint

    print("\n=== 探测结果 ===")
    if max_input_probe is not None:
        print(f"估算最大输入 token（max_tokens=1 条件下）: ~{max_input_probe}")
    else:
        print("估算最大输入 token: 未探测到（可能被非上下文错误中断）")

    if context_final is not None:
        print(f"估算上下文窗口 token: {context_final}")
    else:
        print("估算上下文窗口 token: 未拿到明确值")

    if context_final is not None and small_prompt_tokens is not None:
        max_output = max(context_final - small_prompt_tokens, 0)
        print(f"估算最大输出 token（极短输入时）: ~{max_output}")
        print(f"（基于: 上下文窗口 {context_final} - 短输入 {small_prompt_tokens}）")
    else:
        print("估算最大输出 token: 依赖上下文窗口，当前无法精确换算")

    print("\n说明: 该脚本给出“接口可接受上限”的近似值，真实可生成长度还受服务端策略、停止词和超时影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
