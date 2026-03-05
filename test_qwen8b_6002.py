#!/usr/bin/env python3
import argparse
import importlib.util
import json
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
        if settings_cls is None:
            return base_url, api_key, model

        base_url = getattr(settings_cls, "LLM_LIGHT_BASE_URL", base_url)
        api_key = getattr(settings_cls, "LLM_LIGHT_API_KEY", api_key)
        model = getattr(settings_cls, "LLM_LIGHT_MODEL_NAME", model)
    except Exception:
        # 配置加载失败时回退到默认值，避免影响连通性测试。
        pass

    return base_url, api_key, model


def build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
        "stream": False,
    }


def call_chat_completion(base_url: str, api_key: str, payload: dict, timeout: float) -> dict:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return {"status": resp.status, "body": raw}


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 6002 端口 Qwen3-8B 模型连通性")
    parser.add_argument("--settings", default="llm_settings.py", help="配置文件路径")
    parser.add_argument("--base-url", default=None, help="覆盖 base url，例如 http://localhost:6002/v1")
    parser.add_argument("--api-key", default=None, help="覆盖 API Key")
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--prompt", default="请只回复：pong", help="测试提示词")
    parser.add_argument("--timeout", type=float, default=15.0, help="请求超时（秒）")
    args = parser.parse_args()

    settings_base_url, settings_api_key, settings_model = load_settings_from_file(Path(args.settings))
    base_url = args.base_url or settings_base_url
    api_key = args.api_key or settings_api_key
    model = args.model or settings_model

    print("=== 8B 模型调用测试 ===")
    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"Timeout  : {args.timeout}s")

    payload = build_payload(model=model, prompt=args.prompt)

    try:
        result = call_chat_completion(base_url=base_url, api_key=api_key, payload=payload, timeout=args.timeout)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[失败] HTTP {exc.code}: {detail}")
        return 2
    except error.URLError as exc:
        print(f"[失败] 网络错误: {exc.reason}")
        return 3
    except Exception as exc:
        print(f"[失败] 未知错误: {exc}")
        return 4

    print(f"[成功] HTTP {result['status']}")
    try:
        data = json.loads(result["body"])
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content:
            print(f"模型回复: {content}")
        else:
            print("模型返回了响应，但未解析到 content。原始响应如下：")
            print(result["body"])
    except json.JSONDecodeError:
        print("返回内容不是 JSON，原始响应如下：")
        print(result["body"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
