import json
import re
from urllib import error, request

from datasets import load_from_disk


BASE_URL = "http://localhost:6002/v1"
API_KEY = "1e174CY6rKs28HcNjxhv"
MODEL = "model/Qwen3-8B"
DATASET_PATH = "/mnt/common/lx/Temp01/math500_hf"
PROMPT_PATH = "/mnt/common/lx/Temp01/solver_prompt.txt"


def extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    buff = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            buff.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buff).strip()
            buff.append(ch)
        else:
            buff.append(ch)
        i += 1
    return None


def normalize_latex_answer(ans: str) -> str:
    s = ans.strip()
    s = s.replace("\n", " ")
    s = s.replace("$", "")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    # 统一普通括号与 latex 大括号包裹的轻微差异
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def split_top_level_commas(s: str) -> list[str]:
    parts = []
    buff = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
            buff.append(ch)
        elif ch in ")]}":
            depth = max(depth - 1, 0)
            buff.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buff).strip())
            buff = []
        else:
            buff.append(ch)
    parts.append("".join(buff).strip())
    return [p for p in parts if p]


def strip_outer_pair(s: str, left: str, right: str) -> str:
    text = s.strip()
    if text.startswith(left) and text.endswith(right):
        return text[1:-1].strip()
    return text


def compare_sympy_expr(a: str, b: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return False

    def to_expr(x: str):
        x = x.strip()
        try:
            return parse_latex(x)
        except Exception:
            pass
        try:
            return parse_expr(x)
        except Exception:
            return None

    ea = to_expr(a)
    eb = to_expr(b)
    if ea is None or eb is None:
        return False

    try:
        return simplify(ea - eb) == 0
    except Exception:
        return False


def is_math_equivalent(pred_norm: str, gold_norm: str) -> bool:
    # 1) 先尝试整体表达式等价（覆盖数字、分数、一般表达式）
    if compare_sympy_expr(pred_norm, gold_norm):
        return True

    # 2) 再尝试元组/向量逐项比较，如 (3,\pi/2)
    pred_tuple = strip_outer_pair(strip_outer_pair(pred_norm, r"\(", r"\)"), "(", ")")
    gold_tuple = strip_outer_pair(strip_outer_pair(gold_norm, r"\(", r"\)"), "(", ")")
    pred_parts = split_top_level_commas(pred_tuple)
    gold_parts = split_top_level_commas(gold_tuple)
    if len(pred_parts) > 1 and len(pred_parts) == len(gold_parts):
        for pa, pb in zip(pred_parts, gold_parts):
            if not compare_sympy_expr(pa, pb):
                return False
        return True

    return False


def is_correct(pred_text: str, gold_answer: str) -> tuple[bool, str]:
    pred_boxed = extract_last_boxed(pred_text)
    if pred_boxed is None:
        return False, "未提取到 \\boxed{...}"

    pred_norm = normalize_latex_answer(pred_boxed)
    gold_norm = normalize_latex_answer(gold_answer)
    if pred_norm == gold_norm:
        return True, "规范化字符串匹配"
    if is_math_equivalent(pred_norm, gold_norm):
        return True, "SymPy 数学等价匹配"
    return False, f"不匹配: pred={pred_norm}, gold={gold_norm}"


def call_chat_completion(base_url: str, api_key: str, payload: dict, timeout: float = 30.0) -> str:
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


def main() -> None:
    # 1) 初始化 LLM 调用参数
    print("初始化 LLM...")
    print(f"BASE_URL={BASE_URL}")
    print(f"MODEL={MODEL}")

    # 2) 加载数据集并取第一道题
    print("加载数据集...")
    ds = load_from_disk(DATASET_PATH)
    dataset = ds["test"] if "test" in ds else next(iter(ds.values()))
    first_problem = dataset[0]["problem"]
    gold_answer = dataset[0]["answer"]

    # 3) 读取 prompt 模板并拼接题目
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{{problem}}", first_problem)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 5000,
        "stream": False,
    }

    # 4) 调用并打印输出
    print("调用 LLM 解第一题...")
    try:
        output = call_chat_completion(base_url=BASE_URL, api_key=API_KEY, payload=payload)
        print("模型输出：")
        print(output)
        ok, reason = is_correct(output, gold_answer)
        print(f"标准答案: {gold_answer}")
        print(f"提取boxed: {extract_last_boxed(output)}")
        print(f"判定结果: {ok} ({reason})")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP 错误 {exc.code}: {detail}")
    except error.URLError as exc:
        print(f"网络错误: {exc.reason}")
    except Exception as exc:
        print(f"未知错误: {exc}")


if __name__ == "__main__":
    main()