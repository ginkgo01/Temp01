import re
from typing import Callable


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
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\!", "")
    s = s.replace(r"\,", ",")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    # 兼容 \frac9{19} 这类简写
    s = re.sub(r"\\frac([^\{\s\\]+)\{", r"\\frac{\1}{", s)
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    # 仅去掉千分位分隔符（如 1,234 或 12,345,678），避免误伤有序对/区间里的逗号
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    s = re.sub(r"\s+", "", s)
    # 选择题常见格式：\text{(E)} / (E) / E -> E
    if re.fullmatch(r"\([A-Za-z]\)", s):
        s = s[1]
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
    if compare_sympy_expr(pred_norm, gold_norm):
        return True

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


def is_correct(pred_text: str, gold_answer: str) -> tuple[bool, str, str]:
    pred_boxed = extract_last_boxed(pred_text)
    if pred_boxed is None:
        return False, "未提取到 \\boxed{...}", ""

    pred_norm = normalize_latex_answer(pred_boxed)
    gold_norm = normalize_latex_answer(gold_answer)
    if pred_norm == gold_norm:
        return True, "规范化字符串匹配", pred_boxed
    if is_math_equivalent(pred_norm, gold_norm):
        return True, "SymPy 数学等价匹配", pred_boxed
    return False, f"不匹配: pred={pred_norm}, gold={gold_norm}", pred_boxed


def llm_fallback_equivalence_judge(
    *,
    base_url: str,
    api_key: str,
    model: str,
    pred_boxed: str,
    gold_answer: str,
    call_chat_completion_fn: Callable[[str, str, dict], str],
    judge_max_tokens: int,
) -> tuple[bool, str]:
    pred_norm = normalize_latex_answer(pred_boxed)
    gold_norm = normalize_latex_answer(gold_answer)
    judge_prompt = (
        "你是数学答案判定器。请判断两个答案是否数学等价，忽略纯格式差异（如空格、"
        "\\dfrac 与 \\frac、\\frac14 与 \\frac{1}{4} 等）。\n"
        "如果等价，只输出 CORRECT；如果不等价，只输出 WRONG。\n\n"
        f"预测答案: {pred_boxed}\n"
        f"标准答案: {gold_answer}\n"
        f"预测规范化: {pred_norm}\n"
        f"标准规范化: {gold_norm}\n"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict math equivalence judge."},
            {"role": "user", "content": judge_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": judge_max_tokens,
        "stream": False,
    }
    raw = call_chat_completion_fn(base_url, api_key, payload)
    text = raw.strip().upper()
    if re.search(r"\bCORRECT\b", text):
        return True, raw
    if re.search(r"\bWRONG\b", text):
        return False, raw
    # 兜底：模型未按要求输出时，按错误处理，避免把明显错误样本误记为正确
    return False, raw
