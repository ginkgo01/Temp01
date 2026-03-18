from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent

SUCCESS_REPORT_PATH = ROOT_DIR / "report_success.json"
FAILED_REPORT_PATH = ROOT_DIR / "report_failed.json"

PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
REPAIR_PROMPT_PATH = PROMPTS_DIR / "repair_prompt.txt"
LOCAL_JUDGE_PROMPT_PATH = PROMPTS_DIR / "local_judge_prompt.txt"
CONTINUE_SOLVE_PROMPT_PATH = PROMPTS_DIR / "continue_solve_prompt.txt"

RUNS_DIR = EXPERIMENT_DIR / "data" / "runs"
REPAIRED_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "repaired_segments"
JUDGED_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "judged_segments"
SUMMARIES_DIR = EXPERIMENT_DIR / "outputs" / "summaries"

L_VALUES = (5, 10)
DEFAULT_RANDOM_SEED = 43


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class StageConfig:
    endpoint: EndpointConfig
    temperature: float
    max_tokens: int
    timeout: float


def _get_env_or_default(prefix: str, key: str, default: str) -> str:
    return os.getenv(f"{prefix}_{key}", default)


def build_endpoint(prefix: str, default: EndpointConfig) -> EndpointConfig:
    return EndpointConfig(
        base_url=_get_env_or_default(prefix, "BASE_URL", default.base_url),
        api_key=_get_env_or_default(prefix, "API_KEY", default.api_key),
        model=_get_env_or_default(prefix, "MODEL", default.model),
    )


DEFAULT_LOCAL_ENDPOINT = EndpointConfig(
    base_url="http://localhost:6002/v1",
    api_key="1e174CY6rKs28HcNjxhv",
    model="model/Qwen3-8B",
)

DEFAULT_GPT54_ENDPOINT = EndpointConfig(
    base_url="https://www.right.codes/codex/v1",
    api_key="sk-d91df136a2a74448adb137096981464a",
    model="gpt-5.4-high",
)

REPAIR_STAGE = StageConfig(
    endpoint=build_endpoint("EXPERIMENT01_REPAIR", DEFAULT_LOCAL_ENDPOINT),
    temperature=float(os.getenv("EXPERIMENT01_REPAIR_TEMPERATURE", "0.3")),
    max_tokens=int(os.getenv("EXPERIMENT01_REPAIR_MAX_TOKENS", "5000")),
    timeout=float(os.getenv("EXPERIMENT01_REPAIR_TIMEOUT", "300")),
)

LOCAL_JUDGE_STAGE = StageConfig(
    endpoint=build_endpoint("EXPERIMENT01_LOCAL_JUDGE", DEFAULT_GPT54_ENDPOINT),
    temperature=float(os.getenv("EXPERIMENT01_LOCAL_JUDGE_TEMPERATURE", "0.0")),
    max_tokens=int(os.getenv("EXPERIMENT01_LOCAL_JUDGE_MAX_TOKENS", "2500")),
    timeout=float(os.getenv("EXPERIMENT01_LOCAL_JUDGE_TIMEOUT", "300")),
)

CONTINUATION_STAGE = StageConfig(
    endpoint=build_endpoint("EXPERIMENT01_CONTINUATION", DEFAULT_LOCAL_ENDPOINT),
    temperature=float(os.getenv("EXPERIMENT01_CONTINUATION_TEMPERATURE", "0.6")),
    max_tokens=int(os.getenv("EXPERIMENT01_CONTINUATION_MAX_TOKENS", "9000")),
    timeout=float(os.getenv("EXPERIMENT01_CONTINUATION_TIMEOUT", "600")),
)

EQUIV_JUDGE_ENDPOINT = build_endpoint("EXPERIMENT01_EQUIV_JUDGE", DEFAULT_LOCAL_ENDPOINT)
EQUIV_JUDGE_MAX_TOKENS = int(os.getenv("EXPERIMENT01_EQUIV_JUDGE_MAX_TOKENS", "2500"))
