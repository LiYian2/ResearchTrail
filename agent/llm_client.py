import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Pro/zai-org/GLM-4.7"


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    data: dict[str, Any] | list[Any] | None = None
    error: str = ""


class LLMClient:
    def __init__(
        self,
        enabled: bool = True,
        provider: str = "siliconflow",
        model: str = DEFAULT_MODEL,
        api_key_env: str = "SILICON_FLOW_API",
        base_url: str | None = None,
        timeout: int = 45,
    ):
        self.enabled = enabled and provider != "none"
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = (base_url or os.environ.get("SILICON_FLOW_BASE_URL") or SILICONFLOW_BASE_URL).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        return self.enabled and self.provider == "siliconflow" and bool(os.environ.get(self.api_key_env))

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> LLMResult:
        result = self.complete(system_prompt, user_prompt, temperature, max_tokens)
        if not result.ok:
            return result
        try:
            result.data = self._extract_json(result.content)
            return result
        except ValueError as exc:
            return LLMResult(ok=False, content=result.content, error=f"invalid_json: {exc}")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResult:
        if not self.enabled:
            return LLMResult(ok=False, error="llm_disabled")
        if self.provider != "siliconflow":
            return LLMResult(ok=False, error=f"unsupported_provider:{self.provider}")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return LLMResult(ok=False, error=f"missing_env:{self.api_key_env}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""
        for _ in range(2):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    last_error = f"http_{resp.status_code}:{resp.text[:200]}"
                    continue
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return LLMResult(ok=True, content=content)
            except Exception as exc:
                last_error = str(exc)
        return LLMResult(ok=False, error=last_error or "unknown_llm_error")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | list[Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        start_positions = [pos for pos in [stripped.find("{"), stripped.find("[")] if pos != -1]
        if not start_positions:
            raise ValueError("no JSON object or array found")
        start = min(start_positions)
        candidate = LLMClient._balanced_json_slice(stripped[start:])
        return json.loads(candidate)

    @staticmethod
    def _balanced_json_slice(text: str) -> str:
        opening = text[0]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[: idx + 1]
        raise ValueError("unterminated JSON object or array")


def load_prompt(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "prompts", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
