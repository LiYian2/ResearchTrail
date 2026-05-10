import json
from typing import Any

from agent.llm_client import LLMClient, load_prompt


class ExplanationWriter:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(enabled=False, provider="none")

    def generate(self, packets: list[dict[str, Any]]) -> dict[str, str]:
        if not packets or not self.llm.available():
            return {}
        prompt = load_prompt("explanation_writer.md")
        llm_packets = [
            {k: v for k, v in packet.items() if k != "template_reason"}
            for packet in packets
        ]
        result = self.llm.complete_json(prompt, json.dumps({"evidence_packets": llm_packets}, indent=2), max_tokens=2600)
        if not result.ok or not isinstance(result.data, dict):
            return {}
        raw = result.data.get("explanations", [])
        if not isinstance(raw, list):
            return {}
        explanations = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if paper_id and reason:
                explanations[paper_id] = reason
        return explanations
