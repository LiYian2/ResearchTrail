import json
import re
from collections import Counter
from typing import Any

from agent.llm_client import LLMClient, load_prompt
from shared.data_layer import SharedDataLayer


class CommunityLabeler:
    def __init__(self, data_layer: SharedDataLayer, llm: LLMClient | None = None):
        self.data = data_layer
        self.llm = llm or LLMClient(enabled=False, provider="none")

    def run(self) -> dict[int, dict[str, str]]:
        evidence = self._build_evidence()
        labels = self._label_with_llm(evidence)
        if not labels:
            labels = self._label_with_rules(evidence)
        self.data.set_metadata("community_labels", labels)
        self.data.set_metadata("community_label_evidence", evidence)
        return labels

    def _build_evidence(self) -> list[dict[str, Any]]:
        scores = self.data.get_all_scores()
        communities: dict[int, list[dict[str, Any]]] = {}
        for pid, score in scores.items():
            paper = self.data.get_paper(pid)
            if not paper:
                continue
            communities.setdefault(score.community, []).append({
                "paper_id": pid,
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "pagerank": score.pagerank,
            })

        evidence = []
        for cid, papers in sorted(communities.items()):
            top = sorted(papers, key=lambda p: p["pagerank"], reverse=True)[:5]
            years = [p["year"] for p in papers if p["year"]]
            evidence.append({
                "community_id": cid,
                "size": len(papers),
                "avg_year": round(sum(years) / len(years), 1) if years else 0,
                "top_titles": [p["title"] for p in top],
                "keywords": self._keywords(papers),
            })
        return evidence

    def _label_with_llm(self, evidence: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        if not evidence or not self.llm.available():
            return {}
        prompt = load_prompt("community_labeler.md")
        result = self.llm.complete_json(prompt, json.dumps({"communities": evidence}, indent=2), max_tokens=1800)
        if not result.ok or not isinstance(result.data, dict):
            self.data.set_metadata("community_label_fallback_reason", result.error)
            return {}
        raw = result.data.get("communities", [])
        labels: dict[int, dict[str, str]] = {}
        if not isinstance(raw, list):
            return {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item["community_id"])
            except Exception:
                continue
            label = str(item.get("label", "")).strip()
            description = str(item.get("description", "")).strip()
            if label:
                labels[cid] = {"label": label, "description": description}
        return labels

    def _label_with_rules(self, evidence: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        labels = {}
        for item in evidence:
            cid = int(item["community_id"])
            keywords = item.get("keywords", [])[:4]
            label = " / ".join(keywords).title() if keywords else f"Community {cid}"
            labels[cid] = {
                "label": label,
                "description": f"Community {cid} groups papers around {', '.join(keywords) or 'related methods'}."
            }
        return labels

    @staticmethod
    def _keywords(papers: list[dict[str, Any]]) -> list[str]:
        stop = {
            "with", "using", "based", "from", "for", "and", "the", "of", "in", "on",
            "to", "a", "an", "by", "via", "paper", "study", "method", "approach",
            "learning", "research", "analysis", "model", "models", "network", "networks",
            "this", "that", "these", "those", "their", "our", "we", "are", "is", "was",
            "were", "be", "been", "being", "novel", "new", "effective", "efficient",
            "practical", "emerging", "early", "modern", "scalable", "robust", "methods",
            "deep", "understand", "understanding", "want", "deeply",
        }
        counter: Counter[str] = Counter()
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            for word in re.findall(r"[a-z][a-z\-]{3,}", text):
                if word not in stop:
                    counter[word] += 1
        return [word for word, _ in counter.most_common(8)]
