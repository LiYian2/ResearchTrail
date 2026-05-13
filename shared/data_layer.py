import json
import re
from typing import Optional, Any
from shared.types import Paper, GraphData, NodeScores, ReadingPath, ResearchProfile


class SharedDataLayer:
    def __init__(self):
        self._store: dict[str, Any] = {
            "research_profile": None,
            "papers": {},
            "graph_data": None,
            "node_scores": {},
            "reading_path": None,
            "corpus_quality": {},
            "metadata": {},
        }

    def set_research_profile(self, profile: ResearchProfile) -> None:
        self._store["research_profile"] = profile

    def get_research_profile(self) -> Optional[ResearchProfile]:
        return self._store["research_profile"]

    def add_papers(self, papers: list[Paper]) -> None:
        for p in papers:
            existing_id = self._find_same_title_id(p.title)
            if existing_id and existing_id != p.paper_id:
                existing = self._store["papers"][existing_id]
                merged = self._merge_papers(existing, p)
                if merged.paper_id != existing_id:
                    del self._store["papers"][existing_id]
                self._store["papers"][merged.paper_id] = merged
                continue
            self._store["papers"][p.paper_id] = p

    def _find_same_title_id(self, title: str) -> Optional[str]:
        title_key = self._normalize_title(title)
        if not title_key:
            return None
        for pid, paper in self._store["papers"].items():
            if self._normalize_title(paper.title) == title_key:
                return pid
        return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()

    @staticmethod
    def _merge_papers(a: Paper, b: Paper) -> Paper:
        keep = a if a.citation_count >= b.citation_count else b
        other = b if keep is a else a
        citation_best = keep
        if other.source == "arxiv" and keep.source != "arxiv":
            keep.abstract = other.abstract or keep.abstract
            keep.url = other.url or keep.url
            keep.authors = other.authors or keep.authors
            keep.year = other.year or keep.year
            keep.venue = other.venue or keep.venue
            keep.source = other.source
            keep.paper_id = other.paper_id or keep.paper_id
        if not keep.abstract and other.abstract:
            keep.abstract = other.abstract
        if not keep.url and other.url:
            keep.url = other.url
        if not keep.authors and other.authors:
            keep.authors = other.authors
        if not keep.year and other.year:
            keep.year = other.year
        if not keep.venue and other.venue:
            keep.venue = other.venue
        keep.citation_count = max(a.citation_count, b.citation_count)
        keep.citation_source = citation_best.citation_source or other.citation_source or keep.citation_source
        refs = list(dict.fromkeys((keep.references or []) + (other.references or [])))
        keep.references = refs
        citations = list(dict.fromkeys((keep.citations or []) + (other.citations or [])))
        keep.citations = citations
        keep.external_ids = {**(other.external_ids or {}), **(keep.external_ids or {})}
        return keep

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self._store["papers"].get(paper_id)

    def get_all_papers(self) -> list[Paper]:
        return list(self._store["papers"].values())

    def get_paper_count(self) -> int:
        return len(self._store["papers"])

    def set_graph_data(self, graph_data: GraphData) -> None:
        self._store["graph_data"] = graph_data

    def get_graph_data(self) -> Optional[GraphData]:
        return self._store["graph_data"]

    def set_node_scores(self, scores: dict[str, NodeScores]) -> None:
        self._store["node_scores"] = scores

    def get_node_scores(self, paper_id: str) -> Optional[NodeScores]:
        return self._store["node_scores"].get(paper_id)

    def get_all_scores(self) -> dict[str, NodeScores]:
        return self._store["node_scores"]

    def set_reading_path(self, path: ReadingPath) -> None:
        self._store["reading_path"] = path

    def get_reading_path(self) -> Optional[ReadingPath]:
        return self._store["reading_path"]

    def set_corpus_quality(self, quality: dict) -> None:
        self._store["corpus_quality"] = quality

    def get_corpus_quality(self) -> dict:
        return self._store["corpus_quality"]

    def set_metadata(self, key: str, value: Any) -> None:
        self._store["metadata"][key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        return self._store["metadata"].get(key)

    def get_all_metadata(self) -> dict[str, Any]:
        return dict(self._store["metadata"])

    def to_dict(self) -> dict:
        return {
            "research_profile": (
                self._store["research_profile"].to_dict()
                if self._store["research_profile"]
                else None
            ),
            "paper_count": len(self._store["papers"]),
            "papers": {
                pid: p.to_dict() for pid, p in self._store["papers"].items()
            },
            "node_scores": {
                pid: s.to_dict() for pid, s in self._store["node_scores"].items()
            },
            "graph_data": (
                self._store["graph_data"].to_dict()
                if self._store["graph_data"]
                else None
            ),
            "reading_path": (
                self._store["reading_path"].to_dict()
                if self._store["reading_path"]
                else None
            ),
            "corpus_quality": self._store["corpus_quality"],
            "metadata": self._store["metadata"],
        }

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def clear_papers(self) -> None:
        self._store["papers"] = {}
        self._store["graph_data"] = None
        self._store["node_scores"] = {}
        self._store["reading_path"] = None
        self._store["corpus_quality"] = {}
