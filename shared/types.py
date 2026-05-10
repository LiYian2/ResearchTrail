from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class UserLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class IntentType(Enum):
    BUILD_READING_PATH = "build_reading_path"
    EXPLAIN_PAPER = "explain_paper"
    FILTER_BY_YEAR = "filter_by_year"
    FIND_BRIDGE_PAPERS = "find_bridge_papers"
    VISUALIZE = "visualize"
    ASK_FOLLOWUP = "ask_followup"


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str = ""
    url: str = ""
    citation_count: int = 0
    citation_source: str = ""
    references: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    source: str = "unknown"
    keywords: list[str] = field(default_factory=list)
    venue: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "citation_count": self.citation_count,
            "citation_source": self.citation_source,
            "references": self.references,
            "citations": self.citations,
            "source": self.source,
            "keywords": self.keywords,
            "venue": self.venue,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", 0),
            abstract=data.get("abstract", ""),
            url=data.get("url", ""),
            citation_count=data.get("citation_count", 0),
            citation_source=data.get("citation_source", ""),
            references=data.get("references", []),
            citations=data.get("citations", []),
            source=data.get("source", "unknown"),
            keywords=data.get("keywords", []),
            venue=data.get("venue", ""),
        )


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str
    weight: float = 1.0
    distance: float = 1.0


@dataclass
class GraphData:
    nodes: list[str]
    edges: list[GraphEdge]

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "weight": e.weight,
                    "distance": e.distance,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphData":
        return cls(
            nodes=data.get("nodes", []),
            edges=[
                GraphEdge(
                    source=e["source"],
                    target=e["target"],
                    type=e.get("type", "citation"),
                    weight=e.get("weight", 1.0),
                    distance=e.get("distance", 1.0),
                )
                for e in data.get("edges", [])
            ],
        )


@dataclass
class NodeScores:
    pagerank: float = 0.0
    betweenness: float = 0.0
    community: int = 0
    foundation_score: float = 0.0
    bridge_score: float = 0.0
    frontier_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pagerank": self.pagerank,
            "betweenness": self.betweenness,
            "community": self.community,
            "foundation_score": self.foundation_score,
            "bridge_score": self.bridge_score,
            "frontier_score": self.frontier_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeScores":
        return cls(
            pagerank=data.get("pagerank", 0.0),
            betweenness=data.get("betweenness", 0.0),
            community=data.get("community", 0),
            foundation_score=data.get("foundation_score", 0.0),
            bridge_score=data.get("bridge_score", 0.0),
            frontier_score=data.get("frontier_score", 0.0),
        )


@dataclass
class ReadingPathItem:
    paper_id: str
    title: str
    reason: str
    read_order: int
    stage: str = ""


@dataclass
class ReadingPath:
    stages: list[dict] = field(default_factory=list)
    items: list[ReadingPathItem] = field(default_factory=list)
    report_markdown: str = ""
    followup_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reading_path": [
                {
                    "stage": s["stage"],
                    "papers": [
                        {"title": p["title"], "reason": p["reason"], "read_order": p["read_order"]}
                        for p in s.get("papers", [])
                    ],
                }
                for s in self.stages
            ],
            "report_markdown": self.report_markdown,
            "followup_suggestions": self.followup_suggestions,
        }


@dataclass
class ResearchProfile:
    topic: str = ""
    seed_papers: list[str] = field(default_factory=list)
    time_range: str = "all"
    max_papers: int = 100
    user_level: UserLevel = UserLevel.BEGINNER
    preferred_length: int = 10
    goal: str = "enter the field"

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "seed_papers": self.seed_papers,
            "time_range": self.time_range,
            "max_papers": self.max_papers,
            "user_level": self.user_level.value,
            "preferred_length": self.preferred_length,
            "goal": self.goal,
        }
