import pytest
from unittest.mock import patch, MagicMock

from shared.data_layer import SharedDataLayer
from shared.types import (
    Paper, ResearchProfile, UserLevel, NodeScores, GraphData, GraphEdge,
    ReadingPath, ReadingPathItem,
)


class TestSharedDataLayer:
    def test_set_and_get_papers(self):
        dl = SharedDataLayer()
        papers = [
            Paper(paper_id="p1", title="Test Paper", authors=["A"], year=2023, abstract="test", url="x"),
            Paper(paper_id="p2", title="Test Paper 2", authors=["B"], year=2024, abstract="test2", url="y"),
        ]
        dl.add_papers(papers)
        assert dl.get_paper_count() == 2
        assert dl.get_paper("p1").title == "Test Paper"

    def test_profile_roundtrip(self):
        dl = SharedDataLayer()
        profile = ResearchProfile(topic="Deep CFR", user_level=UserLevel.BEGINNER)
        dl.set_research_profile(profile)
        retrieved = dl.get_research_profile()
        assert retrieved.topic == "Deep CFR"
        assert retrieved.user_level == UserLevel.BEGINNER

    def test_scores_roundtrip(self):
        dl = SharedDataLayer()
        scores = {
            "p1": NodeScores(pagerank=0.5, betweenness=0.3, community=1),
        }
        dl.set_node_scores(scores)
        assert dl.get_node_scores("p1").pagerank == 0.5
        assert dl.get_all_scores()["p1"].community == 1

    def test_graph_data_roundtrip(self):
        dl = SharedDataLayer()
        gd = GraphData(
            nodes=["p1", "p2"],
            edges=[GraphEdge(source="p1", target="p2", type="citation", weight=1.0)],
        )
        dl.set_graph_data(gd)
        retrieved = dl.get_graph_data()
        assert len(retrieved.nodes) == 2
        assert len(retrieved.edges) == 1

    def test_reading_path_roundtrip(self):
        dl = SharedDataLayer()
        path = ReadingPath(
            stages=[{"stage": "Foundations", "papers": [{"title": "T", "reason": "R", "read_order": 1}]}],
            report_markdown="# Report",
            followup_suggestions=["Q1"],
        )
        dl.set_reading_path(path)
        assert dl.get_reading_path().report_markdown == "# Report"

    def test_clear_papers(self):
        dl = SharedDataLayer()
        dl.add_papers([Paper(paper_id="p1", title="T", authors=[], year=0, abstract="", url="")])
        dl.clear_papers()
        assert dl.get_paper_count() == 0


class TestPaper:
    def test_paper_to_dict(self):
        p = Paper(paper_id="p1", title="A", authors=["X"], year=2024, abstract="abs", url="http://x")
        d = p.to_dict()
        assert d["paper_id"] == "p1"
        assert d["authors"] == ["X"]
        assert d["references"] == []

    def test_paper_from_dict(self):
        d = {"paper_id": "p2", "title": "B", "authors": ["Y"], "year": 2023, "abstract": "abs", "url": "y"}
        p = Paper.from_dict(d)
        assert p.paper_id == "p2"
        assert p.title == "B"


class TestNodeScores:
    def test_scores_to_dict(self):
        s = NodeScores(pagerank=0.1, betweenness=0.2, community=3, foundation_score=0.5)
        d = s.to_dict()
        assert d["pagerank"] == 0.1
        assert d["community"] == 3

    def test_scores_from_dict(self):
        d = {"pagerank": 0.3, "betweenness": 0.4, "community": 1}
        s = NodeScores.from_dict(d)
        assert s.pagerank == 0.3


class TestGraphData:
    def test_to_dict(self):
        gd = GraphData(nodes=["a"], edges=[GraphEdge("a", "b", "citation", 0.5)])
        d = gd.to_dict()
        assert d["nodes"] == ["a"]
        assert d["edges"][0]["weight"] == 0.5


class TestReadingPath:
    def test_to_dict(self):
        path = ReadingPath(
            stages=[{"stage": "S", "papers": [{"title": "T", "reason": "R", "read_order": 1}]}],
            followup_suggestions=["Q"],
        )
        d = path.to_dict()
        assert len(d["reading_path"]) == 1
        assert d["reading_path"][0]["stage"] == "S"
