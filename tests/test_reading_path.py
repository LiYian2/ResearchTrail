import pytest

from shared.data_layer import SharedDataLayer
from shared.types import (
    Paper, ResearchProfile, UserLevel, NodeScores, GraphData, GraphEdge, ReadingPath,
)
from skill_reading_path.path_generator import ReadingPathSkill
from skill_reading_path.report import ReportGenerator


class TestReadingPathSkill:
    def _setup(self):
        dl = SharedDataLayer()
        profile = ResearchProfile(
            topic="Deep CFR for imperfect-information games",
            user_level=UserLevel.BEGINNER,
            max_papers=100,
            preferred_length=10,
        )
        dl.set_research_profile(profile)

        papers = [
            Paper(paper_id="p1", title="Survey of Game Theory", authors=["A"], year=2005,
                  abstract="comprehensive survey", citation_count=1000),
            Paper(paper_id="p2", title="CFR Algorithm", authors=["B"], year=2007,
                  abstract="regret matching", citation_count=800),
            Paper(paper_id="p3", title="Deep CFR", authors=["C"], year=2019,
                  abstract="neural CFR", citation_count=300),
            Paper(paper_id="p4", title="NFSP", authors=["D"], year=2016,
                  abstract="neural fictitious self play", citation_count=400),
            Paper(paper_id="p5", title="ReBeL", authors=["E"], year=2020,
                  abstract="recursive belief-based learning", citation_count=200),
            Paper(paper_id="p6", title="Bridge Paper", authors=["F"], year=2018,
                  abstract="connecting game theory and deep RL", citation_count=150,
                  references=["p2", "p3", "p4"]),
            Paper(paper_id="p7", title="Foundation Paper", authors=["G"], year=1995,
                  abstract="foundational game theory", citation_count=2000),
            Paper(paper_id="p8", title="Recent Survey 2023", authors=["H"], year=2023,
                  abstract="recent survey of the field", citation_count=50),
            Paper(paper_id="p9", title="Applied CFR", authors=["I"], year=2021,
                  abstract="applied CFR in real games", citation_count=100, references=["p2", "p3"]),
            Paper(paper_id="p10", title="Open Problems", authors=["J"], year=2024,
                  abstract="open problems in game AI", citation_count=30, references=["p3", "p5"]),
        ]
        dl.add_papers(papers)

        gd = GraphData(
            nodes=[f"p{i}" for i in range(1, 11)],
            edges=[
                GraphEdge("p3", "p2", "citation", 1.0),
                GraphEdge("p4", "p2", "citation", 1.0),
                GraphEdge("p5", "p2", "citation", 1.0),
                GraphEdge("p6", "p2", "citation", 1.0),
                GraphEdge("p6", "p3", "citation", 1.0),
                GraphEdge("p6", "p4", "citation", 1.0),
                GraphEdge("p10", "p3", "citation", 1.0),
                GraphEdge("p10", "p5", "citation", 1.0),
                GraphEdge("p9", "p2", "citation", 1.0),
                GraphEdge("p9", "p3", "citation", 1.0),
            ],
        )
        dl.set_graph_data(gd)

        from skill_graph.analysis import GraphAnalysisSkill
        ga = GraphAnalysisSkill(dl)
        ga.run()

        return dl

    def test_generates_path(self):
        dl = self._setup()
        skill = ReadingPathSkill(dl)
        path = skill.run()
        assert path is not None
        assert len(path.stages) > 0
        total = sum(len(s["papers"]) for s in path.stages)
        assert total > 0

    def test_path_has_report(self):
        dl = self._setup()
        skill = ReadingPathSkill(dl)
        path = skill.run()
        assert len(path.report_markdown) > 0
        assert "ResearchTrail" in path.report_markdown

    def test_path_has_followups(self):
        dl = self._setup()
        skill = ReadingPathSkill(dl)
        path = skill.run()
        assert len(path.followup_suggestions) > 0

    def test_stages_have_reasons(self):
        dl = self._setup()
        skill = ReadingPathSkill(dl)
        path = skill.run()
        for stage in path.stages:
            for paper in stage["papers"]:
                assert paper["reason"], f"No reason for {paper['title']}"

    def test_no_duplicate_papers(self):
        dl = self._setup()
        skill = ReadingPathSkill(dl)
        path = skill.run()
        all_ids = []
        for stage in path.stages:
            for paper in stage["papers"]:
                all_ids.append(paper["paper_id"])
        assert len(all_ids) == len(set(all_ids)), "Duplicate papers in reading path"


class TestReportGenerator:
    def _setup(self):
        dl = SharedDataLayer()
        profile = ResearchProfile(topic="Test", user_level=UserLevel.BEGINNER)
        dl.set_research_profile(profile)

        papers = [
            Paper(paper_id="p1", title="Paper 1", authors=["A", "B"], year=2020,
                  abstract="test", citation_count=50),
            Paper(paper_id="p2", title="Paper 2", authors=["C"], year=2021,
                  abstract="test", citation_count=30),
        ]
        dl.add_papers(papers)

        dl.set_node_scores({
            "p1": NodeScores(pagerank=0.2, betweenness=0.1, community=0),
            "p2": NodeScores(pagerank=0.1, betweenness=0.3, community=0),
        })

        from shared.types import ReadingPathItem
        path = ReadingPath(
            stages=[{
                "stage": "Foundations",
                "papers": [
                    {"paper_id": "p1", "title": "Paper 1", "reason": "Key paper", "read_order": 1},
                ],
            }],
            items=[
                ReadingPathItem(paper_id="p1", title="Paper 1", reason="Key paper", read_order=1, stage="Foundations"),
            ],
            followup_suggestions=["Q1", "Q2"],
        )
        dl.set_reading_path(path)
        return dl

    def test_generates_report(self):
        dl = self._setup()
        rg = ReportGenerator(dl)
        report = rg.generate()
        assert "ResearchTrail" in report
        assert "Paper 1" in report

    def test_explain_paper_role(self):
        dl = self._setup()
        rg = ReportGenerator(dl)
        explanation = rg.explain_paper_role("p1")
        assert "Paper 1" in explanation
        assert "Foundation" in explanation

    def test_generate_bridge_report(self):
        dl = self._setup()
        rg = ReportGenerator(dl)
        report = rg.generate_bridge_report()
        assert "Bridge" in report
        assert "Paper 2" in report

    def test_generate_seven_day_plan(self):
        dl = self._setup()
        rg = ReportGenerator(dl)
        plan = rg.generate_seven_day_plan()
        assert "Monday" in plan
        assert "7-Day" in plan

    def test_explain_nonexistent_paper(self):
        dl = self._setup()
        rg = ReportGenerator(dl)
        result = rg.explain_paper_role("nonexistent")
        assert "No paper found" in result
