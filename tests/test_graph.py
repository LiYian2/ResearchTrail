import pytest
from unittest.mock import patch, MagicMock

from shared.data_layer import SharedDataLayer
from shared.types import Paper, ResearchProfile, UserLevel, NodeScores, GraphData, GraphEdge
from skill_graph.graph_builder import GraphBuilder
from skill_graph.analysis import GraphAnalysisSkill


class TestGraphBuilder:
    def test_build_empty(self):
        dl = SharedDataLayer()
        gb = GraphBuilder(dl)
        gd = gb.build()
        assert len(gd.nodes) == 0
        assert len(gd.edges) == 0

    def test_build_citation_edges(self):
        dl = SharedDataLayer()
        papers = [
            Paper(paper_id="p1", title="A", authors=["X"], year=2020, abstract="intro to topic", url="",
                  references=["p2"]),
            Paper(paper_id="p2", title="B", authors=["Y"], year=2015, abstract="foundational work", url=""),
        ]
        dl.add_papers(papers)
        gb = GraphBuilder(dl)
        gd = gb.build()
        assert len(gd.nodes) == 2
        assert any(e.source == "p1" and e.target == "p2" and e.type == "citation" for e in gd.edges)

    def test_build_similarity_edges(self):
        dl = SharedDataLayer()
        papers = [
            Paper(paper_id="p1", title="Deep CFR", authors=["X"], year=2024,
                  abstract="counterfactual regret minimization in imperfect-information games using neural networks"),
            Paper(paper_id="p2", title="CFR Methods", authors=["Y"], year=2023,
                  abstract="counterfactual regret minimization for extensive-form games with function approximation"),
        ]
        dl.add_papers(papers)
        gb = GraphBuilder(dl)
        gd = gb.build()
        sim_edges = [e for e in gd.edges if e.type == "similarity"]
        assert len(sim_edges) >= 1
        assert dl.get_metadata("similarity_backend") == "tfidf"

    def test_build_lsa_similarity_edges(self):
        dl = SharedDataLayer()
        papers = [
            Paper(paper_id="p1", title="Graph Convolutional Networks", authors=["X"], year=2017,
                  abstract="graph neural networks with spectral graph convolution for semi supervised node classification"),
            Paper(paper_id="p2", title="Graph Attention Networks", authors=["Y"], year=2018,
                  abstract="graph neural networks with attention mechanisms for node classification on citation networks"),
            Paper(paper_id="p3", title="Transformers for Images", authors=["Z"], year=2020,
                  abstract="vision transformers use self attention for image recognition and visual representation learning"),
        ]
        dl.add_papers(papers)
        gb = GraphBuilder(dl)
        gd = gb.build(mode="similarity", similarity_backend="lsa")
        sim_edges = [e for e in gd.edges if e.type == "similarity"]
        assert len(sim_edges) >= 1
        assert dl.get_metadata("similarity_backend") in {"lsa", "tfidf"}
        assert all(e.distance > 0 for e in sim_edges)

    def test_build_citation_edges_with_external_aliases(self):
        dl = SharedDataLayer()
        papers = [
            Paper(
                paper_id="openalex:W1",
                title="New Method",
                authors=["X"],
                year=2022,
                abstract="new method extending a known arxiv paper",
                references=["1811.00164"],
            ),
            Paper(
                paper_id="arxiv:1811.00164",
                title="Deep Counterfactual Regret Minimization",
                authors=["Y"],
                year=2018,
                abstract="counterfactual regret minimization with deep neural networks",
                external_ids={"arxiv": "1811.00164"},
            ),
        ]
        dl.add_papers(papers)
        gd = GraphBuilder(dl).build(mode="citation")
        assert any(
            e.source == "openalex:W1" and e.target == "arxiv:1811.00164" and e.type == "citation"
            for e in gd.edges
        )

    def test_to_networkx(self):
        dl = SharedDataLayer()
        gd = GraphData(
            nodes=["a", "b", "c"],
            edges=[
                GraphEdge("a", "b", "citation", 1.0),
                GraphEdge("b", "c", "similarity", 0.8),
            ],
        )
        dl.set_graph_data(gd)
        gb = GraphBuilder(dl)
        G = gb.to_networkx(gd)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2


class TestGraphAnalysis:
    def _setup_analysis(self):
        dl = SharedDataLayer()
        papers = [
            Paper(paper_id="p1", title="Foundation", authors=["A"], year=2000, abstract="fundamental theory",
                  citation_count=500, references=["p2", "p3"]),
            Paper(paper_id="p2", title="Method A", authors=["B"], year=2010, abstract="method extending foundation",
                  citation_count=200, references=["p3"]),
            Paper(paper_id="p3", title="Method B", authors=["C"], year=2015, abstract="alternative method",
                  citation_count=150),
            Paper(paper_id="p4", title="Recent", authors=["D"], year=2024, abstract="state of the art approach",
                  citation_count=50, references=["p2", "p3"]),
            Paper(paper_id="p5", title="Frontier", authors=["E"], year=2025, abstract="latest advances",
                  citation_count=10, references=["p4"]),
        ]
        dl.add_papers(papers)

        gb = GraphBuilder(dl)
        gb.build()

        ga = GraphAnalysisSkill(dl)
        return dl, ga

    def test_run_produces_scores(self):
        dl, ga = self._setup_analysis()
        scores = ga.run()
        assert len(scores) == 5
        for pid in ["p1", "p2", "p3", "p4", "p5"]:
            assert pid in scores
            assert scores[pid].pagerank >= 0
            assert scores[pid].betweenness >= 0
            assert scores[pid].community >= 0

    def test_foundation_score_favors_old_high_pr(self):
        dl, ga = self._setup_analysis()
        scores = ga.run()
        # p1 (year 2000) should have higher foundation score than p5 (year 2025)
        assert scores["p1"].foundation_score > scores["p5"].foundation_score

    def test_frontier_score_favors_recent(self):
        dl, ga = self._setup_analysis()
        scores = ga.run()
        # p5 (year 2025) should have higher frontier score than p1 (year 2000)
        assert scores["p5"].frontier_score > scores["p1"].frontier_score

    def test_community_info(self):
        dl, ga = self._setup_analysis()
        ga.run()
        info = ga.get_community_info()
        assert len(info) > 0
        for cid, cinfo in info.items():
            assert cinfo["size"] > 0
            assert len(cinfo["top_papers"]) > 0

    def test_get_nx_graph(self):
        dl, ga = self._setup_analysis()
        ga.run()
        G = ga.get_nx_graph()
        assert G is not None
        assert G.number_of_nodes() == 5
