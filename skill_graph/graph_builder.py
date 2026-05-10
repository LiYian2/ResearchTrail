import re
import numpy as np
import networkx as nx
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.data_layer import SharedDataLayer
from shared.types import Paper, GraphData, GraphEdge


class GraphBuilder:
    def __init__(self, data_layer: SharedDataLayer):
        self.data = data_layer

    def build(self, mode: str = "hybrid") -> GraphData:
        papers = self.data.get_all_papers()
        nodes = [p.paper_id for p in papers]
        edges: list[GraphEdge] = []

        citation_edges = self._build_citation_edges(papers)
        if mode in ("citation", "hybrid"):
            edges += citation_edges

        n_cite = len(citation_edges)
        if n_cite < 5:
            sim_threshold = 0.05
        elif n_cite < 20:
            sim_threshold = 0.08
        else:
            sim_threshold = 0.08
        sim_edges = []
        if mode in ("similarity", "hybrid"):
            sim_edges = self._build_similarity_edges(papers, min_threshold=sim_threshold)
            edges += sim_edges

        paper_ids = set(nodes)
        edges = [e for e in edges if e.source in paper_ids and e.target in paper_ids]

        graph_data = GraphData(nodes=nodes, edges=edges)
        self.data.set_graph_data(graph_data)
        self.data.set_metadata("graph_mode", mode)
        self.data.set_metadata("graph_metrics", self.compute_metrics(graph_data))
        return graph_data

    def _build_citation_edges(self, papers: list[Paper]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        paper_ids = {p.paper_id for p in papers}
        for p in papers:
            for ref_id in p.references:
                if ref_id in paper_ids and ref_id != p.paper_id:
                    edges.append(GraphEdge(
                        source=p.paper_id,
                        target=ref_id,
                        type="citation",
                        weight=1.0,
                        distance=1.0,
                    ))
        return edges

    def _build_similarity_edges(self, papers: list[Paper], min_threshold: float = 0.08) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        valid_papers = [p for p in papers if p.abstract and len(p.abstract) > 50]
        if len(valid_papers) < 2:
            return edges

        texts = [p.abstract for p in valid_papers]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)

        sim_matrix = cosine_similarity(tfidf_matrix)

        n = len(sim_matrix)
        if n <= 5:
            threshold = 0.05
        else:
            nonzero = sim_matrix[sim_matrix > 0]
            if len(nonzero) > 0:
                p70 = float(np.percentile(sim_matrix[sim_matrix > 0.01], 70))
            else:
                p70 = min_threshold * 3
            threshold = max(min_threshold, p70 * 0.6)

        for i in range(len(valid_papers)):
            for j in range(i + 1, len(valid_papers)):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    edges.append(GraphEdge(
                        source=valid_papers[i].paper_id,
                        target=valid_papers[j].paper_id,
                        type="similarity",
                        weight=float(sim),
                        distance=float(1.0 / max(sim, 1e-6)),
                    ))

        return edges

    def to_networkx(self, graph_data: GraphData) -> nx.Graph:
        G = nx.Graph()
        for node in graph_data.nodes:
            G.add_node(node)
        for edge in graph_data.edges:
            G.add_edge(
                edge.source,
                edge.target,
                type=edge.type,
                weight=edge.weight,
                distance=edge.distance,
            )
        return G

    def compute_metrics(self, graph_data: GraphData) -> dict:
        G = self.to_networkx(graph_data)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        components = list(nx.connected_components(G)) if n_nodes else []
        largest = max((len(c) for c in components), default=0)
        citation_edges = sum(1 for e in graph_data.edges if e.type == "citation")
        similarity_edges = sum(1 for e in graph_data.edges if e.type == "similarity")
        avg_degree = (sum(dict(G.degree()).values()) / n_nodes) if n_nodes else 0.0
        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "citation_edges": citation_edges,
            "similarity_edges": similarity_edges,
            "connected_components": len(components),
            "largest_component_ratio": largest / max(n_nodes, 1),
            "average_degree": round(avg_degree, 4),
        }
