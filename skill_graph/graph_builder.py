import re
import numpy as np
import networkx as nx
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from shared.data_layer import SharedDataLayer
from shared.types import Paper, GraphData, GraphEdge


class GraphBuilder:
    def __init__(self, data_layer: SharedDataLayer):
        self.data = data_layer
        self._last_similarity_backend = "none"

    def build(self, mode: str = "hybrid", similarity_backend: str = "tfidf") -> GraphData:
        papers = self.data.get_all_papers()
        nodes = [p.paper_id for p in papers]
        edges: list[GraphEdge] = []
        backend = self._resolve_similarity_backend(similarity_backend)

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
            sim_edges = self._build_similarity_edges(
                papers,
                min_threshold=sim_threshold,
                backend=backend,
            )
            edges += sim_edges

        paper_ids = set(nodes)
        edges = [e for e in edges if e.source in paper_ids and e.target in paper_ids]

        graph_data = GraphData(nodes=nodes, edges=edges)
        self.data.set_graph_data(graph_data)
        self.data.set_metadata("graph_mode", mode)
        self.data.set_metadata("similarity_backend", self._last_similarity_backend)
        metrics = self.compute_metrics(graph_data)
        metrics["similarity_backend"] = self._last_similarity_backend
        self.data.set_metadata("graph_metrics", metrics)
        return graph_data

    def _build_citation_edges(self, papers: list[Paper]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        paper_ids = {p.paper_id for p in papers}
        alias_to_paper_id = self._build_alias_index(papers)
        seen: set[tuple[str, str]] = set()
        for p in papers:
            for ref_id in p.references:
                target_id = alias_to_paper_id.get(self._normalize_id(ref_id), ref_id)
                if target_id in paper_ids and target_id != p.paper_id and (p.paper_id, target_id) not in seen:
                    seen.add((p.paper_id, target_id))
                    edges.append(GraphEdge(
                        source=p.paper_id,
                        target=target_id,
                        type="citation",
                        weight=1.0,
                        distance=1.0,
                    ))
        return edges

    def _build_alias_index(self, papers: list[Paper]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for paper in papers:
            for alias in self._paper_aliases(paper):
                normalized = self._normalize_id(alias)
                if normalized:
                    aliases[normalized] = paper.paper_id
        return aliases

    def _paper_aliases(self, paper: Paper) -> set[str]:
        aliases = {paper.paper_id}
        external_ids = getattr(paper, "external_ids", {}) or {}
        for value in external_ids.values():
            if value:
                aliases.add(str(value))
        if paper.source == "arxiv" or "arxiv.org/abs/" in (paper.url or ""):
            arxiv_match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", f"{paper.paper_id} {paper.url}")
            if arxiv_match:
                aliases.add(arxiv_match.group(1))
        if paper.source == "openalex" and paper.paper_id:
            aliases.add(paper.paper_id.split("/")[-1])
        return aliases

    @staticmethod
    def _normalize_id(value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return ""
        value = value.replace("https://openalex.org/", "")
        value = value.replace("https://doi.org/", "")
        value = value.replace("ARXIV:", "")
        value = re.sub(r"v\d+$", "", value)
        return value.lower()

    def _build_similarity_edges(
        self,
        papers: list[Paper],
        min_threshold: float = 0.08,
        backend: str = "tfidf",
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        valid_papers = [p for p in papers if p.abstract and len(p.abstract) > 50]
        if len(valid_papers) < 2:
            return edges

        texts = [self._similarity_text(p) for p in valid_papers]
        sim_matrix = self._similarity_matrix(texts, backend)

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

    def _similarity_matrix(self, texts: list[str], backend: str) -> np.ndarray:
        tfidf_matrix = self._tfidf_matrix(texts)
        if backend == "tfidf":
            self._last_similarity_backend = "tfidf"
            return cosine_similarity(tfidf_matrix)
        if backend == "lsa":
            return self._lsa_similarity(tfidf_matrix)
        if backend == "sentence-transformer":
            return self._sentence_transformer_similarity(texts, tfidf_matrix)
        self._last_similarity_backend = "tfidf"
        return cosine_similarity(tfidf_matrix)

    @staticmethod
    def _tfidf_matrix(texts: list[str]):
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
            return vectorizer.fit_transform(texts)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
            return vectorizer.fit_transform(texts)

    def _lsa_similarity(self, tfidf_matrix) -> np.ndarray:
        n_samples, n_features = tfidf_matrix.shape
        n_components = min(64, n_samples - 1, n_features - 1)
        if n_components < 2:
            self._last_similarity_backend = "tfidf"
            return cosine_similarity(tfidf_matrix)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = normalize(svd.fit_transform(tfidf_matrix))
        self._last_similarity_backend = "lsa"
        return cosine_similarity(embeddings)

    def _sentence_transformer_similarity(self, texts: list[str], fallback_tfidf_matrix) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = normalize(model.encode(texts, show_progress_bar=False))
            self._last_similarity_backend = "sentence-transformer"
            return cosine_similarity(embeddings)
        except Exception:
            return self._lsa_similarity(fallback_tfidf_matrix)

    @staticmethod
    def _resolve_similarity_backend(backend: str) -> str:
        backend = (backend or "tfidf").strip().lower().replace("_", "-")
        aliases = {
            "st": "sentence-transformer",
            "sbert": "sentence-transformer",
            "sentence-transformers": "sentence-transformer",
        }
        backend = aliases.get(backend, backend)
        if backend in {"tfidf", "lsa", "sentence-transformer"}:
            return backend
        return "tfidf"

    @staticmethod
    def available_similarity_backends() -> list[str]:
        backends = ["tfidf", "lsa"]
        try:
            import sentence_transformers  # noqa: F401

            backends.append("sentence-transformer")
        except Exception:
            pass
        return backends

    @staticmethod
    def _similarity_text(paper: Paper) -> str:
        # Repeat title/keywords so method names and acronyms matter more than
        # generic abstract vocabulary.
        title = paper.title or ""
        keywords = " ".join(paper.keywords or [])
        abstract = paper.abstract or ""
        return f"{title} {title} {keywords} {abstract}"

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
