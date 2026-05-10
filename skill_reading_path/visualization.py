import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "researchtrail-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from shared.data_layer import SharedDataLayer


class VisualizationEngine:
    def __init__(self, data_layer: SharedDataLayer):
        self.data = data_layer

    def render_network(
        self,
        output_path: str = "research_graph.png",
        figsize: tuple = (16, 12),
        node_size_attr: str = "pagerank",
        color_by: str = "community",
    ) -> str:
        graph_data = self.data.get_graph_data()
        scores = self.data.get_all_scores()

        if not graph_data or not scores:
            raise ValueError("Graph data and scores not available.")

        G = nx.Graph()
        for node in graph_data.nodes:
            G.add_node(node)
        for edge in graph_data.edges:
            G.add_edge(edge.source, edge.target, type=edge.type, weight=edge.weight)

        if G.number_of_nodes() == 0:
            raise ValueError("Empty graph, nothing to visualize.")

        plt.figure(figsize=figsize)

        connected = G.subgraph(max(nx.connected_components(G), key=len))
        if connected.number_of_nodes() < G.number_of_nodes():
            print(f"[Vis] Using largest connected component: {connected.number_of_nodes()}/{G.number_of_nodes()} nodes")
        G = connected

        if G.number_of_nodes() > 200:
            pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
        else:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        node_sizes = self._compute_node_sizes(G, scores, node_size_attr)
        node_colors, cmap, norm, legend_labels = self._compute_node_colors(G, scores, color_by)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.9,
        )

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.15,
            edge_color="gray",
            width=0.5,
        )

        top_nodes = sorted(
            G.nodes(),
            key=lambda n: scores.get(n, None) and getattr(scores[n], node_size_attr, 0) or 0,
            reverse=True,
        )[:15]

        labels = {}
        for n in top_nodes:
            paper = self.data.get_paper(n)
            if paper:
                title = paper.title
                labels[n] = title[:50] + "..." if len(title) > 50 else title

        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold")

        if legend_labels:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
                for label, color in legend_labels.items()
            ]
            plt.legend(handles=handles, loc="upper left", fontsize=8, title="Communities")

        plt.title(f"Research Citation Network\n{self.data.get_research_profile() and self.data.get_research_profile().topic or ''}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return os.path.abspath(output_path)

    def _compute_node_sizes(
        self, G: nx.Graph, scores: dict, attr: str
    ) -> list[float]:
        vals = []
        for node in G.nodes():
            sc = scores.get(node)
            val = getattr(sc, attr, 0.0) if sc else 0.0
            vals.append(val)

        if not vals or max(vals) == 0:
            return [200] * len(G.nodes())

        min_val, max_val = min(vals), max(vals)
        if max_val == min_val:
            return [400] * len(G.nodes())

        return [100 + 900 * (v - min_val) / (max_val - min_val) for v in vals]

    def _compute_node_colors(
        self, G: nx.Graph, scores: dict, color_by: str
    ) -> tuple:
        if color_by == "community":
            communities = [scores.get(n, None) and scores[n].community or 0 for n in G.nodes()]
            unique_comms = sorted(set(communities))
            cmap = plt.cm.Set3
            norm = plt.Normalize(vmin=min(unique_comms), vmax=max(unique_comms) + 1)

            legend_labels = {}
            for cid in unique_comms:
                papers_in_comm = []
                for n in G.nodes():
                    sc = scores.get(n)
                    if sc and sc.community == cid:
                        p = self.data.get_paper(n)
                        if p:
                            papers_in_comm.append(p.title[:30])
                label = f"Community {cid}" + (f" ({papers_in_comm[0]}...)" if papers_in_comm else "")
                legend_labels[label] = cmap(norm(cid))

            return communities, cmap, norm, legend_labels
        else:
            vals = [scores.get(n, None) and getattr(scores[n], color_by, 0) or 0 for n in G.nodes()]
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=min(vals), vmax=max(vals) + 0.01)
            return vals, cmap, norm, {}

    def render_score_distribution(self, output_path: str = "scores_distribution.png") -> str:
        scores = self.data.get_all_scores()
        if not scores:
            raise ValueError("No scores available.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = [
            ("pagerank", "PageRank Distribution", "blue"),
            ("betweenness", "Betweenness Centrality Distribution", "orange"),
            ("bridge_score", "Bridge Score Distribution", "green"),
            ("frontier_score", "Frontier Score Distribution", "red"),
        ]

        for ax, (attr, title, color) in zip(axes.flat, metrics):
            vals = [getattr(s, attr) for s in scores.values()]
            ax.hist(vals, bins=20, color=color, alpha=0.7, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return os.path.abspath(output_path)
