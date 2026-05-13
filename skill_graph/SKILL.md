---
name: research-graph-analysis
description: "Construct citation/similarity paper networks and identify foundation, bridge, and frontier papers."
author: researchtrail-team
version: 1.0.0
tags:
  - social-network-analysis
  - graph-analysis
  - pagerank
  - community-detection
  - centrality
---

# Research Graph Analysis Skill

## When to Use
Use this Skill after the Literature Retrieval Skill has produced a paper corpus and the Agent needs to understand the structure of a research field.

## Inputs
- Paper corpus from the shared data layer.
- Graph mode: `citation`, `similarity`, or `hybrid`.
- Optional similarity backend: `tfidf`, `lsa`, or `sentence-transformer`.
- Optional community labeling mode: `llm`, `rule`, or `off`.

## Procedure
1. Build citation edges from references when both papers are in the corpus.
2. Build semantic similarity edges from title, keywords, and abstract using the selected backend:
   - `tfidf`: lexical TF-IDF cosine similarity.
   - `lsa`: TF-IDF followed by TruncatedSVD and normalized cosine similarity, used as an offline embedding-style backend.
   - `sentence-transformer`: optional SBERT-style embeddings if the package/model is installed; otherwise fall back to LSA.
3. Compute PageRank, betweenness centrality, and Louvain communities deterministically.
4. Score each paper as foundation, bridge, and frontier.
5. Optionally label communities with an LLM using only top papers, keywords, and year distribution as evidence.
6. Save graph, scores, graph metrics, and community labels to the shared data layer.

## Outputs
- `GraphData`: nodes and typed weighted edges.
- `NodeScores`: PageRank, betweenness, community id, foundation score, bridge score, frontier score.
- Community summaries and optional semantic labels.
- Graph-level metrics for evaluation.

## Evaluation Protocol
- Compare `citation`, `similarity`, and `hybrid` graph modes.
- Record node count, edge count, citation edge count, similarity edge count, connected components, largest component ratio, number of communities, and modularity.
- Record reading-path community coverage for each graph mode to evaluate whether graph construction improves field coverage, not just connectivity.
- Compare `tfidf` and `lsa` similarity backends under the same hybrid graph mode to test whether embedding-style similarity improves connectivity and reading-path coverage.
- Inspect LLM-labeled communities against top papers and keywords for interpretability.
- Use hybrid graph as the default if citation-only is sparse and similarity-only over-connects the corpus.

## Current Evaluation Artifacts
- `outputs/evaluation/skill_graph_report.md`
- `outputs/graph_ablation/full_12_topics/graph_ablation_summary.md`
- `outputs/graph_ablation/full_12_topics/graph_ablation_results.json`
- `outputs/similarity_backend_ablation/full_12_topics/similarity_backend_ablation_summary.md`
- `outputs/similarity_backend_ablation/full_12_topics/similarity_backend_ablation_results.json`
- `outputs/report_materials/final_evaluation_and_implementation_summary.md`

## Known Limitations
- Citation PageRank uses a directed citation graph where citing papers point to cited papers.
- Betweenness uses `distance = 1 / weight`; similarity weight itself remains a strength score for PageRank/degree-style uses.
- Citation edges depend on OpenAlex/Semantic Scholar/reference availability and can be sparse for arXiv-heavy corpora.
- TF-IDF similarity can miss semantic matches when terminology differs; LSA mitigates this but can over-smooth small corpora.
- Similarity edges can connect papers with similar language or latent semantics but weak citation relationships.
- Community labels are semantic summaries; community assignments and graph scores remain deterministic.

## Failure Handling
- If citation edges are sparse, use semantic similarity edges.
- If Louvain fails, fall back to connected components.
- If LLM labeling fails, use deterministic keyword-based community labels.

## Do Not
- Do not ask the LLM to compute centrality, community assignments, or paper role scores.
- Do not mutate the paper corpus.
- Do not generate the final reading report.
