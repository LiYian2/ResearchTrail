---
name: research-graph-analysis
description: "Construct citation/similarity paper networks and identify foundational, bridge, and frontier papers with deterministic SNA metrics."
author: LiYian2
version: 1.0.0
tags:
  - graph-analysis
  - pagerank
  - betweenness-centrality
  - community-detection
  - citation-network
metadata:
  openclaw:
    requires:
      bins:
        - python
---

# Research Graph Analysis Skill

You are helping the ResearchTrail agent understand the structure of a research field from a retrieved paper corpus.

## When to Use

Use this skill after the Literature Retrieval Skill has produced papers and the user needs:

- foundational paper identification
- bridge paper identification
- frontier paper identification
- research community detection
- graph/network visual evidence
- community labels before report writing

Do not use this skill to retrieve papers or write the final reading report.

## How to Run

Preferred full-agent invocation:

```bash
python main.py "<user research goal>" --llm auto --max-papers 45 --output-dir outputs/<run_name>
```

Graph-mode ablation for this skill:

```bash
python -m evaluation.graph_ablation --topic diffusion_models --max-papers 30 --output-dir outputs/graph_ablation/diffusion
```

From saved benchmark states:

```bash
python -m evaluation.graph_ablation --state-root outputs/benchmark_full_llm_normalized --output-dir outputs/graph_ablation/full_12_topics
```

## Inputs

- Paper corpus in the shared data layer.
- Graph mode: `citation`, `similarity`, or `hybrid`.
- Optional LLM client for community labels.

## Procedure

1. Build citation edges from references when both endpoints are in the corpus.
2. Build semantic similarity edges from TF-IDF cosine similarity over titles and abstracts.
3. Preserve citation direction for PageRank/foundation scoring.
4. Use inverse-distance weighting for betweenness when similarity weights are strengths.
5. Detect communities on the undirected projection.
6. Compute PageRank, betweenness, community id, foundation score, bridge score, and frontier score.
7. Optionally ask the LLM to label communities using only top papers, keywords, and year distribution.

## Outputs

- `GraphData`: nodes and typed weighted edges.
- `NodeScores`: centrality, community, and role scores.
- Graph metrics: edge count, citation/similarity edge count, connected components, largest component ratio, modularity, community count.
- Optional community labels.

## Error Handling

- If citation edges are sparse, switch to `hybrid` mode.
- If community detection fails, fall back to connected components.
- If LLM community labeling fails, use deterministic keyword labels.
- If graph has too few edges, ask the retrieval skill to broaden queries or improve metadata enrichment.

## Evaluation Evidence

See `docs/evaluation.md`. The main graph ablation compares citation-only, similarity-only, and hybrid graphs across benchmark topics.

## Do Not

- Do not let the LLM compute centrality or role scores.
- Do not invent citation edges.
- Do not mutate the paper corpus.
