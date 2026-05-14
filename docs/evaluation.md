# ResearchTrail Final Evaluation and Implementation Summary

This document fixes the report-ready evaluation material for the current ResearchTrail implementation. The metrics come from the saved artifacts under `outputs/` and are intended for group/skill report writing.

## Artifact Map

- Overall 11-topic agent benchmark: `outputs/benchmark_full_llm_normalized/benchmark_summary.md` and `benchmark_results.json`, aggregated over the selected report benchmark set.
- Skill 1 corpus ablation: `outputs/corpus_ablation/full_12_topics/corpus_ablation_batch_summary.md` and `corpus_ablation_batch_results.json`.
- Skill 2 graph ablation: `outputs/graph_ablation/full_12_topics/graph_ablation_summary.md` and `graph_ablation_results.json`.
- Skill 2 similarity backend ablation: `outputs/similarity_backend_ablation/full_12_topics/similarity_backend_ablation_summary.md` and `similarity_backend_ablation_results.json`.
- Skill 3 reading-path ablation: `outputs/reading_path_ablation/full_11_topics/reading_path_ablation_summary.md`, `reading_path_ablation_results.json`, and `paths_for_human_eval/`.
- Manual qualitative rubric: `evaluation/manual_quality_rubric.md`.

## Overall Agent Result

| Scope | Topics | Papers | Edges | Communities | Landmark Hit | Topic Precision | Ordering | Community Coverage | Stage Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benchmark topics | 11 | 22.7 | 131.7 | 4.0 | 67.6% | 93.3% | 74.6% | 90.6% | 97.7% |
Interpretation: the full agent is strongest on topic precision, stage coverage, and community coverage. The main weakness is landmark recall/order on topics where scholarly metadata is incomplete or where the topic contains important non-arXiv/non-OpenAlex artifacts, especially mechanistic interpretability and newer web-native research areas.

## Skill 1: Literature Retrieval Implementation

Skill 1 turns a research topic and user profile into a graph-ready paper corpus. It uses LLM-assisted query normalization when available, deterministic fallback queries otherwise, arXiv/OpenAlex retrieval, duplicate removal by normalized title, topic filtering, verified landmark recovery, Semantic Scholar citation/reference enrichment, and corpus quality metrics. The implementation also uses HTTPS arXiv API calls, global arXiv throttling with retry/backoff, and `S2_API_KEY` when available.

| Variant | Topics | Avg Papers | Avg Raw Records | Avg Duplicates Removed | Abstract Coverage | Citation Metadata | Reference Coverage | Landmark Hit | Topic Precision | Avg Graph Edges | Graph Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_arxiv_only | 11 | 27.3 | 29.4 | 2.1 | 100.0% | 48.7% | 50.5% | 79.4% | 77.2% | 183.9 | 6.44 |
| B_openalex_only | 11 | 31.8 | 40.0 | 8.2 | 88.9% | 100.0% | 92.9% | 69.1% | 30.2% | 95.4 | 3.12 |
| C_arxiv_openalex | 11 | 45.0 | 69.4 | 12.8 | 97.0% | 92.1% | 88.5% | 91.2% | 61.4% | 225.6 | 5.01 |
| D_plus_topic_filtering | 11 | 40.2 | 69.4 | 12.8 | 98.7% | 92.4% | 88.3% | 91.2% | 67.2% | 245.5 | 6.00 |
| E_plus_verified_landmarks | 11 | 40.3 | 69.4 | 12.8 | 98.7% | 93.4% | 89.3% | 92.7% | 67.3% | 245.9 | 5.99 |

Skill 1 conclusion: combining arXiv and OpenAlex improves landmark recall over either source alone. Topic filtering reduces corpus size from 45.0 to about 40.2 papers while raising topic precision from 61.4% to 67.2% and graph edge yield from 5.01 to 6.00. Verified landmarks give a small additional recall gain, from 91.2% to 92.7%, without degrading precision.

Important nuance: E is not supposed to dominate every metric. It is optimized for milestone recall and downstream graph readiness, while arXiv-only can show higher topic precision because arXiv search returns narrower technical papers and OpenAlex has broader, noisier coverage.

## Skill 2: Research Graph Analysis Implementation

Skill 2 builds citation and semantic-similarity graph structure. Citation edges preserve scholarly dependency, while semantic similarity edges recover connectivity when citation/reference metadata is sparse. The production default uses TF-IDF; the Skill also supports an optional LSA backend that turns TF-IDF into normalized low-dimensional vectors with TruncatedSVD, giving an offline embedding-style graph without requiring a large pretrained model. Analysis computes PageRank on a directed citation graph, undirected/hybrid community structure, betweenness with inverse-distance semantics for weighted similarity edges, Louvain/greedy communities, and foundation/bridge/frontier role scores. LLM is used only for community labels, not for centrality computation.

| Mode | Topics | Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Bridge Plausibility | Foundation Landmark Hit | Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| citation | 11 | 7.9 | 18.1 | 20.9% | 0.1086 | 11.6 | 67.3% | 19.1% | 81.8% | 0.27 |
| similarity | 11 | 128.2 | 1.8 | 96.5% | 0.1824 | 4.1 | 88.2% | 77.1% | 81.8% | 5.11 |
| hybrid | 11 | 136.1 | 1.4 | 97.7% | 0.1912 | 4.0 | 90.6% | 81.8% | 72.7% | 5.38 |

Graph ablation conclusion: citation-only is theoretically clean but too sparse in this dataset, averaging only 7.9 edges and 20.9% largest component ratio. Similarity-only gives strong connectivity, but lacks citation direction. Hybrid is the production choice because it keeps citation evidence while improving largest component ratio to 97.7%, path community coverage to 90.6%, and bridge plausibility to 81.8%.

Across benchmark topics, hybrid graph construction preserves the semantic connectivity of similarity graphs while adding citation-direction evidence for PageRank and role scoring.

Similarity backend ablation, with graph mode fixed as hybrid:

| Backend | Topics | Edges | Similarity Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Bridge Plausibility | Foundation Landmark Hit | Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 11 | 75.0 | 67.1 | 2.3 | 92.1% | 0.2879 | 4.6 | 86.0% | 84.6% | 72.7% | 3.15 |
| lsa | 11 | 75.8 | 67.9 | 2.1 | 94.3% | 0.2633 | 4.8 | 88.8% | 89.5% | 81.8% | 3.19 |

Backend conclusion: LSA modestly improves connectivity, path community coverage, bridge plausibility, and foundation landmark hit. TF-IDF has slightly higher modularity, so the best engineering choice is to expose the backend as a configurable option instead of claiming one backend always dominates.

## Skill 3: Reading Path and Report Implementation

Skill 3 converts graph scores and communities into staged reading paths. It assigns papers to prerequisite/foundation/core/development/bridge/frontier stages, uses role evidence packets for explanations, calls the LLM explanation writer when enabled, and generates Markdown reports with abstracts, URLs, role scores, why-read text, graph figures, score distributions, and follow-up suggestions. The GUI additionally lets users mark papers as read and fold completed items.

| Variant | Topics | Path Papers | Landmark Hit | Ordering | Community Coverage | Stage Coverage | Topic Precision | Explanation Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_order | 11 | 10.5 | 63.0% | 55.4% | 95.1% | 0.0% | 94.9% | 100.0% |
| citation_count_order | 11 | 10.5 | 76.1% | 76.3% | 96.4% | 0.0% | 93.4% | 100.0% |
| pagerank_order | 11 | 10.5 | 68.5% | 59.9% | 91.9% | 0.0% | 93.7% | 100.0% |
| researchtrail_staged | 11 | 10.5 | 67.6% | 72.0% | 90.6% | 97.7% | 95.3% | 100.0% |

Skill 3 conclusion: citation-count and PageRank are strong baselines for landmark recall on some topics, but they do not create a pedagogical path. ResearchTrail staged paths are the only variant with high stage coverage, 97.7%, while keeping topic precision at 95.3% and ordering quality at 72.0%. This supports the claim that Skill 3 adds value by turning graph rankings into a usable learning sequence rather than just sorting papers.

## End-to-End Architecture

The agent is a markdown-described, code-backed, LLM-assisted workflow:

1. Agent planner parses the user request, topic, level, and desired output. LLM mode uses SiliconFlow/OpenAI-compatible chat completions with `SILICON_FLOW_API`; fallback mode uses deterministic rules.
2. Literature Retrieval Skill constructs a corpus through query expansion, retrieval, filtering, deduplication, landmark verification, and metadata enrichment.
3. Graph Skill builds citation/similarity/hybrid graphs and computes deterministic SNA metrics and paper role scores. Semantic edges can use TF-IDF or optional LSA embedding-style similarity.
4. Community labeler optionally uses LLM to name graph communities from evidence packets.
5. Reading Path Skill creates staged reading paths and evidence-grounded explanations.
6. Report/visualization layer writes Markdown reports, graph visualizations, score distributions, state files, and GUI-ready outputs.

## Report Claims Supported by Evaluation

- Skill 1 is robust as a corpus builder because multi-source retrieval plus topic filtering improves precision and graph edge yield without sacrificing landmark recall.
- Skill 2 is necessary because citation-only metadata is too sparse for many modern topics; the hybrid graph materially improves connectivity and bridge-paper analysis.
- Optional LSA similarity gives Skill 2 an additional robustness knob: it improves several downstream graph/path metrics on the 11-topic backend ablation while remaining fully offline and reproducible.
- Skill 3 is necessary because ranking baselines do not provide stage structure; staged reading paths preserve high topicality while adding pedagogical organization.
- The full system is not just a summarizer: it retrieves papers, builds a research network, assigns structural paper roles, and generates a personalized reading path with visual evidence.

## Limitations to State Explicitly

- The benchmark landmark lists are curated for evaluation. In the system, verified landmarks are recovered through API metadata and should be described as a recall aid, not as manual insertion of final results.
- OpenAlex has broad coverage but lower topic precision on some engineering-heavy topics; arXiv is narrower and often cleaner but weaker on citation/reference metadata.
- Citation metadata remains incomplete for some domains, which is why the hybrid graph is more reliable than citation-only graph construction.
- LSA is an offline embedding-style backend, not a pretrained scientific embedding model. A sentence-transformer backend is supported as an optional extension, but was not used in the reported 11-topic run because the environment did not include the package/model.
- Landmark recall remains the most difficult metric because many benchmark landmark lists are broader than the final reading path and because some research areas include important non-arXiv or web-native artifacts.
- Role scores are still heuristic combinations of network metrics, recency, citation count, and community position. LLM helps explain evidence but does not replace the deterministic scores.

## Suggested Manual Evaluation Dimensions

Use the files in `outputs/reading_path_ablation/full_11_topics/paths_for_human_eval/` and score each path 1-5 on: topical relevance, landmark coverage, reading-order coherence, community/subfield coverage, explanation usefulness, beginner friendliness, and whether bridge papers are plausible.
