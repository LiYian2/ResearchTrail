# ResearchTrail Final Evaluation and Implementation Summary

This document fixes the report-ready evaluation material for the current ResearchTrail implementation. The metrics come from the saved artifacts under `outputs/` and are intended for group/skill report writing.

## Artifact Map

- Overall 12-topic agent benchmark: `outputs/benchmark_full_llm_normalized/benchmark_summary.md` and `benchmark_results.json`.
- Skill 1 corpus ablation: `outputs/corpus_ablation/full_12_topics/corpus_ablation_batch_summary.md` and `corpus_ablation_batch_results.json`.
- Skill 2 graph ablation: `outputs/graph_ablation/full_12_topics/graph_ablation_summary.md` and `graph_ablation_results.json`.
- Skill 2 similarity backend ablation: `outputs/similarity_backend_ablation/full_12_topics/similarity_backend_ablation_summary.md` and `similarity_backend_ablation_results.json`.
- Skill 3 reading-path ablation: `outputs/reading_path_ablation/full_11_topics/reading_path_ablation_summary.md`, `reading_path_ablation_results.json`, and `paths_for_human_eval/`.
- Manual qualitative rubric: `evaluation/manual_quality_rubric.md`.

## Overall Agent Result

| Scope | Topics | Papers | Edges | Communities | Landmark Hit | Topic Precision | Ordering | Community Coverage | Stage Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All benchmark topics | 12 | 21.5 | 121.8 | 3.9 | 61.9% | 91.7% | 68.4% | 91.4% | 95.8% |
| Excluding protein low-corpus case | 11 | 22.7 | 131.7 | 4.0 | 67.6% | 93.3% | 74.6% | 90.6% | 97.7% |

Interpretation: the full agent is strongest on topic precision, stage coverage, and community coverage. The main weakness is landmark recall/order on topics where scholarly metadata is incomplete or where the topic contains important non-arXiv/non-OpenAlex artifacts, especially mechanistic interpretability and the saved protein benchmark corpus.

## Skill 1: Literature Retrieval Implementation

Skill 1 turns a research topic and user profile into a graph-ready paper corpus. It uses LLM-assisted query normalization when available, deterministic fallback queries otherwise, arXiv/OpenAlex retrieval, duplicate removal by normalized title, topic filtering, verified landmark recovery, Semantic Scholar citation/reference enrichment, and corpus quality metrics. The latest implementation also uses HTTPS arXiv API calls, global arXiv throttling with retry/backoff, and `S2_API_KEY` when available.

| Variant | Topics | Avg Papers | Avg Raw Records | Avg Duplicates Removed | Abstract Coverage | Citation Metadata | Reference Coverage | Landmark Hit | Topic Precision | Avg Graph Edges | Graph Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_arxiv_only | 12 | 26.2 | 28.6 | 2.4 | 100.0% | 51.8% | 52.8% | 72.8% | 75.5% | 172.5 | 6.18 |
| B_openalex_only | 12 | 32.2 | 40.0 | 7.8 | 88.4% | 100.0% | 93.5% | 70.0% | 31.1% | 98.0 | 3.16 |
| C_arxiv_openalex | 12 | 45.0 | 68.6 | 12.6 | 96.7% | 92.8% | 89.1% | 90.3% | 60.6% | 224.8 | 5.00 |
| D_plus_topic_filtering | 12 | 40.0 | 68.6 | 12.6 | 98.6% | 92.8% | 88.8% | 90.3% | 66.6% | 241.4 | 5.93 |
| E_plus_verified_landmarks | 12 | 40.1 | 68.6 | 12.6 | 98.6% | 93.7% | 89.8% | 91.7% | 66.7% | 241.8 | 5.92 |

Skill 1 conclusion: combining arXiv and OpenAlex improves landmark recall over either source alone. Topic filtering reduces corpus size from 45.0 to about 40.0 papers while raising topic precision from 60.6% to 66.6% and graph edge yield from 5.00 to 5.93. Verified landmarks give a small additional recall gain, from 90.3% to 91.7%, without degrading precision.

Important nuance: E is not supposed to dominate every metric. It is optimized for milestone recall and downstream graph readiness, while arXiv-only can show higher topic precision because arXiv search returns narrower technical papers and OpenAlex has broader, noisier coverage.

## Skill 2: Research Graph Analysis Implementation

Skill 2 builds citation and semantic-similarity graph structure. Citation edges preserve scholarly dependency, while semantic similarity edges recover connectivity when citation/reference metadata is sparse. The production default uses TF-IDF; the Skill also supports an optional LSA backend that turns TF-IDF into normalized low-dimensional vectors with TruncatedSVD, giving an offline embedding-style graph without requiring a large pretrained model. Analysis computes PageRank on a directed citation graph, undirected/hybrid community structure, betweenness with inverse-distance semantics for weighted similarity edges, Louvain/greedy communities, and foundation/bridge/frontier role scores. LLM is used only for community labels, not for centrality computation.

| Mode | Topics | Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Bridge Plausibility | Foundation Landmark Hit | Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| citation | 12 | 7.2 | 17.2 | 20.2% | 0.0996 | 10.8 | 70.1% | 17.5% | 75.0% | 0.25 |
| similarity | 12 | 118.5 | 1.8 | 95.7% | 0.1712 | 4.0 | 89.2% | 77.6% | 75.0% | 4.81 |
| hybrid | 12 | 125.8 | 1.4 | 96.9% | 0.1792 | 3.9 | 91.4% | 81.9% | 66.7% | 5.06 |

Graph ablation conclusion: citation-only is theoretically clean but too sparse in this dataset, averaging only 7.2 edges and 20.2% largest component ratio. Similarity-only gives strong connectivity, but lacks citation direction. Hybrid is the production choice because it keeps citation evidence while improving largest component ratio to 96.9%, path community coverage to 91.4%, and bridge plausibility to 81.9%.

Secondary average excluding the low-corpus protein run shows the same trend: hybrid reaches 97.7% largest component ratio and 90.6% path community coverage.

Similarity backend ablation, with graph mode fixed as hybrid:

| Backend | Topics | Edges | Similarity Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Bridge Plausibility | Foundation Landmark Hit | Edge Yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 12 | 69.8 | 62.6 | 2.2 | 91.7% | 0.2663 | 4.5 | 87.2% | 84.5% | 66.7% | 3.02 |
| lsa | 12 | 70.2 | 63.0 | 2.1 | 93.7% | 0.2462 | 4.7 | 89.7% | 87.4% | 75.0% | 3.01 |

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
- Optional LSA similarity gives Skill 2 an additional robustness knob: it improves several downstream graph/path metrics on the 12-topic backend ablation while remaining fully offline and reproducible.
- Skill 3 is necessary because ranking baselines do not provide stage structure; staged reading paths preserve high topicality while adding pedagogical organization.
- The full system is not just a summarizer: it retrieves papers, builds a research network, assigns structural paper roles, and generates a personalized reading path with visual evidence.

## Limitations to State Explicitly

- The benchmark landmark lists are curated for evaluation. In the system, verified landmarks are recovered through API metadata and should be described as a recall aid, not as manual insertion of final results.
- OpenAlex has broad coverage but lower topic precision on some engineering-heavy topics; arXiv is narrower and often cleaner but weaker on citation/reference metadata.
- Citation metadata remains incomplete for some domains, which is why the hybrid graph is more reliable than citation-only graph construction.
- LSA is an offline embedding-style backend, not a pretrained scientific embedding model. A sentence-transformer backend is supported as an optional extension, but was not used in the reported 12-topic run because the environment did not include the package/model.
- The saved 12-topic overall benchmark has a weak protein structure run with only 8 papers; the Skill 1 rerun shows this topic can retrieve a much healthier corpus, so report the original end-to-end protein result as low-confidence or rerun the full agent if time permits.
- Role scores are still heuristic combinations of network metrics, recency, citation count, and community position. LLM helps explain evidence but does not replace the deterministic scores.

## Suggested Manual Evaluation Dimensions

Use the files in `outputs/reading_path_ablation/full_11_topics/paths_for_human_eval/` and score each path 1-5 on: topical relevance, landmark coverage, reading-order coherence, community/subfield coverage, explanation usefulness, beginner friendliness, and whether bridge papers are plausible.
