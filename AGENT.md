# ResearchTrail Agent Instructions

Use these instructions when an external coding or writing agent, such as Claude Code, OpenClaw, or Codex, operates on this repository or post-processes ResearchTrail reports.

## Project Role

ResearchTrail is a markdown-described, code-backed research navigation agent. The Python pipeline should perform retrieval, graph construction, scoring, and report generation. External agents may improve presentation and personalization, but should not invent papers, citations, graph scores, or paper roles.

## Personalization Layer

If `MEMORY.md` exists, read it before rewriting or polishing a generated report. Use it to adapt wording, pacing, examples, and study-plan format to the user's background.

Allowed personalization:

- Adjust explanation depth for the user's background.
- Rephrase "Why read" sections in a style that matches the user's goals.
- Add brief prerequisite reminders when `MEMORY.md` indicates the user is a beginner.
- Add optional study-plan framing, such as 3-day, 7-day, or exam-oriented plans, if consistent with the report.
- Emphasize application areas the user cares about.

Not allowed:

- Do not add new papers unless the user explicitly asks for a new retrieval run.
- Do not change citation counts, graph metrics, communities, or role scores.
- Do not claim a paper says something unsupported by its title, abstract, or existing report evidence.
- Do not remove URLs, abstracts, or citation source labels.
- Do not hide limitations such as sparse citation metadata or low-confidence corpora.

## Recommended Post-Processing Workflow

1. Read `MEMORY.md` if present.
2. Read the generated `research_report.md`.
3. Preserve factual sections: corpus summary, graph metrics, paper metadata, URLs, abstracts, and score tables.
4. Rewrite only narrative sections: overview, why-read explanations, stage introductions, and study plan.
5. Add a short "Personalized Notes" section if useful.
6. Save the polished report as a separate file, for example `research_report_personalized.md`.

## Evidence Discipline

Treat ResearchTrail's generated report and `state.json` as the evidence source. If the report and `state.json` disagree, prefer `state.json` for paper metadata and graph metrics.
