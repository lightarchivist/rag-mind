# Changelog

All notable changes to RagMind. Newest first. Versions follow
[semantic versioning](https://semver.org): MAJOR for changes that need action
(for example rebuilding the index), MINOR for new features, PATCH for fixes.
The web app shows the newest entry under **About**.

## [2.2.0] - 2026-11-26

Based on a 14-question test of two models against two real papers, with every quoted number checked against the papers.

### Added
- Numbered source tags that match the [1], [2]… citations in answers. Click a tag in the web app to read the passage the model was given; the CLI lists sources by number.
- Answers that say "not found in the indexed documents" no longer show source tags.

### Changed
- Summaries use each paper's whole abstract (up to the introduction), so the results it reports are included, and up to eight chunks of its conclusion.
- Appendices and reference lists are left out of summaries and comparisons, and reference lists are left out of other answers unless the question asks about references.
- Each retrieved passage is extended with the text that follows it, so details that continue past a chunk boundary are kept.
- New rules for the model: cite passages by number, name every item when asked "which…", say which table row a number comes from, check the question's assumptions, and keep the authors' own work apart from work they cite.

### Fixed
- Conclusions headed like "6. Conclusion, limitations and future work" were not recognised.
- Follow-up questions were not rewritten for gpt-oss (and other models that think before answering), so searches like "who are the authors of the SDN paper?" missed the title page.
- Limiting the web chat to one document (CONTEXT) turned off some retrieval improvements.

## [2.1.0] - 2026-11-26

### Changed
- Summaries ("Summarise the key findings") use each document's abstract and conclusion, marked as such for the model, and skip title-page text such as authors, affiliations and e-mails.
- The earlier conversation is given to the model only for follow-up questions. Complete questions are answered from the documents alone.

### Fixed
- Short questions such as "Summarise the key findings" were rewritten from the previous answer, which could produce a meaningless search (seen with Ministral).
- Claims from earlier answers could reappear in new answers as if they came from the documents.

### Added
- Version number and release notes in the About box, `rag_cli.py --version`, and this changelog.

## [2.0.0] - 2026-11-25

Major update. Indexes built by 1.x must be rebuilt once: run `python rag_cli.py reindex`, or click **Rebuild index** in the web app.

### Added
- Switch between installed Ollama models: menu beside the status light in the web app; `models`, `chat --model` and `/model` in the CLI.
- Answers cite page numbers for PDFs.
- Follow-up questions ("and its limitations?") are rewritten into full questions before searching.
- Questions about all documents ("compare the documents") draw passages from every indexed document.
- Reasoning models: their thinking is hidden, and "thinking…" is shown while they reason.
- `reindex` command and **Rebuild index** button; old or mismatched indexes are detected.
- Settings through environment variables or `~/.ragmind/config.json`; optional LLM grading of passages.
- Offline test suite (no models needed).

### Changed
- The CLI and the web app share one core (`ragmind/`), so fixes apply to both.
- Embeddings use nomic-embed-text's task prefixes and are sent in batches: better search, faster indexing.
- The web index is stored as `index.npz` + `index.json`, saved atomically, instead of `index.pkl`.
- The Docker image is built from the repository root.
- fonttools is installed for better PDF text extraction.

### Fixed
- Re-indexing a changed file could lose its old version if Ollama failed part-way.
- Large books failed to index in a single ChromaDB batch.
- Long conversations could push the instructions and passages out of the model's context window.
- "Model not installed" was reported as "Cannot reach Ollama"; timeouts and mid-answer errors crashed the CLI or went unnoticed.
- An answer cut off by an error was shown as complete and kept in the conversation.
- `clear` crashed on a fresh install and did not ask for confirmation.
- One unreadable file stopped a whole `add` run.
- The CLI ignored `OLLAMA_BASE` and other settings.
- The web page could keep the typing indicator spinning when a reply had no header line.
- Complete questions asked after earlier chat were rewritten and missed the relevant pages.
- Similarity scores leaked into answers.
- Long document names pushed the bottom bar out of view.

## [1.0.0] - 2026-04-28 to 2026-10-29

Original release: CLI with ChromaDB; web app (FastAPI, numpy index) with background indexing, document selector, About box, theme switcher and Docker image.
