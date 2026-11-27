# Changelog

All notable changes to RagMind. Newest first. Versions follow
[semantic versioning](https://semver.org): MAJOR for changes that need action
(for example rebuilding the index), MINOR for new features, PATCH for fixes.
The web app shows the newest entry under **About**.

## [3.0.7] - 2026-11-27

### Fixed
- A two-part question about one named document ("How was X split, and which settings did the Y system use?") could miss the passage on one of its parts. Inside one named document, passages found by both similarity and keyword search come first again, as in 3.0.3 and as across the whole library. The order by meaning from 3.0.5 now applies only to questions that compare documents.

## [3.0.6] - 2026-11-27

### Fixed
- A question that names one document is searched with its wording as asked again, the document's name included: passages are embedded with their document's title and section, and the name's words help find the right one. Only a question that compares documents leaves out names, and then only the other documents' names when searching each one.

## [3.0.5] - 2026-11-27

### Fixed
- A short question about a named document ("How was the X study set up?") was searched as "How was set up?" once the document's name was taken out. The name now stays when too little of the question is left without it.
- Inside a named document, passages that matched the question's words were put before those that matched its meaning, so for "How was X chunked, and which retrieval settings…" the settings passages pushed out the one on chunking. Inside a named document the order now follows meaning, and word matches are added after.
- "Explain the second step" is now read from the list in the previous answer ("the second step" → its second item), instead of depending on the model to rewrite it. The model only rewrites follow-ups that this doesn't cover.

## [3.0.4] - 2026-11-27

### Fixed
- A question that names documents searches each of them for its topic, without the names. Comparing two papers' safeguards against hallucinations found the passages of one paper that mention the other paper's name, not its safeguards.
- A question that names two documents gets four passages from each, up from three.
- "Explain the second step in more detail" could be answered about the second item of a different list. A rewritten follow-up is now given to the model next to the question as asked, so the answer is about what was searched for.

## [3.0.3] - 2026-11-27

### Fixed
- Networking questions such as "Does uRPF check both source and destination addresses?" or "Do all sources need IGMPv3?" were taken as questions about all the documents and answered from every document's summary. "Source" and "text" no longer count as words for a document.
- "Summarise segment routing" and "Give me an overview of LDP" summarised every document instead of the topic. A short request for a summary covers all the documents only when it names no topic.
- Questions with "across the network", "throughout the session" or "Where is the IP address configured?" were treated as "where do the documents cover this?", which spreads passages over many sections. Only questions about where the documents cover a topic are treated that way now.
- "Is there a default…?", "labels above 15", "the previous hop" and "the same label" made complete questions count as follow-ups, so they were rewritten from the conversation.
- "The Cisco systems" could limit the search to the documents with "Cisco" in their title.
- A follow-up to a "not found" answer also searched with that answer.
- The repair for letter spacing ("T able") could join real text in the same document: "V is the set of nodes" became "Vis the set of nodes". Letters are now joined only when the result is a word.
- One damaged page made a whole PDF unreadable. The page is now skipped.
- PDFs whose bookmarks are not in page order (an "Index" bookmark first) got wrong sections.
- Ligatures in bookmark titles ("Speciﬁcation") are replaced with plain letters.
- eBook watermarks with accented names or lowercase particles ("van der") were not removed.
- Rebuild index stopped with an error when one file could no longer be read. That document now keeps its stored text.
- Deleting a file while it was being indexed put it back in the index when indexing finished. Deleting is now refused until then.
- The Index button stayed at "0%" when the server refused to index (for example during a rebuild).
- The CONTEXT list could close by itself while open, because it was rebuilt every 10 seconds.
- An index whose two files were saved only halfway (a crash during a save) is detected and asks for a rebuild instead of giving wrong results.

## [3.0.2] - 2026-11-26

### Fixed
- "How many … are there?" was answered "not found" when the documents list the items without giving a total. The answer now lists the ones the passages name and counts them.
- A question such as "Tell me about route types in EVPN what they do" was treated as a follow-up, because of "they", and rewritten from the previous answer. A pronoun after a question word that follows a complete first part now counts as pointing within the question.
- Conversational words ("I would like to understand…", "please help") no longer count as keywords, so they don't bring in unrelated passages.

## [3.0.1] - 2026-11-26

### Fixed
- The eBook watermark "From the Library of <name>" stayed in the text of some books: those whose PDF puts no-break spaces or a line break inside it, and those where a running header follows the name on most pages. It is now removed in all these cases. Re-index affected books once with `POST /index/<file>?force=true` (web) or `rag_cli.py add --force` (CLI).
- In a chat limited to one document (CONTEXT), a short question such as "What are the main limitations?" was taken as a follow-up to the previous answer and rewritten into a different question. It is now answered as asked, about that document.
- When a question names a document, the model is now told to answer in that document's own terms, and to say so when the question uses a term from another document.

## [3.0.0] - 2026-11-26

**Rebuild the index once after updating**: click **Rebuild index** in the web app, or run `python rag_cli.py reindex`. Your files are re-read; nothing needs uploading again.

### Added
- Sections. Every passage now knows the section it is in, taken from the PDF's bookmarks or, without them, from the numbered headings in the text. The section is shown on the source tags and above each passage, and the model sees it too.
- Document titles. Each document's title is read from the PDF, or from its first page. A question can name a document by words from its title, even when the file name doesn't contain them ("the anomaly detection paper"), and the search then stays in that document.
- A question that names two or more documents ("How do the X paper and the Y paper differ…") gets passages from each of them, instead of whichever document happens to match best.
- "Explain how …" and "… in detail" questions read further into the section of the best passage.
- "Which chapters discuss …" and "Where does the book cover …" questions get passages from more sections, at most two from any one.
- Follow-up questions ("Explain the second step in more detail") also search with the previous answer, so they find the passages that answer was based on.

### Changed
- The section heading and the document title are embedded with each passage, so a passage that says "the proposed method" is also found by what its section and document are about. This is why the index must be rebuilt.
- Keyword search treats word forms as one word: "chunked", "chunks" and "chunking" match each other.

### Fixed
- A question whose second half refers to its first ("What are the three signals in the framework, and how are they combined?") was treated as a follow-up and rewritten from the conversation.
- "Summarise the X paper" summarised every indexed document.

## [2.4.4] - 2026-11-26

### Fixed
- eBook watermarks that show only "From the Library of <name>", without the ptg number, stayed in the text, including the owner's name. A name that follows "From the Library of" on many pages of a document is now removed; a single occurrence in ordinary text is left alone. Re-index affected books once.

## [2.4.3] - 2026-11-26

### Fixed
- Typographic ligatures ("efﬁciency" written with a single "ﬁ" character) split words in keyword search. They are now replaced with plain letters.
- Letter spacing that splits capitals from their words ("T able 1", "T o ensure", "V ery") is repaired in documents where it happens throughout.
- Small-capital section headings in IEEE papers ("VIII. C ONCLUSION") were not recognised, so summaries missed the conclusion.
- Table captions written without a space ("Table 1.Comparative…") are recognised.
- The IEEE Xplore download stamp ("Authorized licensed use limited to: … Restrictions apply."), which names the downloading institution, is removed from every page.
- A model that returned nothing (a reasoning model can spend all its room on thinking) left a blank reply. It is now reported as an error and kept out of the conversation.

Re-index affected documents once with `POST /index/<file>?force=true` (web) or `rag_cli.py add --force` (CLI).

## [2.4.2] - 2026-11-26

### Fixed
- Some PDFs use a font that maps the letter "s" to a minus sign, so their text was indexed as "the default label di−tribution protocol". Such passages could hardly be found, by keyword or by similarity. Documents where this happens throughout are now repaired when indexed; real minus signs are left alone. Re-index affected documents once (see below).
- The buyer's watermark on eBooks ("From the Library of … ptg…") is removed from every page.

### Added
- Re-index one document without deleting it: `POST /index/<file>?force=true` in the web app, `rag_cli.py add --force FILE` in the CLI.

## [2.4.1] - 2026-11-26

### Fixed
- Keyword search could push the right passage out: a single rare word in the question (such as "reserved") brought in unrelated passages ("All rights reserved", "Reserved Instances") that took the place of good similarity results. Keyword search now only adds to the similarity results (at most two passages they missed), and a passage must match at least two distinctive words or phrases of the question.
- Two-word phrases such as "label 3", "UDP 646" or "type 2" are now searched as phrases.

## [2.4.0] - 2026-11-26

### Added
- Hybrid search: similarity search is combined with keyword (BM25) search, so passages containing the question's exact terms ("label 3", "UDP 646", "Adj-SID") are found even when a passage from a neighbouring chapter or book is more similar in topic. Turn it off with `RAGMIND_KEYWORDS=0`. No re-indexing needed.

### Fixed
- Summaries of collections that include books described the books by their first pages (copyright, contents, companion-website and offer pages). Long documents without an abstract are now treated as books: no "abstract" or "conclusion" passages are taken from them, and publisher pages are skipped.

## [2.3.1] - 2026-11-26

### Fixed
- Complete questions containing "that" as a joining word ("What does X provide that Y does not?") were treated as follow-ups: rewritten, and answered with the earlier conversation. "That" now counts only when it points back ("explain that", "is that true?", "that one").

## [2.3.0] - 2026-11-26

### Added
- View the PDF page behind a source: open a source tag and click **Show page** to see the page with its figures and diagrams, which text extraction leaves out. Click the image to open it full size. Pages are drawn on demand and cached.

### Fixed
- A follow-up such as "and the smallest one?" could be rewritten with a name taken from the previous answer, sending the search to the wrong place. Descriptions are now kept as asked.
- Table labels recognise chapter-numbered tables in books ("Table 5-3"), and command output with IP addresses is no longer mistaken for a table.

## [2.2.1] - 2026-11-26

### Fixed
- Numbers from one table could be read as belonging to the next one when a passage ran across a table caption. Passages that continue a table now name that table.
- Passages that start mid-sentence could lose who did the work, such as a reference to cited research. The sentence or two before a passage are now included.
- "Who are the authors?" questions often found nothing. They now include the title page of the document named in the question.
- Citations written as 【2】 are shown as [2].

### Changed
- Passages from the same document, page and section share one source tag, e.g. [1, 3, 5].

## [2.2.0] - 2026-11-26

### Added
- Numbered source tags that match the [1], [2]… citations in answers. Click a tag in the web app to see the passage the model read; the CLI lists sources by number.
- Answers that say "not found in the indexed documents" no longer show source tags.

### Changed
- Summaries use each document's whole abstract and more of its conclusion.
- Appendices and reference lists are left out of summaries, and reference lists out of other answers unless asked about.
- Retrieved passages include the text that follows them, so details split across chunks are kept.
- Stricter answer rules: cite by number, list every item asked for, name the table row behind a number, check the question's assumptions, separate the authors' work from work they cite.

### Fixed
- Some numbered conclusion headings were not recognised.
- Follow-up questions were not rewritten for models that think before answering.
- Limiting the chat to one document turned off some retrieval improvements.

## [2.1.0] - 2026-11-26

### Changed
- Summaries ("Summarise the key findings") use each document's abstract and conclusion, marked as such for the model, and skip title-page text such as authors, affiliations and e-mails.
- The earlier conversation is given to the model only for follow-up questions. Complete questions are answered from the documents alone.

### Fixed
- Short questions such as "Summarise the key findings" were rewritten from the previous answer and could produce a meaningless search.
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
