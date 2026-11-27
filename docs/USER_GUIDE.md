# RagMind User Guide

RagMind answers questions about your own documents (papers, books, notes) using
AI models that run on your computer. Nothing is sent to the cloud. This guide
covers everyday use of the web app; the README covers installation details and
development.

**Contents**

1. [What RagMind does, and what it doesn't](#1-what-ragmind-does-and-what-it-doesnt)
2. [Starting and stopping](#2-starting-and-stopping)
3. [A tour of the page](#3-a-tour-of-the-page)
4. [Adding documents](#4-adding-documents)
5. [Asking questions](#5-asking-questions)
6. [Checking answers](#6-checking-answers)
7. [Choosing a model](#7-choosing-a-model)
8. [Papers, books and large libraries](#8-papers-books-and-large-libraries)
9. [Settings you may want to change](#9-settings-you-may-want-to-change)
10. [Your data and privacy](#10-your-data-and-privacy)
11. [Updating RagMind](#11-updating-ragmind)
12. [Troubleshooting](#12-troubleshooting)
13. [The command-line version](#13-the-command-line-version)

---

## 1. What RagMind does, and what it doesn't

When you ask a question, RagMind searches your indexed documents for the
passages most likely to contain the answer, gives those passages to a language
model, and asks it to answer **only from them**, citing each passage by number.

- **It answers from your documents, not from the model's general knowledge.**
  If the answer isn't in the passages it found, it says *"This information is
  not found in the indexed documents."* That is the intended behaviour, not an
  error.
- **It reads a handful of passages per question, not whole documents.** It is
  excellent at specific questions ("What is the default SRGB range on IOS XR?")
  and good at summaries of papers. Questions that need a whole book at once
  ("summarise this 900-page book") get a partial answer.
- **Answers can be wrong.** Smaller models sometimes misread tables or mix up
  documents. Every answer shows its sources so you can check it in seconds
  (section 6).
- **It reads text, not pictures.** Figures and diagrams aren't searched, but you
  can view the PDF page next to any passage.

---

## 2. Starting and stopping

RagMind needs two things running: **Ollama** (which runs the AI models) and the
**RagMind container**.

**First time:** install Ollama, then pull a chat model and the search model:

```bash
ollama pull gpt-oss:20b          # or any chat model you prefer
ollama pull nomic-embed-text     # required: used to search your documents
```

On Linux, Ollama must accept connections from Docker: start it with
`OLLAMA_HOST=0.0.0.0`, or set that in Ollama's systemd service. See the README.

**Start:**

```bash
cd ~/GIT/rag-mind/web
docker compose up -d
```

Then open **http://localhost:8000**.

**Stop:**

```bash
cd ~/GIT/rag-mind/web
docker compose stop
```

Ollama unloads models from the graphics card by itself after 5 minutes without
questions, so it can usually be left running.

> **Your documents survive** stopping, restarting, rebooting and updating.
> They're stored in a Docker volume, not in the container. Never use
> `docker compose down -v`: the `-v` deletes that volume, and with it your
> library and index.

---

## 3. A tour of the page

**Top bar**
- **Model menu**: the chat model that answers your questions, from those
  installed in Ollama. The number after each name is how much text it can read
  at once (for example *8K*).
- **Status light**: green when Ollama is reachable; it shows *ollama offline* if
  not.
- **Theme** (☀ ◐ ☾) and **ℹ About**, which shows the version and what changed.

**Left side**
- **Upload Files**: click or drop files here.
- **Library**: every uploaded file. **Index** makes a file searchable;
  **✓ indexed** means it already is. **✕** deletes the file *and* removes it
  from the index.
- **Indexed Documents**: what can currently be searched. **✕** here removes a
  document from the index but keeps the file in the Library.

**Chat**
- **CONTEXT**: search *All indexed documents*, or limit the chat to one document.
- The box at the bottom is where you type. Enter sends; Shift+Enter adds a new
  line.
- The suggestions on the welcome screen ("Summarise the key findings", …) are
  just shortcuts.
- To start a fresh conversation, reload the page. Your documents stay indexed.

---

## 4. Adding documents

**Formats:** PDF, EPUB, DOCX, TXT, MD, RST and HTML.

1. Drop the file on **Upload Files**.
2. Click **Index** next to it in the Library. The button counts up in percent;
   a large book can take several minutes.
3. When it shows **✓ indexed**, you can ask questions about it.

**Automatic clean-up.** While indexing, RagMind repairs common PDF text
problems, so you don't need to do anything:

- eBook watermarks (*From the Library of …*) and download stamps (*Authorized
  licensed use limited to …*) are removed;
- fonts that turn "s" into "−" are repaired;
- ligatures (a single "ﬁ" character) are split into normal letters;
- letter spacing that splits words ("T able") is joined again.

**Titles and sections.** RagMind also reads each document's title (from the
PDF, or from its first page) and the section every passage is in (from the
PDF's bookmarks or, without them, from numbered headings such as "3.2 Data" or
"IV. Results"). You'll see both when you check sources (section 6), and you can
use the title to name a document in a question (section 5). A document without
recognisable headings still works; its passages just have no section.

**Scanned PDFs** (pictures of pages without a text layer) index as empty. Run
them through OCR first.

**Replacing a file with a new version:** delete the old one (**✕** in the
Library), then upload and index the new one. If you keep both, two copies of
the same text compete for every question.

**Re-indexing one document** (for example after an update that reads documents
better), without deleting it:

```bash
curl -X POST "http://localhost:8000/index/<file name>?force=true"
```

Write spaces in the file name as `%20`. The old version stays searchable until
the new one is ready.

**"Rebuild index" banner:** after some updates, or if you change the search
model, RagMind asks for a rebuild. Click **Rebuild index** once. It re-reads all
your stored files; you don't need to upload anything again.

---

## 5. Asking questions

**Be specific.** The best questions name the thing you want:

| Instead of | Ask |
|---|---|
| "Tell me about LDP" | "How does LDP discover neighbours, and which port does it use?" |
| "What does the table say?" | "What accuracy did Llama 3.1 8B reach on SWaT?" |
| "Explain the paper" | "Summarise the key findings" |

**Summaries and comparisons.** "Summarise the key findings", "Compare the
documents" or "How do the two papers differ…" draw on **every** indexed
document. For papers, RagMind uses each paper's abstract and conclusion; for
books it uses their most relevant passages. A note under the answer says
*passages from all N documents*.

**One document at a time.** Choose it in **CONTEXT**. This is the best way to
study one book, because similar books can't compete with it. Remember to set
CONTEXT back to *All indexed documents* afterwards.

**Naming a document** keeps the search in that document: "Who are the authors
of the SDN paper?", "What threshold did the anomaly detection study use?", "What
does the Secure Firewall guide say about…". Use words from its title or file
name followed by *paper*, *book*, *guide*, *study*, *article*, *report*,
*survey*, *framework* or *system*. A name that fits several documents ("the
cert guide" in a shelf of cert guides) is ignored, so add a distinguishing
word. Questions about authors and affiliations include the title page of the
document you name.

**Comparing named documents.** "How do the anomaly detection paper and the
SDN paper differ in…" gets passages from each of the two, and the
answer keeps them apart.

**Depth and breadth.** "Explain how … works" or "… in detail" reads further
into the section where the answer starts. "Which chapters discuss …" or "Where
does the book cover …" collects passages from many sections instead, at most
two from each, and the answer names the sections.

**Follow-up questions** work: after "Which model performed best?", you can ask
"And how did the smallest model compare?" or "Explain the second step in more
detail". RagMind rewrites the follow-up into a complete question, searches for
it, also looks for the passages the previous answer was based on, and gives the
model both your question and the rewritten one. The rewritten question is shown
under the answer as *searched for: …*. If that line shows a misunderstanding,
ask the full question instead.

**"Not found" answers.** If you're sure the information is there, try the
document's own wording (for example "hello messages" rather than "discovery"),
narrow the question, or set CONTEXT to the document.

---

## 6. Checking answers

Every answer ends with **numbered source tags**: file, page and section, such
as *[2] paper.pdf, p. 4 · 3.2 Training data*. **[2]** in the answer is the tag
marked **[2]**. Passages from the same page and section share a tag, such as
**[1, 3, 5]**.

- **Click a tag** to read the exact passages the model was given. Above them
  you'll see the document's title and the full section path, for example
  *"Title" › 3. Method › 3.2 Training data*.
- **Show page** under a passage displays the PDF page itself, with its figures,
  diagrams and tables as printed. Click the image to open it full size.
- **Page numbers are PDF pages**, not the numbers printed in a book. In books
  they differ by the front matter (title, contents, introduction).
- Passages that continue a table start with *[The table rows in this passage …
  are from Table N: …]*, so numbers aren't attributed to the wrong table.

A quick habit for numbers and quotes: click the tag, check the passage, and use
**Show page** for anything in a table.

---

## 7. Choosing a model

Pick a model in the menu at the top. Any chat model installed in Ollama works;
`nomic-embed-text` is used for searching and doesn't appear in the menu.

From tests on research papers and Cisco Press books:

| Model | Strengths | Weaknesses |
|---|---|---|
| **gpt-oss:20b** | Most accurate; reads tables carefully; says "not found" rather than inventing | Slower (often 5–20 s per answer); occasionally returns no answer |
| **Foundation-Sec-8B-Instruct** | About three times faster; good for security topics | More often misreads tables and mixes up documents |

Some general points:

- **Reasoning models** (gpt-oss, Qwen3, DeepSeek-R1…) think before answering.
  RagMind hides the thinking and shows *thinking…* while it happens.
- If a model **returns no answer**, RagMind shows an *[ERROR: …]* message
  instead of a blank reply. Ask again, or ask a narrower question.
- **Bigger isn't always better** for RAG: a careful mid-size model that sticks to
  the passages beats a large one that answers from memory.
- To change the default model, set `LLM_MODEL` (section 9).

---

## 8. Papers, books and large libraries

**Papers** work best: RagMind recognises their abstract, conclusion, appendices
and reference list, so summaries draw on the real results and ignore the
bibliography.

**Books** are treated differently: they have no abstract, so summaries use their
most relevant passages instead of the first pages (which are copyright notices,
contents and publisher offers). Ask books specific questions, or ask for a
summary *of a topic*: "Summarise what the book says about segment routing".

**Large libraries.** With several books on related subjects (for example three
networking guides), similar chapters compete for the same question. If an answer
comes from the wrong book, name the book in the question ("in the SPCOR
guide…") or set CONTEXT to it.

**Sections in books** come from the PDF's bookmarks, so a passage shows where it
is, for example *Chapter 5 › Label Distribution Protocol*. Ask "Which chapters
discuss…" to find every place a topic comes up.

---

## 9. Settings you may want to change

Settings are environment variables. For your own machine, put them in
`web/docker-compose.override.yml` (it isn't part of the repository, so updates
won't overwrite it):

```yaml
services:
  ragmind:
    environment:
      - LLM_MODEL=gpt-oss:20b       # default chat model
      - RAGMIND_MAX_CTX=16384       # more room for long answers (uses more GPU memory)
```

Then run `docker compose up -d` again.

| Setting | Default | When to change it |
|---|---|---|
| `LLM_MODEL` | `llama3` | To start with your preferred model |
| `RAGMIND_MAX_CTX` | `8192` | Summaries of many documents get thin: try `16384` if your GPU has room |
| `RAGMIND_TOP_K` | `6` | More passages per answer (slower, more context) |
| `RAGMIND_MIN_SCORE` | `0.25` | Lower it if too many questions come back "not found" |
| `RAGMIND_KEYWORDS` | `1` | `0` turns off keyword search, leaving similarity search only |
| `RAGMIND_REWRITE` | `1` | `0` turns off follow-up rewriting |

The full list is in the README.

---

## 10. Your data and privacy

- **Everything stays on your computer.** Documents, the index and the
  conversation never leave it; models run in Ollama locally.
- **Where it's stored:** in the Docker volume `web_ragmind_data` (with
  `docker compose`) or `ragmind_data` (with the `docker run` command from the
  README). Uploaded files, the index and cached page images are all there.
- **Watermarks and download stamps** are removed from the indexed text, so names
  in them don't reach answers or prompts. **Show page** draws the original PDF,
  so they are still visible on page images, as on paper.
- **Deleting a file** removes it, its passages and its page images.
- **There is no login.** Anyone on your network who can open
  `http://<this computer>:8000` can read, upload and delete documents. Use it on a
  network you trust, or block port 8000 for other machines in your firewall.

**Back up** your library (saves `ragmind-backup.tgz` in the current folder):

```bash
docker run --rm -v web_ragmind_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/ragmind-backup.tgz -C /data .
```

**Restore** it, with RagMind stopped:

```bash
docker run --rm -v web_ragmind_data:/data -v "$PWD":/backup alpine \
  sh -c "cd /data && tar xzf /backup/ragmind-backup.tgz"
```

---

## 11. Updating RagMind

```bash
cd ~/GIT/rag-mind
git pull
cd web && docker compose up --build -d
```

Or, if you use the Docker Hub image: `docker pull lightarchivist/rag-mind:latest`
and recreate the container with the same volume. Your documents stay. **ℹ About**
shows the running version and what changed. If an update needs it, a **Rebuild
index** banner appears.

---

## 12. Troubleshooting

| Problem | What to do |
|---|---|
| *ollama offline* at the top | Start Ollama. On Linux, make sure it listens on `0.0.0.0` (section 2). |
| *Model '…' is not installed in Ollama* | `ollama pull <model>`, then reload the page. |
| A file indexes with *No text found* | It's a scanned PDF: run OCR first. |
| *This information is not found…* but you know it's there | Use the document's own wording, narrow the question, or set CONTEXT (section 5). |
| The answer quotes the wrong book or paper | Set CONTEXT to the right document, or name it in the question. |
| Numbers look wrong | Click the source tag and use **Show page** to check the table. Try a more careful model. |
| *The model returned no answer* | Ask again, or ask a narrower question. |
| Answers are slow | Normal for large or reasoning models; try a smaller model. The first question after a pause also loads the model. |
| *Rebuild index* banner | Click it once and wait; your files are re-read automatically. |
| The page looks out of date after an update | Hard refresh: Ctrl+Shift+R. |

---

## 13. The command-line version

The same engine works in a terminal, without the browser (see the README for
installing it):

```bash
python rag_cli.py add paper.pdf book.epub     # index files
python rag_cli.py add --force book.pdf        # re-index an unchanged file
python rag_cli.py folder ~/Papers -r          # index a folder and its sub-folders
python rag_cli.py list                        # what's indexed
python rag_cli.py chat                        # ask questions
python rag_cli.py chat -m gpt-oss:20b         # … with a specific model
python rag_cli.py models                      # installed chat models
python rag_cli.py reindex                     # rebuild after an upgrade
```

In `chat`: `/model <name>` switches model, `/sources` lists the indexed
documents, `/history` shows the conversation, `/reset` forgets it, and `/quit`
exits.

The command-line index is separate from the web app's: documents added in one
aren't visible in the other.
