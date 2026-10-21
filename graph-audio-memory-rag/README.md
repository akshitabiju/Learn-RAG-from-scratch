# Graph-Augmented Personal Audio Memory RAG

## In plain English

You talk into your phone all day: ideas, to-dos, things people told you. The transcript is messy
("um, so, like, basically..."). This app:

1. **Cleans** the transcript by removing the filler words.
2. **Draws a map of your memories.** Each short passage becomes a dot, and each topic you
   mentioned (Airflow, Priya, hiking trip...) becomes another dot. Lines connect a passage to the
   topics it talks about, and topics that show up together get linked too.
3. **Answers your questions by following the map.** Ask "What did Priya say about Airflow?" and the
   app starts at the *Priya* and *Airflow* dots, walks along the lines to find the closest
   passages, and hands only those passages to Claude. Claude answers from them and cites which
   memory it used. If your memories don't contain the answer, it says so instead of guessing.
4. **Shows you the map** in a clean web page, with the passages it used highlighted.

Everything except the final Claude call runs on your own computer. There are no
LangChain/LlamaIndex shortcuts and no vector database. The logic is plain Python so you can read
every step.

## Technical overview

A from-scratch **graph-based Retrieval-Augmented Generation** pipeline. In place of a flat array
of dense-embedding chunks with nearest-neighbour search, memories are modelled as a heterogeneous
**in-memory property graph (`networkx.Graph`)**. Retrieval is **structural**: a query activates
seed nodes, and relevance is propagated with **personalized PageRank (PPR)** over weighted edges.

```
raw transcript -> strip_fillers() -> make_segments() -> build_graph() -> retrieve() -> ask_claude()
                  (regex)            (sentence windows)  (NetworkX)       (PPR)         (Anthropic SDK)
                                                                     \-----------> draw_graph() (Matplotlib)
```

### 1. Ingestion: `strip_fillers`
**Simple version:** it deletes "um", "uh", "basically" and "y'know" wherever they appear, and
deletes "like" and "you know" only when they are clearly filler. "I like pizza" survives.

**Technical detail:**
- Unconditional filler lexicon (case-insensitive, word-boundary anchored): `u+m+`, `u+h+`, `er+m`,
  `basically`, `y['’]know`. The pattern also consumes adjacent commas and whitespace.
- Polysemous tokens (`like`, `you know`) are removed only under **syntactic delimiting
  constraints**: comma-bracketed (`, like,`), comma-led (`, you know`), or sentence-initial followed
  by a comma. This trades recall for precision and avoids corrupting legitimate usage.
- A normalisation pass repairs whitespace before punctuation, collapses duplicate commas and
  `".,"` artefacts, and re-capitalises sentence starts.
- **Known limitation:** comma-bracketed "like" in genuine "such as" usage ("dishes, like, pasta")
  is over-stripped, and unpunctuated filler "like" is retained. Tune the patterns per speaker.

### 2. Graph index: `build_graph`
**Simple version:** short passages and topics become dots; lines show "this passage mentions that
topic", "these topics appear together", and "this passage came right after that one".

**Technical detail:**

| Element | Semantics |
|---|---|
| `memory` node (`m0...`) | Non-overlapping window of N consecutive sentences |
| `concept` node (`c:<key>`) | Explicit `#hashtags` and user-supplied tags, plus the top-K recurring keywords (doc frequency >= 2, stopword-filtered, suffix-normalised) |
| memory-concept edge | `mentions`; weight = term count (+1 if hashtagged) |
| concept-concept edge | `co-occurs`; weight = number of shared segments |
| memory-memory edge | `next`; constant weight 0.5, preserving temporal adjacency |

Concept keys are normalised by a lightweight suffix stripper (not a full stemmer), so
`dashboards` and `dashboard` resolve to one node.

### 3. Grounded retrieval: `retrieve`
**Simple version:** your question lights up the matching topic dots. The "glow" spreads along the
lines to nearby passages, and the brightest passages are the ones Claude gets to read.

**Technical detail:**
1. Tokenise and normalise the query, then seed concept nodes whose key matches. If none match,
   fall back to lexical overlap against memory nodes.
2. Run `nx.pagerank` with a **personalization vector** concentrated on the seeds (damping `alpha`,
   default 0.85, honoring the `weight` edge attribute).
3. Select the top-k memory nodes by PPR score and re-order them chronologically. Emit a context
   block with node text, adjacent concepts, and `co-occurs` relations incident to the seeds.
4. `ask_claude` sends the context in a delimited `<memory_graph_context>` block. The system prompt
   requires `[Mx]` citations and **abstention** when the context is insufficient.

If no seed is found, retrieval returns an empty result and the app **does not call the model**.
This prevents ungrounded answers.

### 4. User interface
**Simple version:** a web page where you paste text, click a button, ask a question, and see the
answer plus a colour-coded picture of your memory map.

**Technical detail:** Streamlit front end with state in `st.session_state`. Matplotlib renders a
force-directed (`spring_layout`, seeded for deterministic layout) projection of the graph. Squares
are memories and circles are concepts. Node area scales with PPR score. Red marks retrieved
memories and amber marks query seeds.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=<API_TOKEN>                    # or paste it in the sidebar
streamlit run app.py
```

You can try the app without a key. It builds the graph and shows the retrieved context, and it
skips the Claude answer step.

## Privacy and security

**Simple version:** your transcript text and the retrieved passages are sent to Anthropic when you
click "Ask". Don't use recordings containing confidential, customer, or regulated information unless
you are allowed to. Never put your API key in the code or commit it to GitHub.

**Technical detail:**
- The outbound payload is the system prompt, the retrieved context, and the question. The full
  transcript is not sent, only the top-k memory nodes.
- The API key is sourced from the environment or a password-type input and held in process memory
  only. `.env` is git-ignored, and `.env.example` contains placeholders only.
- The graph is process-local and non-persistent. Closing the session discards it.
- Errors from the API call are surfaced by exception type and message, with no key material logged.

## Design trade-offs

- **Keyword concepts:** transparent and dependency-free, but blind to synonyms and paraphrase.
  A natural upgrade is LLM-based concept tagging or a hybrid graph + embedding retriever.
- **Complexity:** PPR runs over the whole graph per query, roughly O(E) per power iteration. That
  is fine for thousands of nodes and unsuitable for millions without sparse or approximate methods.
- **Windowing:** non-overlapping sentence windows keep context compact, while overlapping windows
  would improve recall at the cost of duplicated context.
- **Reproducibility:** the layout seed is a UI setting, and retrieval is deterministic for a fixed
  graph, query and parameters. LLM output is not deterministic.

## Roadmap

- Table-driven unit tests for `strip_fillers` and golden-graph tests for `retrieve`
- Persist the graph with `nx.node_link_data` JSON
- Local ASR (speech-to-text) feeding the same pipeline
- A small labeled question set to measure retrieval recall@k
