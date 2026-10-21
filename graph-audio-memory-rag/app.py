"""Graph-Augmented Personal Audio Memory RAG (pure Python, no LangChain/LlamaIndex).

Pipeline: raw transcript -> filler stripping (regex) -> memory nodes + concept nodes in a
NetworkX graph -> personalized-PageRank retrieval -> grounded Claude answer -> Matplotlib view.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import matplotlib

matplotlib.use("Agg")  # headless backend; Streamlit renders the figure
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

DEFAULT_MODEL = "claude-sonnet-5-5"

SAMPLE_TRANSCRIPT = """\
So, um, today I finally decided to move the #analytics pipeline from cron jobs to Airflow. \
Basically, the nightly cron jobs kept failing silently and nobody noticed for days. \
Uh, I talked to Priya about it and she said the Airflow migration would take, like, two sprints. \
Y'know, the biggest risk is the #warehouse schema because the dashboards depend on it. \
I want to, um, freeze the warehouse schema until the migration is done. \
Later I had lunch and, uh, thought about the #hiking trip with Marco in the Alps next June. \
We need to book the mountain hut early, basically by February, because the huts fill up fast. \
Also, um, I should ask Priya whether the Airflow migration can include alerting for the dashboards.
"""

STOPWORDS = frozenset(
    """a about after all also am an and any are as at be because been before being but by can could
    did do does done for from had has have he her here him his how i if in into is it its just me
    more my need no not now of on one or our out she so some than that the their them then there
    these they this to too up us want was we were what when where which who why will with would
    you your should today later really very thing things""".split()
)


# --------------------------------------------------------------------------- #
# 1. Ingestion engine
# --------------------------------------------------------------------------- #
# Always-filler tokens: um/umm, uh/uhh, erm, basically, y'know. Eats adjacent commas.
_FILLER_CORE = re.compile(r"\s*,?\s*\b(?:u+m+|u+h+|er+m|basically|y['’]know)\b\s*,?\s*", re.I)
# Ambiguous tokens ("like", "you know") are only removed when comma-delimited, so
# "I like pizza" and "do you know?" survive but "it was, like, huge" does not.
_FILLER_LIKE = re.compile(r",\s*like\s*,", re.I)
_FILLER_YOUKNOW = re.compile(r"\s*,\s*you\s+know\b(?:\s*,)?", re.I)
_FILLER_SENT_START = re.compile(r"(?:^|(?<=[.!?]\s))(?:like|you\s+know),\s*", re.I)


def strip_fillers(text: str) -> tuple[str, int]:
    """Remove conversational filler and tidy punctuation. Returns (clean_text, n_removed)."""
    removed = 0
    for pattern, repl in (
        (_FILLER_SENT_START, ""),
        (_FILLER_LIKE, " "),
        (_FILLER_YOUKNOW, " "),
        (_FILLER_CORE, " "),
    ):
        text, n = pattern.subn(repl, text)
        removed += n
    text = re.sub(r"\s+([,.;!?])", r"\1", text)       # "word ." -> "word."
    text = re.sub(r",\s*,+", ",", text)                # ",," -> ","
    text = re.sub(r"([.!?])\s*,", r"\1", text)         # ".," -> "."
    text = re.sub(r"^[,\s]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text, removed


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def make_segments(text: str, window: int) -> list[str]:
    """Group `window` consecutive sentences into one memory node (non-overlapping)."""
    sents = split_sentences(text)
    return [" ".join(sents[i : i + window]) for i in range(0, len(sents), max(1, window))]


# --------------------------------------------------------------------------- #
# 2. In-memory graph index
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]+", text.lower())


def normalize(tok: str) -> str:
    """Tiny suffix-stripping normalizer so 'dashboards' and 'dashboard' share a concept."""
    tok = tok.removesuffix("'s")
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def content_keys(text: str) -> list[str]:
    return [normalize(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def build_graph(
    sanitized: str, extra_tags: list[str], window: int = 2, max_auto_concepts: int = 12
) -> nx.Graph:
    """Bipartite-plus graph: memory nodes <-> concept nodes, with temporal + co-occurrence edges."""
    segments = make_segments(sanitized, window)
    seg_hashtags = [{normalize(t.lower()) for t in re.findall(r"#([A-Za-z][\w-]*)", s)} for s in segments]
    seg_text = [re.sub(r"#(\w)", r"\1", s) for s in segments]  # display text without '#'

    explicit = {normalize(t.strip().lstrip("#").lower()) for t in extra_tags if t.strip()}
    explicit |= set().union(*seg_hashtags) if seg_hashtags else set()

    freq = Counter(k for s in seg_text for k in content_keys(s))
    auto = {k for k, c in freq.most_common(max_auto_concepts * 2) if c >= 2}
    auto = set(sorted(auto, key=lambda k: (-freq[k], -len(k)))[:max_auto_concepts])
    concepts = explicit | auto

    G = nx.Graph()
    for i, text in enumerate(seg_text):
        G.add_node(f"m{i}", kind="memory", label=f"M{i + 1}", text=text)
    for key in concepts:
        G.add_node(f"c:{key}", kind="concept", label=key, explicit=key in explicit)

    for i, text in enumerate(seg_text):
        counts = Counter(content_keys(text))
        present = [k for k in concepts if counts[k] > 0 or k in seg_hashtags[i]]
        for k in present:
            w = counts[k] + (1 if k in seg_hashtags[i] else 0)
            G.add_edge(f"m{i}", f"c:{k}", weight=float(max(w, 1)), rel="mentions")
        for a, b in combinations(sorted(present), 2):  # concept-concept co-occurrence
            u, v = f"c:{a}", f"c:{b}"
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u, v, weight=1.0, rel="co-occurs")
        if i > 0:  # temporal adjacency keeps the narrative thread
            G.add_edge(f"m{i - 1}", f"m{i}", weight=0.5, rel="next")
    return G


# --------------------------------------------------------------------------- #
# 3. Grounded retrieval
# --------------------------------------------------------------------------- #
@dataclass
class Retrieval:
    scores: dict[str, float] = field(default_factory=dict)  # all nodes
    memory_ids: list[str] = field(default_factory=list)     # top-k, chronological
    seed_concepts: list[str] = field(default_factory=list)
    context: str = ""


def retrieve(G: nx.Graph, query: str, top_k: int = 4, alpha: float = 0.85) -> Retrieval:
    """Seed the graph with query concepts, spread relevance with personalized PageRank."""
    qkeys = set(content_keys(query))
    seeds = {n: 1.0 for n, d in G.nodes(data=True) if d["kind"] == "concept" and d["label"] in qkeys}
    if not seeds:  # fallback: lexical overlap against memory text
        for n, d in G.nodes(data=True):
            if d["kind"] == "memory":
                overlap = len(qkeys & set(content_keys(d["text"])))
                if overlap:
                    seeds[n] = float(overlap)
    if not seeds:
        return Retrieval()

    pers = {n: seeds.get(n, 0.0) for n in G}
    scores = nx.pagerank(G, alpha=alpha, personalization=pers, weight="weight")
    mem = sorted((n for n in G if G.nodes[n]["kind"] == "memory"), key=lambda n: -scores[n])[:top_k]
    mem_sorted = sorted(mem, key=lambda n: int(n[1:]))  # chronological for the prompt

    lines = []
    for n in mem_sorted:
        tags = sorted(G.nodes[c]["label"] for c in G.neighbors(n) if G.nodes[c]["kind"] == "concept")
        lines.append(f"[{G.nodes[n]['label']}] (concepts: {', '.join(tags) or 'none'}) {G.nodes[n]['text']}")
    seed_labels = sorted(G.nodes[n]["label"] for n in seeds)
    rel = []
    for u, v, d in G.edges(data=True):
        if d["rel"] == "co-occurs" and (u in seeds or v in seeds):
            rel.append(f"{G.nodes[u]['label']} <-co-occurs x{int(d['weight'])}-> {G.nodes[v]['label']}")
    context = "MEMORY NODES:\n" + "\n".join(lines)
    if rel:
        context += "\n\nGRAPH RELATIONS:\n" + "\n".join(sorted(rel)[:15])
    return Retrieval(scores, mem_sorted, seed_labels, context)


SYSTEM_PROMPT = (
    "You answer questions about the user's personal voice memos using ONLY the supplied memory "
    "graph context. Cite memory nodes like [M2]. If the context does not contain the answer, say "
    "so plainly; never invent details. Be concise."
)


def ask_claude(api_key: str, model: str, context: str, question: str) -> str:
    import anthropic  # imported lazily so the UI loads without a key

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<memory_graph_context>\n{context}\n</memory_graph_context>\n\nQuestion: {question}",
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- #
# 4. Visualization
# --------------------------------------------------------------------------- #
def draw_graph(G: nx.Graph, ret: Retrieval | None = None, seed: int = 42):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_axis_off()
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.2 / max(len(G) ** 0.5, 1))
    hits = set(ret.memory_ids) if ret else set()
    seeds = {f"c:{s}" for s in ret.seed_concepts} if ret else set()
    scores = ret.scores if ret and ret.scores else {}

    widths = [0.6 + 0.8 * d["weight"] for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#c9ced6", alpha=0.8)

    def size(n, base):
        return base + (scores.get(n, 0) * 6000 if scores else 0)

    mem = [n for n in G if G.nodes[n]["kind"] == "memory"]
    con = [n for n in G if G.nodes[n]["kind"] == "concept"]
    nx.draw_networkx_nodes(G, pos, nodelist=con, ax=ax, node_shape="o",
                           node_size=[size(n, 450) for n in con],
                           node_color=["#f59e0b" if n in seeds else "#8b5cf6" for n in con],
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=mem, ax=ax, node_shape="s",
                           node_size=[size(n, 600) for n in mem],
                           node_color=["#ef4444" if n in hits else "#3b82f6" for n in mem],
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, {n: G.nodes[n]["label"] for n in G}, ax=ax,
                            font_size=8, font_color="#111827")
    ax.set_title("Memory graph  |  squares = memory, circles = concept  |  red = retrieved, amber = query seed",
                 fontsize=10, color="#374151")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Streamlit UI
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Graph Audio Memory RAG", page_icon="🧠", layout="wide")
    st.title("🧠 Graph-Augmented Personal Audio Memory")
    st.caption("Voice transcript → filler stripping → knowledge graph → graph retrieval → Claude")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Anthropic API key", value=os.getenv("ANTHROPIC_API_KEY", ""),
                                type="password", help="Held in memory only; never written to disk.")
        model = st.text_input("Model", value=DEFAULT_MODEL)
        window = st.slider("Sentences per memory node", 1, 4, 2)
        top_k = st.slider("Memory nodes retrieved", 1, 8, 4)
        alpha = st.slider("PageRank damping (alpha)", 0.5, 0.99, 0.85)
        layout_seed = st.number_input("Layout seed", value=42, step=1)
        st.warning("Transcripts and retrieved context are sent to the Anthropic API. "
                   "Do not ingest confidential or regulated data without approval.")

    left, right = st.columns(2)
    with left:
        raw = st.text_area("Raw transcription", SAMPLE_TRANSCRIPT, height=260)
        tags_in = st.text_input("Explicit concept tags (comma-separated, optional)", "")
        if st.button("Build memory graph", type="primary"):
            clean, n_removed = strip_fillers(raw)
            extra = [t for t in tags_in.split(",") if t.strip()]
            st.session_state["state"] = {
                "clean": clean, "removed": n_removed,
                "graph": build_graph(clean, extra, window=window),
            }
    state = st.session_state.get("state")
    if not state:
        st.info("Paste a transcript and click **Build memory graph**.")
        return

    G: nx.Graph = state["graph"]
    with right:
        st.text_area("Sanitized text", state["clean"], height=260, disabled=True)
        n_mem = sum(1 for _, d in G.nodes(data=True) if d["kind"] == "memory")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fillers removed", state["removed"])
        c2.metric("Memory nodes", n_mem)
        c3.metric("Concept nodes", len(G) - n_mem)
        c4.metric("Edges", G.number_of_edges())

    st.divider()
    question = st.text_input("Ask your memory", "What did Priya say about the Airflow migration?")
    ret = None
    if st.button("Ask") and question.strip():
        ret = retrieve(G, question, top_k=top_k, alpha=alpha)
        if not ret.memory_ids:
            st.warning("No graph nodes matched the query; nothing to ground an answer on.")
        else:
            with st.expander("Retrieved graph context", expanded=True):
                st.code(ret.context, language="text")
            if not api_key:
                st.info("Add an Anthropic API key in the sidebar to generate an answer.")
            else:
                try:
                    with st.spinner("Asking Claude..."):
                        st.markdown("### Answer\n" + ask_claude(api_key, model, ret.context, question))
                except Exception as exc:  # surface API/network errors without leaking the key
                    st.error(f"Claude API call failed: {type(exc).__name__}: {exc}")

    fig = draw_graph(G, ret, seed=int(layout_seed))
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
