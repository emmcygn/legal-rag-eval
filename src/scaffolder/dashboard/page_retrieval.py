"""Page 2: Retrieval demo — query against chunked & embedded documents."""

from __future__ import annotations

from typing import Any

import streamlit as st

from scaffolder.models import StrategyName

FIXTURE_OPTIONS = [
    "uk_service_agreement",
    "uk_terms_conditions",
    "us_msa",
    "us_terms_of_service",
    "eu_gdpr_excerpt",
]

MODEL_OPTIONS = ["all-MiniLM-L6-v2", "bge-base-en-v1.5"]


def _load_annotated_queries(document_id: str) -> list[dict[str, Any]]:
    """Load annotated queries for a specific document."""
    from scaffolder.queries import load_queries_for_document

    queries = load_queries_for_document("queries", document_id)
    return [
        {
            "id": q.id,
            "text": q.text,
            "failure_mode": q.failure_mode,
            "relevant_sections": [
                {"section_id": s.section_id, "relevance": s.relevance, "description": s.description}
                for s in q.relevant_sections
            ],
        }
        for q in queries
    ]


def _check_retrieval_available() -> bool:
    """Check if the retrieval pipeline is available."""
    try:
        from scaffolder.retrieval import simulator  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def render_page() -> None:
    """Render the retrieval demo page."""
    st.title("Retrieval Demo")
    st.markdown("Ask a legal question and compare retrieval results across chunking strategies.")

    if not _check_retrieval_available():
        st.warning(
            "Retrieval requires embedding dependencies. Install with: `pip install -e '.[local]'`"
        )
        _render_query_browser()
        return

    # --- Settings ---
    col1, col2, col3 = st.columns(3)

    with col1:
        doc_id = st.selectbox("Document", options=FIXTURE_OPTIONS, key="ret_doc")

    with col2:
        # Use global model setting from sidebar if available
        global_model = st.session_state.get("config", {}).get("embedding_model", "all-MiniLM-L6-v2")
        model_default = MODEL_OPTIONS.index(global_model) if global_model in MODEL_OPTIONS else 0
        st.selectbox("Embedding Model", options=MODEL_OPTIONS, index=model_default, key="ret_model")

    with col3:
        st.slider("Top-k results", min_value=1, max_value=20, value=5, key="ret_k")

    # Strategies come from sidebar config (set in app.py)
    strategies = st.session_state.get("config", {}).get(
        "strategies", [StrategyName.LEXICHUNK.value, StrategyName.RCTS_512.value]
    )

    # --- Query Input ---
    st.subheader("Query")

    query_source = st.radio(
        "Query source",
        options=["Type your own", "Select annotated query"],
        horizontal=True,
        key="query_source",
    )

    query = ""

    if query_source == "Type your own":
        query = st.text_input(
            "Legal question",
            placeholder="e.g., What are the termination provisions?",
            key="ret_query_text",
        )
    else:
        annotated = _load_annotated_queries(doc_id)
        if annotated:
            options = [f"[{q['failure_mode']}] {q['text']}" for q in annotated]
            selected_idx = st.selectbox(
                "Select query",
                options=range(len(options)),
                format_func=lambda i: options[i],
                key="ret_annotated_query",
            )
            query = annotated[selected_idx]["text"]
            st.session_state["ret_annotated_info"] = annotated[selected_idx]
            st.caption(
                f"Query ID: {annotated[selected_idx]['id']} | "
                f"Tests: {annotated[selected_idx]['failure_mode']}"
            )
        else:
            st.warning(f"No annotated queries found for {doc_id}")
            query = st.text_input("Type a query instead", key="ret_fallback_query")

    # --- Run Retrieval ---
    run_button = st.button(
        "Search",
        type="primary",
        disabled=not query.strip(),
        use_container_width=True,
    )

    if run_button and query.strip():
        model = st.session_state.get("ret_model", MODEL_OPTIONS[0])
        _run_retrieval(doc_id, query, strategies, model)

    # --- Results ---
    st.markdown("---")
    st.subheader("Results")
    if "ret_results" not in st.session_state:
        st.info("Click 'Search' to run retrieval and see results.")
    elif not run_button:
        _display_retrieval_results(st.session_state["ret_results"])

    # --- Filtered Retrieval Demo ---
    _render_filtered_retrieval(doc_id)


def _run_retrieval(
    doc_id: str,
    query_text: str,
    strategies: list[str],
    model_name: str,
) -> None:
    """Run retrieval for a query across selected strategies and display results."""
    k = st.session_state.get("ret_k", 5)

    total = len(strategies)
    progress = st.progress(0, text="Initialising retrieval pipeline...")

    try:
        from scaffolder.chunking import get_strategy
        from scaffolder.embedding.pipeline import EmbeddingPipeline
        from scaffolder.fixtures import FixtureManager
        from scaffolder.models import EmbeddingModelName
        from scaffolder.retrieval.index import VectorIndex

        manager = FixtureManager()
        document = manager.get_by_id(doc_id)

        model_enum = EmbeddingModelName(model_name)
        embed_pipeline = EmbeddingPipeline()

        results_by_strategy: dict[str, list[dict[str, Any]]] = {}

        for i, strat_name in enumerate(strategies):
            pct_base = int(i / total * 100)
            progress.progress(pct_base, text=f"{strat_name}: chunking...")

            # Get or cache chunk set
            chunk_key = f"chunks_{doc_id}_{strat_name}"
            chunk_set = st.session_state.get(chunk_key)
            if chunk_set is None:
                strategy = get_strategy(StrategyName(strat_name))
                chunk_set = strategy.chunk(document)
                st.session_state[chunk_key] = chunk_set

            chunks = list(chunk_set.chunks)
            if not chunks:
                continue

            progress.progress(
                pct_base + int(30 / total),
                text=f"{strat_name}: embedding & indexing...",
            )

            # Get or cache index
            index_key = f"index_{doc_id}_{strat_name}_{model_name}"
            index = st.session_state.get(index_key)
            if index is None:
                chunk_texts = [c.text for c in chunks]
                embeddings = embed_pipeline.embed_texts(chunk_texts, model_enum)
                index = VectorIndex(dimension=embeddings.shape[1])
                index.add(chunks, embeddings)
                st.session_state[index_key] = index

            progress.progress(
                pct_base + int(70 / total),
                text=f"{strat_name}: searching...",
            )

            # Embed query and search
            query_emb = embed_pipeline.embed_texts([query_text], model_enum)
            hits = index.search(query_emb[0], k=k)

            results_by_strategy[strat_name] = [
                {
                    "rank": h.rank,
                    "score": h.score,
                    "text": h.chunk.text[:500],
                    "full_text": h.chunk.text,
                    "chunk_id": h.chunk.id,
                    "document_id": h.chunk.document_id,
                    "clause_type": h.chunk.metadata.get("clause_type", ""),
                }
                for h in hits
            ]

        progress.progress(100, text="Retrieval complete!")
        st.session_state["ret_results"] = results_by_strategy

    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        return

    # Display results
    _display_retrieval_results(results_by_strategy)


def _display_retrieval_results(results_by_strategy: dict[str, list[dict[str, Any]]]) -> None:
    """Display retrieval results with P@k chart, metrics table, and per-strategy chunks."""
    if not results_by_strategy:
        st.warning("No results returned.")
        return

    # Resolve relevant section descriptions for relevance matching
    annotated_info: dict[str, Any] | None = st.session_state.get("ret_annotated_info")
    relevant_descriptions: list[str] = []
    if annotated_info and "relevant_sections" in annotated_info:
        relevant_descriptions = [
            s["description"].lower()
            for s in annotated_info["relevant_sections"]
            if s.get("description")
        ]

    # --- P@k Comparison Chart ---
    _render_precision_chart(results_by_strategy, relevant_descriptions)

    # --- Metrics Summary Table ---
    _render_metrics_table(results_by_strategy, relevant_descriptions)

    # --- Per-strategy results in columns ---
    strategy_names = list(results_by_strategy.keys())
    cols = st.columns(len(strategy_names))

    for col, strat_name in zip(cols, strategy_names, strict=True):
        with col:
            st.markdown(f"**{strat_name}**")
            hits = results_by_strategy[strat_name]
            if not hits:
                st.caption("No results")
                continue

            for hit in hits:
                is_relevant = _is_hit_relevant(hit, relevant_descriptions)
                _render_hit(hit, is_relevant)


def _is_hit_relevant(hit: dict[str, Any], relevant_descriptions: list[str]) -> bool:
    """Check if a retrieved chunk matches any annotated relevant section.

    Uses text overlap: if the chunk text contains keywords from a relevant
    section description, it's considered relevant. This is a heuristic since
    we don't have exact section-to-chunk mappings.
    """
    if not relevant_descriptions:
        return False
    chunk_text = hit.get("full_text", hit.get("text", "")).lower()
    for desc in relevant_descriptions:
        # Check if key terms from the section description appear in the chunk
        terms = [t for t in desc.split() if len(t) > 4]
        if terms and sum(1 for t in terms if t in chunk_text) >= len(terms) * 0.5:
            return True
    return False


def _count_relevant_in_top_k(
    hits: list[dict[str, Any]], relevant_descriptions: list[str], k: int
) -> int:
    """Count how many of the top-k hits are relevant."""
    return sum(1 for h in hits[:k] if _is_hit_relevant(h, relevant_descriptions))


def _compute_mrr(hits: list[dict[str, Any]], relevant_descriptions: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for h in hits:
        if _is_hit_relevant(h, relevant_descriptions):
            return 1.0 / float(h["rank"])
    return 0.0


def _compute_ndcg(hits: list[dict[str, Any]], relevant_descriptions: list[str], k: int) -> float:
    """Compute NDCG@k with binary relevance."""
    import math

    dcg = 0.0
    for h in hits[:k]:
        if _is_hit_relevant(h, relevant_descriptions):
            dcg += 1.0 / math.log2(h["rank"] + 1)

    # Ideal DCG: all relevant items ranked at top
    n_relevant = sum(1 for h in hits if _is_hit_relevant(h, relevant_descriptions))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))

    return dcg / idcg if idcg > 0 else 0.0


def _render_precision_chart(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    relevant_descriptions: list[str],
) -> None:
    """Render a Plotly grouped bar chart comparing P@k across strategies."""
    if not relevant_descriptions:
        return

    import plotly.graph_objects as go

    k_values = [1, 3, 5, 10]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    fig = go.Figure()
    for i, (strat_name, hits) in enumerate(results_by_strategy.items()):
        p_at_k_values = []
        for kv in k_values:
            n_rel = _count_relevant_in_top_k(hits, relevant_descriptions, kv)
            p_at_k_values.append(n_rel / kv if kv > 0 else 0.0)

        fig.add_trace(
            go.Bar(
                name=strat_name,
                x=[f"P@{kv}" for kv in k_values],
                y=p_at_k_values,
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        title="Precision@k Comparison",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis={"range": [0, 1.05]},
        barmode="group",
        height=350,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_metrics_table(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    relevant_descriptions: list[str],
) -> None:
    """Render a summary table of retrieval metrics per strategy."""
    if not relevant_descriptions:
        st.caption("Select an annotated query to see retrieval metrics.")
        return

    rows: list[dict[str, str]] = []
    for strat_name, hits in results_by_strategy.items():
        mrr_val = _compute_mrr(hits, relevant_descriptions)
        ndcg_val = _compute_ndcg(hits, relevant_descriptions, k=10)
        p5 = _count_relevant_in_top_k(hits, relevant_descriptions, 5) / 5 if hits else 0.0
        rows.append(
            {
                "Strategy": strat_name,
                "P@5": f"{p5:.3f}",
                "MRR": f"{mrr_val:.3f}",
                "NDCG@10": f"{ndcg_val:.3f}",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_hit(hit: dict[str, Any], is_relevant: bool) -> None:
    """Render a single retrieval hit with optional relevance highlighting."""
    clause_label = f" ({hit['clause_type']})" if hit.get("clause_type") else ""

    if is_relevant:
        # Green border + RELEVANT badge
        st.markdown(
            f'<div style="border-left: 4px solid #4CAF50; padding-left: 8px; '
            f'margin-bottom: 4px;">'
            f"<strong>#{hit['rank']}</strong> — score: {hit['score']:.3f}{clause_label} "
            f'<span style="background: #C8E6C9; color: #2E7D32; padding: 1px 6px; '
            f'border-radius: 8px; font-size: 0.8em; font-weight: bold;">RELEVANT</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="border-left: 4px solid #ccc; padding-left: 8px; '
            f'margin-bottom: 4px;">'
            f"<strong>#{hit['rank']}</strong> — score: {hit['score']:.3f}{clause_label}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander(f"Chunk: {hit['chunk_id']}", expanded=hit["rank"] <= 3):
        st.text(hit["text"])


def _render_query_browser() -> None:
    """Render a query browser showing annotated queries without retrieval."""
    st.markdown("---")
    st.subheader("Annotated Query Browser")
    st.markdown("Browse the ground-truth queries while waiting for the retrieval pipeline.")

    doc_id = st.selectbox(
        "Document",
        options=FIXTURE_OPTIONS,
        key="browse_doc",
    )

    queries = _load_annotated_queries(doc_id)
    if not queries:
        st.info(f"No queries found for {doc_id}")
        return

    for q in queries:
        with st.expander(f"[{q['failure_mode']}] {q['text']}"):
            st.markdown(f"**ID:** {q['id']}")
            st.markdown(f"**Failure mode:** {q['failure_mode']}")
            st.markdown(f"**Query:** {q['text']}")


def _render_filtered_retrieval(doc_id: str) -> None:
    """Render the filtered retrieval demo.

    Shows that LexiChunk's clause_type metadata enables targeted retrieval
    that baselines cannot perform.
    """
    st.markdown("---")
    st.subheader("Filtered Retrieval (LexiChunk Exclusive)")
    st.markdown(
        "LexiChunk tags each chunk with a clause type, enabling **targeted retrieval** "
        "that general-purpose chunkers cannot support. Select a clause type to find "
        "all matching chunks across the document."
    )

    clause_types = [
        "definitions",
        "termination",
        "payment",
        "indemnification",
        "limitation_of_liability",
        "confidentiality",
        "intellectual_property",
        "data_protection",
        "governing_law",
        "dispute_resolution",
        "force_majeure",
        "warranties",
        "obligations",
        "notices",
    ]

    selected_type = st.selectbox(
        "Clause type to find",
        options=clause_types,
        key="filter_clause_type",
        help="Find all chunks classified as this clause type",
    )

    if st.button("Find Clauses", key="filter_search"):
        chunk_set_key = f"chunks_{doc_id}_lexichunk"
        chunk_set = st.session_state.get(chunk_set_key)

        if chunk_set is None:
            with st.spinner("Chunking with LexiChunk..."):
                try:
                    from scaffolder.chunking import get_strategy
                    from scaffolder.fixtures import FixtureManager

                    manager = FixtureManager()
                    document = manager.get_by_id(doc_id)
                    strategy = get_strategy(StrategyName.LEXICHUNK)
                    chunk_set = strategy.chunk(document)
                    st.session_state[chunk_set_key] = chunk_set
                except Exception as e:
                    st.error(f"Failed to chunk document: {e}")
                    return

        matching = [c for c in chunk_set.chunks if c.metadata.get("clause_type") == selected_type]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**LexiChunk: {len(matching)} matching chunks**")
            if matching:
                for i, chunk in enumerate(matching):
                    confidence = float(str(chunk.metadata.get("confidence", 0)))
                    with st.expander(
                        f"{selected_type} — Confidence: {confidence:.0%}",
                        expanded=i < 3,
                    ):
                        hierarchy = chunk.metadata.get("hierarchy_path", "")
                        if hierarchy:
                            st.caption(f"Path: {hierarchy}")
                        st.text(chunk.text[:800])
            else:
                st.info(f"No {selected_type} clauses found in this document.")

        with col2:
            st.markdown("**Baseline: Not possible**")
            st.warning(
                f"Baseline chunkers do not classify chunk types. "
                f"There is no way to filter for '{selected_type}' clauses "
                f"without LexiChunk's metadata. A keyword search for "
                f"'{selected_type.replace('_', ' ')}' would miss clauses "
                f"that don't contain the exact phrase."
            )

            baseline_key = f"chunks_{doc_id}_{StrategyName.RCTS_512.value}"
            baseline_set = st.session_state.get(baseline_key)
            if baseline_set:
                keyword = selected_type.replace("_", " ")
                keyword_matches = [
                    c for c in baseline_set.chunks if keyword.lower() in c.text.lower()
                ]
                st.caption(
                    f"Naive keyword search for '{keyword}' in baseline chunks: "
                    f"{len(keyword_matches)} matches (likely imprecise)"
                )
