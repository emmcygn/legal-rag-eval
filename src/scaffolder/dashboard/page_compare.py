"""Page 1: Side-by-side chunk comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from scaffolder.models import ChunkSet, Document

# Map display names to StrategyName enum values. rcts_1024 is the size-matched
# control for LexiChunk's ~790-char chunks; rcts_512 is kept as the older,
# unmatched comparison. LexiChunk Contextual is included as a baseline here too so
# it can be compared against plain LexiChunk, the same way the other strategies are.
_BASELINE_OPTIONS: dict[str, str] = {
    "LangChain RCTS (512)": "rcts_512",
    "LangChain RCTS (1024, size-matched)": "rcts_1024",
    "Sentence Split": "sentence_split",
    "Fixed Size (512)": "fixed_size",
    "LexiChunk Contextual": "lexichunk_contextual",
}

_FIXTURE_IDS = [
    "uk_service_agreement",
    "uk_terms_conditions",
    "us_msa",
    "us_terms_of_service",
    "eu_gdpr_excerpt",
]

MAX_UPLOAD_SIZE_BYTES = 100 * 1024  # 100 KB
MAX_DOCUMENT_CHARS = 200_000
MIN_DOCUMENT_CHARS = 50


def _validate_upload(uploaded_file: Any) -> tuple[str | None, str | None]:
    """Validate an uploaded file.

    Returns (content, None) on success, (None, error_message) on failure.
    """
    file_size = uploaded_file.size
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return None, (
            f"File too large: {file_size / 1024:.0f} KB. "
            f"Maximum allowed: {MAX_UPLOAD_SIZE_BYTES // 1024} KB."
        )

    raw_bytes = uploaded_file.getvalue()
    content: str | None = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            content = raw_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if content is None:
        return None, (
            "Unable to decode file. Please ensure it is a UTF-8 or Latin-1 encoded text file."
        )

    if len(content.strip()) < MIN_DOCUMENT_CHARS:
        return None, (
            f"Document too short ({len(content.strip())} characters). "
            f"Minimum: {MIN_DOCUMENT_CHARS} characters."
        )

    if len(content) > MAX_DOCUMENT_CHARS:
        content = content[:MAX_DOCUMENT_CHARS]

    return content, None


def _validate_document(doc: Document) -> str | None:
    """Validate a document before chunking.

    Returns None if valid, error message if invalid.
    """
    text = doc.text.strip()

    if not text:
        return "Document is empty. Please provide a document with text content."

    if len(text) < MIN_DOCUMENT_CHARS:
        return (
            f"Document too short ({len(text)} characters). "
            f"Chunking requires at least {MIN_DOCUMENT_CHARS} characters."
        )

    if "\x00" in doc.text:
        return "Document appears to contain binary content. Please upload a text file."

    return None


def _load_fixture(fixture_name: str) -> Document:
    """Load a fixture document by name."""
    from scaffolder.fixtures import FixtureManager

    manager = FixtureManager()
    return manager.get_by_id(fixture_name)


def _create_document_from_text(text: str, name: str = "pasted") -> Document:
    """Create a Document from raw text."""
    from scaffolder.models import Document, DocumentType, Jurisdiction

    return Document(
        id=name,
        text=text,
        jurisdiction=Jurisdiction.UK,
        document_type=DocumentType.SERVICE_AGREEMENT,
        source="<pasted>",
    )


def _chunk_document(doc: Document, strategy_value: str) -> ChunkSet:
    """Chunk a document using the specified strategy."""
    from scaffolder.chunking import get_strategy
    from scaffolder.models import StrategyName

    strategy_name = StrategyName(strategy_value)
    strategy = get_strategy(strategy_name)
    return strategy.chunk(doc)


def render_page() -> None:
    """Render the chunk comparison page."""
    st.title("Chunk Comparison")
    st.markdown(
        "Compare how **LexiChunk** and baseline chunkers split the same legal document. "
        "Select a document source and a baseline strategy, then click **Run Chunking**."
    )

    # --- Document Selection ---
    st.subheader("1. Select Document")
    doc_source = st.radio(
        "Document source",
        options=["Built-in fixture", "Upload file", "Paste text"],
        horizontal=True,
        key="doc_source",
    )

    document: Document | None = None

    if doc_source == "Built-in fixture":
        fixture = st.selectbox(
            "Select fixture",
            options=_FIXTURE_IDS,
            key="fixture_select",
        )
        try:
            document = _load_fixture(fixture)
            with st.expander("Preview document", expanded=False):
                st.text(document.text[:2000] + ("..." if len(document.text) > 2000 else ""))
                st.caption(
                    f"Jurisdiction: {document.jurisdiction.value} | "
                    f"Type: {document.document_type.value} | "
                    f"Length: {document.char_count:,} chars"
                )
        except Exception as e:
            st.error(f"Failed to load fixture: {e}")

    elif doc_source == "Upload file":
        uploaded = st.file_uploader(
            f"Upload a .txt legal document (max {MAX_UPLOAD_SIZE_BYTES // 1024} KB)",
            type=["txt"],
            key="file_upload",
        )
        if uploaded is not None:
            content, error = _validate_upload(uploaded)
            if error:
                st.error(error)
            elif content is not None:
                document = _create_document_from_text(content, uploaded.name.replace(".txt", ""))
                st.success(f"Loaded: {uploaded.name} ({len(content):,} chars)")

    elif doc_source == "Paste text":
        text = st.text_area(
            "Paste legal document text",
            height=200,
            placeholder="Paste your legal document text here...",
            key="paste_text",
        )
        if text.strip():
            document = _create_document_from_text(text.strip())
            st.caption(f"Length: {len(text):,} chars")

    # --- Strategy Selection ---
    st.subheader("2. Select Baseline")
    baseline_label = st.selectbox(
        "Baseline chunking strategy",
        options=list(_BASELINE_OPTIONS.keys()),
        key="baseline_select",
    )
    baseline_value = _BASELINE_OPTIONS[baseline_label]

    # --- Run Chunking ---
    st.markdown("---")

    run_button = st.button(
        "Run Chunking",
        type="primary",
        disabled=document is None,
        use_container_width=True,
    )

    if document is None and run_button:
        st.warning("Please select or provide a document first.")
        return

    if run_button and document is not None:
        validation_error = _validate_document(document)
        if validation_error:
            st.error(validation_error)
            return

        lexi_key = f"chunks_{document.id}_lexichunk"
        base_key = f"chunks_{document.id}_{baseline_value}"

        progress = st.progress(0, text="Chunking with LexiChunk...")
        try:
            lexi_result = _chunk_document(document, "lexichunk")
            st.session_state[lexi_key] = lexi_result
        except Exception as e:
            st.error(f"LexiChunk chunking failed: {e}")
            return

        progress.progress(50, text=f"Chunking with {baseline_label}...")
        try:
            base_result = _chunk_document(document, baseline_value)
            st.session_state[base_key] = base_result
        except Exception as e:
            st.error(f"{baseline_label} chunking failed: {e}")
            return

        progress.progress(100, text="Chunking complete!")

        st.session_state["last_doc_id"] = document.id
        st.session_state["last_baseline"] = baseline_value
        st.session_state["last_baseline_label"] = baseline_label

    # --- Display Results ---
    doc_id = st.session_state.get("last_doc_id")
    last_baseline = st.session_state.get("last_baseline", baseline_value)
    last_baseline_label = st.session_state.get("last_baseline_label", baseline_label)

    if doc_id:
        lexi_key = f"chunks_{doc_id}_lexichunk"
        base_key = f"chunks_{doc_id}_{last_baseline}"

        lexi_result = st.session_state.get(lexi_key)  # type: ignore[assignment]
        base_result = st.session_state.get(base_key)  # type: ignore[assignment]

        if lexi_result and base_result:
            _render_results(lexi_result, base_result, last_baseline_label)


def _render_results(lexi: ChunkSet, baseline: ChunkSet, baseline_name: str) -> None:
    """Render the comparison results using enhanced components."""
    from scaffolder.dashboard.components import render_chunk_list, render_chunk_size_chart

    st.subheader("3. Results")

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta = baseline.count - lexi.count
        st.metric(
            "LexiChunk Chunks",
            lexi.count,
            delta=f"{-delta}" if delta != 0 else None,
            help="Fewer chunks often means less fragmentation",
        )
    with col2:
        st.metric(f"{baseline_name} Chunks", baseline.count)
    with col3:
        st.metric("LexiChunk Avg Size", f"{lexi.avg_chunk_size:.0f} chars")
    with col4:
        st.metric("LexiChunk Time", f"{lexi.elapsed_seconds:.3f}s")

    # Chunk size distribution chart
    render_chunk_size_chart(lexi, baseline, baseline_name)

    st.markdown("---")

    # Side-by-side chunk display with enhanced components
    left, right = st.columns(2)

    with left:
        render_chunk_list(lexi, strategy_label="LexiChunk", show_metadata=True)

    with right:
        render_chunk_list(baseline, strategy_label=baseline_name, show_metadata=False)
