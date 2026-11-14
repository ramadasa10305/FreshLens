"""Streamlit interface for FreshLens answer freshness analysis."""

from __future__ import annotations

import html
from urllib.parse import urlparse

import streamlit as st

from history_store import HistoryRecord, HistoryStore
from schemas import ClaimStatus, FreshnessReport, validate_answer_length


_STATUS_PRESENTATION = {
    ClaimStatus.SUPPORTED: ("Supported", "supported"),
    ClaimStatus.POTENTIALLY_OUTDATED: (
        "Potentially outdated",
        "outdated",
    ),
    ClaimStatus.CONFLICTING: ("Conflicting", "conflicting"),
    ClaimStatus.INSUFFICIENT_EVIDENCE: (
        "Insufficient evidence",
        "insufficient",
    ),
}


_APP_STYLES = """
<style>
    :root {
        color-scheme: dark;
        --background-color: #0b1220;
        --secondary-background-color: #111c2e;
        --text-color: #f8fafc;
        --primary-color: #2dd4bf;
        --border-color: #334155;
        --fl-bg: #0b1220;
        --fl-surface: #111c2e;
        --fl-surface-raised: #17253a;
        --fl-border: #334155;
        --fl-text: #f8fafc;
        --fl-secondary: #cbd5e1;
        --fl-muted: #94a3b8;
        --fl-accent: #2dd4bf;
    }

    .stApp {
        --background-color: #0b1220;
        --secondary-background-color: #111c2e;
        --text-color: #f8fafc;
        --primary-color: #2dd4bf;
        --border-color: #334155;
        background:
            radial-gradient(ellipse at 50% -24%, rgba(45, 212, 191, .12), transparent 52%),
            linear-gradient(180deg, #101b2d 0%, var(--fl-bg) 44rem);
        color: var(--fl-text) !important;
    }
    [data-testid="stMain"] { background: transparent; color: var(--fl-text) !important; }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, .92); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        color: var(--fl-text) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #142238, #111c2e);
        border-color: var(--fl-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 14px 36px rgba(0, 0, 0, .2);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--fl-text) !important;
        letter-spacing: -.025em;
        font-weight: 620;
    }
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] strong,
    .stApp [data-testid="stMarkdownContainer"] em,
    .stApp [data-testid="stMarkdownContainer"] blockquote,
    .stApp [data-testid="stText"],
    .stApp [data-testid="stText"] *,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] *,
    .stApp [data-testid="stHeading"],
    .stApp [data-testid="stHeading"] *,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp label,
    .stApp p {
        color: var(--fl-secondary) !important;
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] * {
        color: var(--fl-muted) !important;
    }

    [data-testid="stTextArea"] textarea {
        color-scheme: light;
        background: #f8fafc !important;
        border: 1px solid #64748b !important;
        border-radius: 10px;
        color: #0f172a !important;
        line-height: 1.65;
        padding: 1rem 1.1rem;
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 3px rgba(45, 212, 191, .24);
    }
    [data-testid="stTextArea"] textarea::placeholder { color: #475569 !important; opacity: 1; }
    [data-testid="stForm"] {
        background: transparent;
        border: 0;
        padding: 0;
    }
    .stApp [data-testid="stFormSubmitButton"] button,
    .stApp button[data-testid="stBaseButton-primary"] {
        min-height: 2.75rem;
        border: 1px solid #2dd4bf !important;
        border-radius: 9px !important;
        background: #2dd4bf !important;
        color: #0b1220 !important;
        font-weight: 650 !important;
        box-shadow: none;
        transition: filter .15s ease, transform .15s ease;
    }
    .stApp [data-testid="stFormSubmitButton"] button *,
    .stApp button[data-testid="stBaseButton-primary"] * {
        color: #0b1220 !important;
    }
    .stApp [data-testid="stFormSubmitButton"] button:hover,
    .stApp button[data-testid="stBaseButton-primary"]:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    .stApp button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #475569 !important;
        border-radius: 9px !important;
        background: #17253a !important;
        color: var(--fl-text) !important;
    }
    .stApp button[data-testid="stBaseButton-secondary"] * {
        color: var(--fl-text) !important;
    }
    [data-testid="stMetric"] {
        min-height: 92px;
        padding: .95rem 1rem;
        border: 1px solid var(--fl-border);
        border-radius: 11px;
        background: #111c2e;
    }
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] *,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] *,
    .stApp [data-testid="stMetricDelta"],
    .stApp [data-testid="stMetricDelta"] * {
        color: var(--fl-text) !important;
    }
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricDelta"] {
        color: var(--fl-secondary) !important;
    }
    .stApp [data-testid="stLinkButton"] a,
    .stApp [data-testid="stLinkButton"] a * {
        border-radius: 8px;
        color: #99f6e4 !important;
    }
    .stApp [data-testid="stLinkButton"] a {
        border: 1px solid #3b556c;
        background: #17253a;
    }
    .stApp [data-testid="stLinkButton"] a:hover {
        border-color: #2dd4bf;
        background: #1b3044;
    }
    .stApp [data-testid="stAlert"], .stApp [data-testid="stAlert"] * {
        color: #f8fafc !important;
    }
    .stApp [data-testid="stAlert"] {
        border: 1px solid #475569;
        border-radius: 10px;
        background: #18263a;
    }
    [data-testid="stExpander"] {
        border-color: var(--fl-border);
        border-radius: 10px;
        background: #111c2e;
    }
    .stApp [data-testid="stExpander"],
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
    .stApp [data-testid="stSidebar"],
    .stApp [data-testid="stSidebar"] * {
        color: var(--fl-text) !important;
    }
    .stApp a {
        color: #5eead4 !important;
        text-decoration-color: rgba(94, 234, 212, .55);
    }
    .stApp a:hover { color: #99f6e4 !important; }
    .stApp :focus-visible {
        outline: 2px solid #5eead4 !important;
        outline-offset: 2px;
    }
    .freshlens-status {
        display: inline-block;
        padding: .35rem .68rem;
        border: 1px solid transparent;
        border-radius: 999px;
        font-size: .82rem;
        font-weight: 650;
        line-height: 1.2;
    }
    .freshlens-status-supported {
        color: #99f6e4;
        background: #123b38;
        border-color: #26786d;
    }
    .freshlens-status-outdated {
        color: #fde68a;
        background: #433313;
        border-color: #85631c;
    }
    .freshlens-status-conflicting {
        color: #fecdd3;
        background: #451f2b;
        border-color: #8c3d53;
    }
    .freshlens-status-insufficient,
    .freshlens-status-unavailable {
        color: #e2e8f0;
        background: #263449;
        border-color: #52647b;
    }

    .freshlens-hero {
        margin: 0 0 1.4rem;
        padding: clamp(1.6rem, 4vw, 3.2rem);
        border: 1px solid #334b63;
        border-radius: 17px;
        background:
            radial-gradient(ellipse at 88% 8%, rgba(45, 212, 191, .12), transparent 38%),
            linear-gradient(120deg, #17263b, #111c2e 70%, #132c39);
        box-shadow: 0 18px 42px rgba(0, 0, 0, .2);
    }
    .freshlens-brand {
        margin-bottom: 1.35rem;
        color: #5eead4 !important;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .freshlens-hero h1 {
        max-width: 760px;
        margin: 0;
        font-size: clamp(2rem, 4.4vw, 3.35rem);
        line-height: 1.08;
        color: #f8fafc !important;
    }
    .freshlens-hero p {
        max-width: 690px;
        margin: 1rem 0 0;
        color: #cbd5e1 !important;
        font-size: 1.03rem;
        line-height: 1.65;
    }
    .freshlens-section-label {
        margin: .25rem 0 .65rem;
        color: #5eead4 !important;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2.5rem; }
        .freshlens-hero { border-radius: 13px; padding: 1.4rem; }
        .freshlens-hero p { font-size: .96rem; }
        [data-testid="stMetric"] { min-height: 82px; padding: .75rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
</style>
"""


def _safe_source_url(value: object) -> str | None:
    """Return a link only for absolute HTTP(S) URLs without embedded credentials."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlparse(value.strip())
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port == 0
        ):
            return None
    except (ValueError, UnicodeError):
        return None
    return value.strip()


def _status_details(value: object) -> tuple[str, str]:
    """Resolve a returned status to a display label and style safely."""
    try:
        status = value if isinstance(value, ClaimStatus) else ClaimStatus(value)
    except (ValueError, TypeError):
        return "Status unavailable", "unavailable"
    return _STATUS_PRESENTATION[status]


def _clear_current_check() -> None:
    st.session_state["answer_input"] = ""
    st.session_state["freshness_report"] = None
    st.session_state["freshness_error"] = None


def _metric_value(value: object, suffix: str = "") -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _short_answer(value: str, max_length: int = 190) -> str:
    """Return a compact, single-line preview of a saved answer."""
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _history_summary(record: HistoryRecord) -> str:
    """Format summary values already present in a saved FreshnessReport."""
    summary = record.report.summary
    score = summary.overall_freshness_score
    score_text = f"{score:.1f}/100" if score is not None else "not scored"
    coverage_text = f"{summary.evidence_coverage:.0%} coverage"
    return (
        f"{summary.total_claims} claims | "
        f"{summary.supported_claims} supported | "
        f"{summary.potentially_outdated_claims} potentially outdated | "
        f"{summary.conflicting_claims} conflicting | "
        f"{summary.insufficient_evidence_claims} insufficient evidence | "
        f"freshness {score_text} | {coverage_text}"
    )


def _render_history(history_store: HistoryStore | None) -> None:
    st.markdown('<div class="freshlens-section-label">Saved checks</div>', unsafe_allow_html=True)
    st.header("Verification history")
    st.caption(
        "Stored locally on this machine for single-user use. This is not shared "
        "cloud storage or a multi-user history service."
    )

    if history_store is None:
        st.warning("Local history is unavailable. Current verification results still work.")
        return

    try:
        records = history_store.list_recent(limit=20)
    except Exception:
        st.warning("Saved history could not be loaded from local storage.")
        return

    if not records:
        st.caption("Successful checks will appear here after they are saved.")
    else:
        for record in records:
            with st.container(border=True):
                created_at = record.created_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                st.markdown(f"**{created_at}**")
                st.text(_short_answer(record.original_answer))
                st.caption(_history_summary(record))
                open_col, delete_col = st.columns([1, 1], gap="small")
                with open_col:
                    open_clicked = st.button(
                        "Open report",
                        key=f"open_history_{record.record_id}",
                        width="stretch",
                    )
                with delete_col:
                    delete_clicked = st.button(
                        "Delete record",
                        key=f"delete_history_{record.record_id}",
                        width="stretch",
                    )

                if open_clicked:
                    try:
                        saved_record = history_store.get(record.record_id)
                    except Exception:
                        st.session_state["history_error"] = (
                            "The saved report could not be opened from local storage."
                        )
                    else:
                        if saved_record is None:
                            st.session_state["history_error"] = (
                                "That history record is no longer available."
                            )
                        else:
                            st.session_state["freshness_report"] = saved_record.report
                            st.session_state["freshness_error"] = None
                            st.session_state["history_error"] = None
                    st.rerun()

                if delete_clicked:
                    try:
                        history_store.delete(record.record_id)
                    except Exception:
                        st.session_state["history_error"] = (
                            "The history record could not be deleted from local storage."
                        )
                    else:
                        st.session_state["history_error"] = None
                    st.rerun()

    if st.session_state.get("history_error"):
        st.warning(st.session_state["history_error"])

    if st.session_state.get("confirm_clear_history", False):
        with st.container(border=True):
            st.warning("Clear all saved verification history? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                confirm_clear = st.button(
                    "Delete all history",
                    key="confirm_delete_all_history",
                    type="primary",
                    width="stretch",
                )
            with cancel_col:
                cancel_clear = st.button(
                    "Cancel",
                    key="cancel_delete_all_history",
                    width="stretch",
                )
            if confirm_clear:
                try:
                    history_store.clear_all()
                except Exception:
                    st.session_state["history_error"] = (
                        "History could not be cleared from local storage."
                    )
                else:
                    st.session_state["history_error"] = None
                st.session_state["confirm_clear_history"] = False
                st.rerun()
            if cancel_clear:
                st.session_state["confirm_clear_history"] = False
                st.rerun()
    elif records and st.button(
        "Clear all history",
        key="request_clear_all_history",
        width="stretch",
    ):
        st.session_state["confirm_clear_history"] = True
        st.rerun()


def _render_evidence(evidence_items: object) -> None:
    if not isinstance(evidence_items, (list, tuple)) or not evidence_items:
        st.caption("No retrieved evidence was provided for this claim.")
        return

    st.markdown("**Retrieved evidence**")
    for index, evidence in enumerate(evidence_items, start=1):
        with st.container(border=True):
            title = getattr(evidence, "title", None)
            title_text = title.strip() if isinstance(title, str) else ""
            url = _safe_source_url(getattr(evidence, "url", None))

            if url:
                st.link_button(title_text or f"Open source {index}", url, width="stretch")
                st.caption(url)
            elif title_text:
                st.text(title_text)
                st.caption("Source link unavailable")
            else:
                st.caption("Source title and link unavailable")

            published_date = getattr(evidence, "published_date", None)
            if isinstance(published_date, str) and published_date.strip():
                st.caption(f"Published: {published_date.strip()}")
            else:
                st.caption("Publication date not provided")

            excerpt = getattr(evidence, "excerpt", None)
            if isinstance(excerpt, str) and excerpt.strip():
                st.text(excerpt.strip())
            else:
                st.caption("No excerpt provided")


def _render_claims(report: FreshnessReport) -> None:
    claims = getattr(report, "claims", None) or []
    verifications = getattr(report, "verifications", None) or []
    verification_by_id = {
        verification.claim_id: verification
        for verification in verifications
        if getattr(verification, "claim_id", None) is not None
    }

    st.markdown('<div class="freshlens-section-label">Analysis</div>', unsafe_allow_html=True)
    st.header("Claim-by-claim results")
    if not claims:
        st.info(
            "No factual claims were extracted from this answer, so there are no "
            "claim-level freshness results to show."
        )
        return

    for index, claim in enumerate(claims, start=1):
        claim_id = getattr(claim, "claim_id", None)
        verification = verification_by_id.get(claim_id)
        with st.container(border=True):
            status_value = getattr(verification, "status", None)
            label, status_style = _status_details(status_value)
            title_col, status_col = st.columns([3, 1], vertical_alignment="center")
            with title_col:
                st.subheader(f"Claim {index}")
            with status_col:
                st.markdown(
                    f'<span class="freshlens-status freshlens-status-{status_style}">{html.escape(label)}</span>',
                    unsafe_allow_html=True,
                )

            claim_text = getattr(claim, "text", None)
            if isinstance(claim_text, str):
                st.text(claim_text)

            category = getattr(claim, "category", None)
            time_sensitive = getattr(claim, "is_time_sensitive", None)
            metadata_cols = st.columns(2)
            with metadata_cols[0]:
                st.caption(f"Category: {category or 'Not provided'}")
            with metadata_cols[1]:
                if time_sensitive is True:
                    st.caption("Time sensitivity: changes over time")
                elif time_sensitive is False:
                    st.caption("Time sensitivity: stable claim")
                else:
                    st.caption("Time sensitivity: not provided")

            sensitivity_reason = getattr(claim, "time_sensitivity_reason", None)
            if isinstance(sensitivity_reason, str) and sensitivity_reason.strip():
                st.caption(sensitivity_reason.strip())

            if verification is None:
                st.warning("No verification record was returned for this claim.")
                continue

            confidence = getattr(verification, "confidence", None)
            freshness_score = getattr(verification, "freshness_score", None)
            verification_cols = st.columns(2)
            with verification_cols[0]:
                st.metric(
                    "Confidence",
                    f"{confidence:.0%}"
                    if isinstance(confidence, (int, float))
                    else "Not available",
                    border=True,
                )
            with verification_cols[1]:
                if freshness_score is None and time_sensitive is False:
                    score_text = "Not scored (stable claim)"
                elif isinstance(freshness_score, (int, float)):
                    score_text = f"{freshness_score:.1f} / 100"
                else:
                    score_text = "Not available"
                st.metric("Claim freshness score", score_text, border=True)

            rationale = getattr(verification, "rationale", None)
            if isinstance(rationale, str) and rationale.strip():
                st.markdown("**Rationale**")
                st.text(rationale.strip())

            newer_information = getattr(verification, "newer_information", None)
            if isinstance(newer_information, str) and newer_information.strip():
                st.markdown("**Newer information**")
                st.text(newer_information.strip())

            _render_evidence(getattr(verification, "evidence", None))


def _render_report(report: FreshnessReport) -> None:
    summary = getattr(report, "summary", None)
    if summary is None:
        st.error("The pipeline returned a report without a freshness summary.")
        return

    st.markdown('<div class="freshlens-section-label">Report</div>', unsafe_allow_html=True)
    st.header("Freshness overview")
    score = getattr(summary, "overall_freshness_score", None)
    coverage = getattr(summary, "evidence_coverage", None)
    overview_cols = st.columns(2)
    with overview_cols[0]:
        score_text = f"{score:.1f} / 100" if isinstance(score, (int, float)) else "Not scored"
        st.metric("Overall freshness score", score_text, border=True)
        st.caption("The pipeline reports status per claim; it does not provide a separate aggregate status.")
    with overview_cols[1]:
        coverage_text = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "Not available"
        st.metric("Evidence coverage", coverage_text, border=True)

    st.subheader("Summary metrics")
    metric_rows = (
        (
            ("Total claims", "total_claims"),
            ("Time-sensitive claims", "time_sensitive_claims"),
            ("Supported", "supported_claims"),
        ),
        (
            ("Potentially outdated", "potentially_outdated_claims"),
            ("Conflicting", "conflicting_claims"),
            ("Insufficient evidence", "insufficient_evidence_claims"),
        ),
    )
    for row in metric_rows:
        cols = st.columns(3)
        for col, (label, field_name) in zip(cols, row):
            with col:
                st.metric(
                    label,
                    _metric_value(getattr(summary, field_name, None)),
                    border=True,
                )

    original_answer = getattr(report, "original_answer", None)
    with st.expander("Original answer", expanded=False):
        if isinstance(original_answer, str):
            st.text(original_answer)
        else:
            st.caption("Original answer was not included in the report.")

    _render_claims(report)


st.set_page_config(
    page_title="FreshLens - Answer freshness",
    layout="wide",
)
st.markdown(_APP_STYLES, unsafe_allow_html=True)

if "answer_input" not in st.session_state:
    st.session_state["answer_input"] = ""
if "freshness_report" not in st.session_state:
    st.session_state["freshness_report"] = None
if "freshness_error" not in st.session_state:
    st.session_state["freshness_error"] = None
if "history_error" not in st.session_state:
    st.session_state["history_error"] = None
if "confirm_clear_history" not in st.session_state:
    st.session_state["confirm_clear_history"] = False

try:
    history_store: HistoryStore | None = HistoryStore()
except Exception:
    history_store = None

st.markdown(
    """
    <section class="freshlens-hero" aria-labelledby="freshlens-title">
      <div class="freshlens-brand">FreshLens</div>
      <h1 id="freshlens-title">Know when an AI answer needs a second look.</h1>
      <p>FreshLens checks factual claims against freshly retrieved web evidence, then shows what the sources support, contradict, or cannot establish.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="freshlens-section-label">Start a freshness check</div>', unsafe_allow_html=True)
st.subheader("Review an answer")
st.caption("Paste a response from an AI assistant. FreshLens will extract and check its factual claims individually.")

with st.form("freshness_check_form", border=True, enter_to_submit=False):
    answer = st.text_area(
        "Paste an AI-generated answer",
        key="answer_input",
        height=240,
        placeholder="Paste an answer with factual claims you want to verify.",
        help="Answers up to 12,000 characters are supported. Claims are checked individually against retrieved evidence.",
    )
    submitted = st.form_submit_button(
        "Verify freshness",
        type="primary",
        width="stretch",
    )

if submitted:
    st.session_state["freshness_report"] = None
    st.session_state["freshness_error"] = None
    st.session_state["history_error"] = None

    if not isinstance(answer, str) or not answer.strip():
        st.session_state["freshness_error"] = "Paste an answer before starting a check."
    else:
        try:
            validate_answer_length(answer)
        except ValueError as exc:
            st.session_state["freshness_error"] = str(exc)
        else:
            try:
                with st.spinner("Extracting claims, retrieving evidence, and evaluating freshness..."):
                    # Import lazily so the page can render a helpful state when the
                    # project credentials or external services are not configured.
                    from pipeline import FreshnessPipeline

                    report = FreshnessPipeline().analyze(answer)
                st.session_state["freshness_report"] = report
                if history_store is not None:
                    try:
                        history_store.save(report)
                    except Exception:
                        # Keep the successful result visible if local persistence fails.
                        st.session_state["history_error"] = (
                            "Verification completed, but this result could not be "
                            "saved to local history."
                        )
            except Exception:
                # Pipeline exceptions can contain request or provider details. Keep
                # the user-facing message useful without rendering exception text.
                st.session_state["freshness_report"] = None
                st.session_state["freshness_error"] = (
                    "Freshness verification could not be completed. Check that the "
                    "configured services are available, then try again."
                )

if st.session_state["freshness_error"]:
    st.error(st.session_state["freshness_error"])

if st.session_state["freshness_report"] is not None:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button(
            "Clear current result",
            key="clear_freshness_result",
            on_click=_clear_current_check,
        )
    _render_report(st.session_state["freshness_report"])

_render_history(history_store)
