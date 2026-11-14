# FreshLens

FreshLens is a stale-answer detector for AI applications. It extracts factual claims from an AI-generated answer, checks each claim against web evidence retrieved by Tavily, asks Gemini through Bifrost to assess that evidence, and returns a structured freshness report.

FreshLens is an evidence-based review aid. Its statuses and scores summarize the retrieved evidence and do not guarantee that a claim is true or false.

## Problem

AI answers can become outdated when facts change after a model's training data or retrieval snapshot. Leadership, software versions, prices, laws, schedules, product availability, and recent events are examples of information that can change. A fluent answer may still present an old claim without making its age clear.

FreshLens addresses this problem by checking claims individually against newly retrieved sources and showing the evidence and scoring behind each result.

## Solution overview

The Streamlit interface accepts an answer and calls the existing `FreshnessPipeline.analyze(answer)` workflow. The pipeline extracts atomic claims and their time-sensitivity classification, retrieves Tavily evidence separately for each claim, evaluates claims with Gemini through Bifrost, and applies deterministic scoring. The pipeline returns the shared `FreshnessReport` model used by the UI and MCP tool.

## Features

- **Atomic claim extraction:** Gemini identifies standalone factual claims in the submitted answer.
- **Time-sensitivity analysis:** Each extracted claim is classified as time-sensitive or stable, with a category and reason.
- **Evidence retrieval:** Tavily is queried separately for each claim, with up to five results.
- **Evidence-based verification:** Gemini evaluates a claim against its retrieved sources and returns a status, rationale, confidence, and optional newer information.
- **Deterministic freshness scoring:** The score is assigned from claim status and time sensitivity; confidence does not weight the score.
- **Persistent verification history:** Successful reports are saved to local SQLite and can be reopened without another verification call.
- **MCP integration:** A STDIO MCP server exposes the `verify_answer_freshness` tool and returns the complete structured report.

## Hackathon problem statement mapping

### Tavily

> "Build a Stale-Answer Detector for AI Applications."

FreshLens uses Tavily search evidence to evaluate whether claims in an AI-generated answer remain current.

### Bifrost

> "Build a Feature the Open-Source Gateway Is Missing."

FreshLens demonstrates a practical MCP integration for Bifrost users: an AI answer-verification tool that returns a structured freshness report. FreshLens integrates with Bifrost; it does not modify Bifrost's core source code.

## Technology

- Python (the current repository environment was validated with Python 3.13)
- Streamlit for the user interface
- Pydantic shared models for claims, evidence, verifications, summaries, and reports
- Tavily Search API for retrieved web evidence
- Bifrost's OpenAI-compatible chat completions API with Gemini as the configured model
- Official Python MCP SDK, pinned as `mcp==2.3.0`
- Python standard-library `sqlite3` for local verification history
- `pytest` for automated tests

## Prerequisites

- Windows and PowerShell (the setup commands below use Windows paths)
- Python 3.13 available as `python` in PowerShell
- A reachable Bifrost OpenAI-compatible inference endpoint configured to use an allowed Gemini model
- A Tavily API key
- A Bifrost virtual key accepted by the configured inference endpoint

The Bifrost and Tavily services are external runtime dependencies. Tests use mocked clients and do not require live API credentials.

## Install on Windows

From the repository root in PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The checked-in `requirements.txt` pins the Python package versions, including Streamlit and the MCP SDK. If `python` does not resolve to Python 3.13, select the installed Python 3.13 executable when creating the virtual environment.

## Configuration

Copy the example file and edit the new local `.env` file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set the following variables in `.env`:

| Variable | Purpose |
|---|---|
| `BIFROST_BASE_URL` | HTTP or HTTPS base URL for the Bifrost OpenAI-compatible API. The client appends `/chat/completions`. |
| `BIFROST_API_KEY` | Bifrost virtual key used for inference. |
| `BIFROST_MODEL` | Model identifier enabled for that Bifrost key and provider configuration. |
| `TAVILY_API_KEY` | Tavily Search API key. |

Use real values only in the local `.env` file. The configuration loader reads `.env` from the project root, validates required values, and avoids including key values in its error messages. `.env` is ignored by Git; do not commit it or paste its contents into logs or documentation.

## Start FreshLens

Start the Streamlit application from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Streamlit prints the local URL in the terminal. The default port may change if it is already in use.

## Use FreshLens

1. Paste an AI-generated answer into the input area.
2. Select **Verify freshness**.
3. Review the overall freshness score, evidence coverage, claim counts, and claim-level statuses.
4. Expand a claim to review its rationale, confidence, and retrieved evidence. Source links are clickable when the returned URL is a safe absolute HTTP(S) URL.
5. Reopen a prior successful report from **Verification history**, or delete an individual record. **Clear all history** requires confirmation.

The UI preserves the current result in the Streamlit session. Persistent history is stored separately, so saved reports remain available across browser refreshes, new sessions, and application restarts.

## Statuses and scoring

Each claim receives one status:

| Status | Meaning in FreshLens |
|---|---|
| `supported` | Retrieved evidence supports the claim as current. |
| `potentially_outdated` | Evidence suggests the claim may no longer be current. |
| `conflicting` | Credible retrieved evidence contradicts the claim. |
| `insufficient_evidence` | Available evidence cannot establish currentness or correctness. Lack of evidence is not proof that the claim is false. |

Scoring follows the existing deterministic policy:

- A stable claim has `freshness_score = null` (`None` in Python).
- A time-sensitive `supported` claim scores `100.0`.
- Every other valid time-sensitive status scores `0.0`. A zero means freshness was not established under this policy; it does not, by itself, mean the claim is false.
- The overall score is the arithmetic mean of the time-sensitive claim scores. It is `null` when there are no time-sensitive claims. Confidence does not weight the score.
- Evidence coverage is the proportion of time-sensitive claims with at least one evidence item; it is `0.0` when there are no time-sensitive claims.
- Status counts include both stable and time-sensitive claims. There is no separate aggregate status field in `FreshnessSummary`.

## Verification history

History uses Python's built-in `sqlite3`; no additional dependency is required. The default database is `.freshlens_history.sqlite3` in the repository root. Each successful pipeline result is stored with a unique record ID, UTC timestamp, submitted answer, and complete serialized `FreshnessReport`.

The interface displays the latest 20 records, with a shortened answer and values from the saved report summary. Opening a record loads its saved report and does not call Tavily or Gemini again. Individual deletion and confirmation-protected clear-all are supported.

This initial history implementation is local and single-user, not cloud-synchronized or multi-user. The SQLite database is unencrypted and contains submitted answers and reports. The database and its SQLite `-wal` and `-shm` sidecars are ignored by Git and must not be committed. Avoid using FreshLens history for information that should not be stored locally.

## MCP integration

`mcp_server.py` creates an MCP server named `FreshLens` using the pinned Python MCP SDK and STDIO transport. It registers one tool:

```text
verify_answer_freshness(answer: str) -> FreshnessReport
```

The tool forwards the answer unchanged to `FreshnessPipeline.analyze()` and returns the shared Pydantic `FreshnessReport`, including `original_answer`, `claims`, `verifications`, and `summary`. The SDK exposes a structured output schema. Pipeline failures are not replaced with fabricated reports.

An MCP host starts the server as a STDIO subprocess using the project's virtual-environment Python and `mcp_server.py`. For a Bifrost MCP Gateway registration, configure the host's supported STDIO client settings separately; this repository does not alter Bifrost Gateway core code or provide a universal host configuration file. Bifrost may show a client-qualified tool alias such as `freshlens-verify_answer_freshness`; the server-side tool name remains `verify_answer_freshness`.

The process command an MCP host can launch from the repository root is:

```powershell
.\.venv\Scripts\python.exe .\mcp_server.py
```

This command runs the STDIO protocol server for an MCP host; it is not a standalone interactive console command.

## Run tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The automated suites cover schemas, claim extraction, scoring, Tavily client behavior, the verification pipeline, mocked end-to-end workflow, MCP tool behavior, and local history storage. External APIs are not called by these tests.

## Security and privacy

- Keep `.env` local. It contains provider credentials and is ignored by Git.
- The configuration and client code avoid logging credential values. Do not put credentials in prompts, answers, evidence, screenshots, or issue reports.
- Submitted answer text is sent to Bifrost for claim extraction. Extracted claim text is sent to Tavily as separate search queries. Bifrost receives each claim and the retrieved evidence for verification.
- Claim and evidence content is encoded as JSON and identified as untrusted input in Gemini prompts; web evidence is treated as information, not instructions.
- The local SQLite history is plaintext and has no user authentication or multi-user access isolation. Use the UI deletion controls when removing records, but do not treat deletion as a secure-erase guarantee.
- No public deployment or cloud history service is included in this repository.

## Limitations and future improvements

- Verification depends on external model output and the coverage, relevance, and recency of retrieved sources. FreshLens does not claim benchmark superiority over another AI system.
- Search evidence may be incomplete, and publication dates may be missing. `insufficient_evidence` represents uncertainty rather than falsity.
- Freshness scores are intentionally simple and deterministic; they do not measure general factual correctness or incorporate confidence.
- History is a local, unencrypted, single-user SQLite file with no synchronization, authentication, or automatic retention policy.
- The repository does not include a public deployment, multi-user hosting setup, or service-level guarantees.

Possible future work includes benchmark-based evaluation, configurable retention/export, protected multi-user storage, and deployment/operations guidance. These capabilities are not implemented here.
