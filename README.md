# ⚖️ Legal Document Assistant

An AI-powered assistant that helps everyday people understand, compare, and
navigate legal documents — without pretending to replace a lawyer.

Built for the **Prompt Wars (GDG)** "Access to Justice / Legal-Tech" theme.

---

## 1. Chosen Vertical

**Individual / Consumer legal help** — renters, freelancers, employees, and
small business owners who regularly receive legal documents (leases,
freelance contracts, offer letters, NDAs, terms of service) but rarely have
easy or affordable access to a lawyer before they have to sign something.

**Persona:** *Riya, a freelance designer*, gets sent a client contract with
an unfamiliar IP-assignment clause and a 90-day non-compete. She doesn't
want to pay a lawyer $200 just to find out if the clause is normal. She
needs to (a) understand it in plain English, (b) know if it's unusually
one-sided, and (c) walk into a paid consultation — if she needs one — with
sharp, specific questions instead of "is this okay?"

---

## 2. Approach & Logic

The assistant is organized around **six concrete tasks**, each mapped to a
distinct, purpose-built prompt rather than one generic "chat with your
PDF" box — because a single vague prompt produces vague, unreliable output
for a domain like this:

| Task | What it does | Why it's separate |
|---|---|---|
| **Simplify** | Plain-language overview + clause-by-clause breakdown + glossary | Rewriting the *whole* document is a different job from finding risk |
| **Highlight risks & obligations** | Buckets clauses into High Attention / Standard Obligations / Protective Terms | Users skim for "what should worry me," not just "what does this mean" |
| **Compare two documents** | Structured diff of key terms + material differences + contradictions | Comparing two versions/offers needs both docs in one context, aligned side-by-side |
| **Ask questions (Q&A)** | Grounded chat over the uploaded document, with history | Free-form follow-ups ("what happens if I terminate early?") |
| **Generate a checklist** | Before-signing / after-signing / red-flags-for-a-professional | Turns understanding into *action* |
| **Prepare for a lawyer** | Summary + prioritized questions + what to bring | Bridges the gap to real professional advice instead of replacing it |

**Core logic decisions:**
- **Grounding, not generation from memory.** Every prompt instructs the
  model to answer *only* from the supplied document text and to say so
  explicitly when something isn't covered, rather than inferring or
  inventing clauses. This is enforced via a strict system prompt
  (`core/prompts.py::SYSTEM_PROMPT`) applied to every call.
- **Never "legal advice."** The system prompt and every screen carry an
  explicit, persistent disclaimer. The model is instructed to explain
  *options and implications*, not tell the user what they must do.
- **Map-reduce for long documents.** Real leases/MSAs can be 20+ pages.
  Rather than truncating silently (and giving wrong answers), long
  documents are chunked on paragraph boundaries (`core/chunker.py`),
  analyzed per-chunk, then merged into one coherent result
  (`core/ai_client.py::_process_long_document`).
- **Business logic is separated from the UI.** `core/` contains pure,
  independently testable functions; `app.py` is a thin Streamlit
  presentation layer. This keeps the system maintainable and lets the
  logic be reused behind a different UI (API, CLI) later.

---

## 3. How the Solution Works (Architecture)

```
┌─────────────────────┐
│      app.py          │  Streamlit UI: file upload, mode selection,
│  (presentation only) │  chat interface, download buttons, disclaimers
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ core/document_parser  │  Validates + extracts text from PDF/DOCX/TXT
│                       │  (size limits, type allow-list, safe errors)
└──────────┬───────────┘
           │ plain text
┌──────────▼───────────┐
│  core/chunker         │  Splits long documents on paragraph boundaries
│                       │  with overlap, for map-reduce processing
└──────────┬───────────┘
           │ chunk(s)
┌──────────▼───────────┐
│  core/prompts         │  Centralized, auditable prompt templates for
│                       │  all 6 tasks + shared system prompt/disclaimer
└──────────┬───────────┘
           │ prompt
┌──────────▼───────────┐
│  core/ai_client       │  Calls the Google Gemini API, retries with
│  (LegalAIClient)      │  backoff, aggregates chunked results, converts
│                       │  all failures into safe user-facing errors
└──────────┬───────────┘
           │
   Gemini (Google AI Studio, free tier)
```

**Flow example — "Simplify a document":**
1. User uploads a `.pdf` → `document_parser.parse_document()` validates
   size/type and extracts text (or raises a clear, safe error).
2. If the text is long, `chunker.chunk_text()` splits it into overlapping
   sections.
3. Each section is sent through `prompts.build_simplify_prompt()` and
   `LegalAIClient.simplify_document()` to Gemini.
4. Results from all sections are merged into one coherent summary.
5. Streamlit renders the summary with a persistent disclaimer banner and
   offers a Markdown download.

---

## 4. Gen AI Services Used

| Service | Where it's used | Purpose |
|---|---|---|
| **Google Gemini API** (`gemini-2.0-flash`, via the official `google-genai` Python SDK) | `core/ai_client.py`, called from every one of the 6 task handlers in `app.py` | All natural-language understanding and generation: document simplification, risk/clause extraction, document comparison, grounded Q&A, checklist generation, and lawyer-consultation prep |

Gemini was chosen specifically because Google AI Studio issues a **free
API key with a generous daily quota and no credit card requirement**,
making the whole submission runnable end-to-end at zero cost.

All prompt engineering (system prompt, task-specific instructions,
grounding/anti-hallucination rules, and the "not legal advice" framing)
lives in `core/prompts.py` and is applied uniformly through
`core/ai_client.py`. No other GenAI provider is used, so behavior is
consistent and auditable end-to-end.

*(The model string is configurable via the `GEMINI_MODEL` environment
variable if you want to point it at a different Gemini model, e.g.
`gemini-1.5-flash`.)*

### Getting a free API key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with any Google account
3. Click **"Create API key"** — no billing information is required for
   the free tier
4. Copy the key into `.env` (local) or your platform's secrets manager
   (deployed) as `GOOGLE_API_KEY`

---

## 5. Assumptions Made

- Users upload **text-based** documents. Scanned/image-only PDFs with no
  extractable text layer are explicitly rejected with a clear message
  (OCR is out of scope for this submission).
- The assistant targets **individuals**, not law firms — output favors
  plain language over legal precision/citation formatting.
- One user session works with one (or two, for comparison) documents at a
  time; there's no persistent multi-document case management, and nothing
  is stored server-side beyond the active session — this is a privacy
  choice, not a limitation to work around.
- "Comparison" assumes both documents are broadly the same *type* of
  document (e.g. two lease offers, two contract drafts) rather than
  unrelated documents.
- Legal domain and jurisdiction are **not assumed**. The assistant never
  states jurisdiction-specific law as fact unless it's present in the
  uploaded document itself — this is a deliberate safety choice, not a
  gap.

---

## 6. Security & Privacy Notes

- API keys are **never hardcoded** — resolved from `st.secrets` (cloud) or
  environment variables (local), see `app.py::get_api_key`.
- Uploaded files are validated by **extension allow-list and size limit
  before parsing** (`core/document_parser.py`) to reduce exposure to
  malformed/oversized files.
- Uploaded documents are processed **in-memory only**, scoped to the
  Streamlit session — nothing is written to disk or a database.
- All internal/library exceptions are caught and converted into safe,
  generic user-facing messages — raw tracebacks/API internals are never
  shown in the UI.
- `.env` and `.streamlit/secrets.toml` are git-ignored so secrets can't be
  committed by accident.

---

## 7. Running Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then add your free GOOGLE_API_KEY into .env

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Running tests

```bash
python -m pytest tests/ -v
```

26 unit tests cover document parsing (valid/invalid files, size limits,
encodings, PDF/DOCX/TXT extraction), text chunking (boundaries, overlap,
content preservation, edge cases), and prompt construction (grounding
rules, disclaimers, all 6 task prompts). AI-client network calls are kept
out of unit tests by design — all decision logic is pushed into pure,
independently testable functions.

---

## 8. Deployment (Streamlit Community Cloud — free, ~5 minutes)

1. Push this repository to a **public** GitHub repo (single `main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"** → select your repo, branch `main`, main file
   `app.py`. Under **Advanced settings**, set the Python version to
   **3.11** (matches `runtime.txt`).
4. Click **"Advanced settings" → "Secrets"** and add your free key
   (get one at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)):
   ```toml
   GOOGLE_API_KEY = "AIza...your-free-key..."
   ```
5. Click **Deploy**. Streamlit installs `requirements.txt` and starts the
   app automatically. You'll get a public URL like
   `https://your-app-name.streamlit.app`.
6. Any future `git push` to `main` auto-redeploys the app.

### Alternative: Render.com

1. Create a new **Web Service** on [render.com](https://render.com), point
   it at your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add an environment variable `GOOGLE_API_KEY` under the service's
   **Environment** tab.
5. Deploy.

---

## 9. Changes Made in the Deployed Version

- **Added `runtime.txt` pinning Python to `3.11`.** Streamlit Community
  Cloud's default image initially provisioned Python 3.14 at deploy time,
  and `pillow` (a transitive dependency pulled in by `streamlit==1.38.0`)
  has no prebuilt wheel for 3.14. The platform fell back to compiling
  Pillow from source and failed on missing `zlib` headers. Pinning the
  runtime to `3.11` — a version Pillow ships prebuilt wheels for —
  resolves the install without touching any application code.
- **Uses the Google Gemini API** (`google-genai` SDK) exclusively, running
  entirely on Google AI Studio's free tier — no paid API dependency
  anywhere in the deployed app.
- Configured `GOOGLE_API_KEY` via the hosting platform's secrets manager
  instead of a local `.env` file (local `.env` is git-ignored and never
  present in the deployed environment).
- Set `MAX_UPLOAD_MB` via environment variable to match the platform's
  request size limits.
- No application code changes were required between local and deployed
  versions beyond the above — the app reads configuration exclusively
  from environment/secrets, which was a deliberate design choice for
  portability (see `app.py::get_api_key`).

---

## 10. Project Structure

```
.
├── app.py                       # Streamlit UI (presentation layer only)
├── core/
│   ├── ai_client.py              # Gemini API wrapper: retries, chunk aggregation
│   ├── document_parser.py        # Safe PDF/DOCX/TXT text extraction
│   ├── chunker.py                 # Paragraph-aware text chunking
│   └── prompts.py                  # All prompt templates + system prompt
├── tests/
│   ├── test_chunker.py
│   ├── test_document_parser.py
│   └── test_prompts.py
├── .github/workflows/tests.yml   # CI: runs pytest on every push
├── .streamlit/config.toml        # Accessible theme (contrast, fonts)
├── requirements.txt
├── runtime.txt                   # Pins Python 3.11 for Streamlit Cloud
├── .env.example
└── README.md
```

---

## 11. Accessibility Notes

- High-contrast theme defined in `.streamlit/config.toml`.
- The legal disclaimer is rendered with an explicit `role="note"` and
  `aria-label` for screen readers.
- All interactive controls use Streamlit's native, keyboard-navigable
  widgets (no custom unlabeled click targets).
- Output is always plain Markdown text (not embedded in images), so it
  works with screen readers and browser text-to-speech/translation tools.
- Language throughout the app and prompts is deliberately plain and
  jargon-light, which is itself an accessibility feature for users with
  cognitive load concerns or limited legal literacy.

---

## Disclaimer

This tool provides general legal information for educational purposes. It
is **not** a law firm, does **not** provide legal advice, and does **not**
create an attorney-client relationship. Always consult a licensed
attorney for decisions with real legal, financial, or personal
consequences.
