# AI Email Suggested-Response System

An end-to-end system that generates suggested replies to incoming emails using **Retrieval-Augmented Generation (RAG)** and evaluates them with a **multi-dimensional accuracy framework**.

## Architecture

```
Incoming Email
    |
    v
[Embedding Model] --> [ChromaDB Vector Store] --> Retrieve top-3 similar past emails
    |
    v
[Prompt Builder] --> System instruction + Retrieved examples + New email
    |
    v
[Google Gemini API] --> Generated Reply
    |
    v
[Evaluation Engine] --> Multi-dimensional accuracy report
```

**Pipeline**: New email → embed → retrieve similar past email-reply pairs → build few-shot prompt → generate reply via Gemini → evaluate against reference.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
# Get one free at https://aistudio.google.com/
```

### 3. Generate the dataset

```bash
python data/generate_dataset.py
```

This creates `data/email_dataset.json` with 100 synthetic email-reply pairs (80 train / 20 test).

### 4. Run the full demo

```bash
python scripts/run_demo.py
```

## Dataset

### Why Synthetic?

| Approach | Pros | Cons |
|----------|------|------|
| **Enron corpus** | Real data, large scale | Heavy cleaning needed; privacy concerns; domain-specific |
| **LLM-generated** | Fast, diverse | Circular reasoning if same LLM generates and evaluates |
| **Synthetic (chosen)** | Full quality control; no privacy issues; designed for evaluation | Not "real" data |

We hand-author **20 scenario templates** across 5 business email categories, then use Gemini to expand each into 5 realistic variations — yielding **100 diverse pairs**.

### Categories

| Category | Scenarios | Examples |
|----------|-----------|---------|
| Meeting Scheduling | 4 | Schedule, reschedule, cancel, confirm |
| Project Updates | 4 | Status request, deadline extension, blocker, milestone |
| Customer Support | 4 | Complaint, refund, product inquiry, billing |
| HR/Internal | 4 | PTO request, onboarding, policy, feedback |
| Sales/Partnership | 4 | Cold outreach, proposal follow-up, pricing, contract |

### Schema

Each entry includes:
- `incoming_email` — from, to, subject, body
- `expected_reply` — subject, body (human-quality reference)
- `metadata` — tone, intent, key_actions, complexity
- `split` — "train" (80) or "test" (20)

The `metadata.key_actions` field is critical for evaluation — it defines what a correct reply *must* do (e.g., `["apologize", "offer_refund", "provide_timeline"]`).

## Response Generator (RAG Pipeline)

### Why RAG + Few-Shot?

| Approach | Why Not |
|----------|---------|
| Zero-shot | No grounding in historical data |
| Fine-tuning | Needs 1000s of examples; expensive |
| **RAG + Few-Shot** | Grounds in real examples; adapts without retraining; interpretable |

### Components

1. **Embeddings** (`src/embeddings.py`) — `sentence-transformers/all-MiniLM-L6-v2` for local, free embedding generation
2. **Vector Store** (`src/vector_store.py`) — ChromaDB persistent collection with cosine similarity
3. **Retriever** (`src/retriever.py`) — Finds top-k similar past emails, filters exact matches to prevent data leakage
4. **Generator** (`src/generator.py`) — Builds augmented prompt, calls Gemini (`gemini-2.0-flash`)
5. **Pipeline** (`src/pipeline.py`) — Orchestrates load → index → retrieve → generate

### Prompt Design

```
System: You are a professional email assistant...

Past Examples (retrieved):
  Example 1: [similar email → reply]
  Example 2: [similar email → reply]
  Example 3: [similar email → reply]

New Email:
  From: ...
  Subject: ...
  Body: ...

Generate the reply:
```

## Evaluation System

### The 5 Dimensions of "Accuracy"

Exact match is meaningless for email replies. We define accuracy across 5 dimensions:

| Dimension | What It Measures |
|-----------|-----------------|
| **Semantic Similarity** | Same meaning as reference? |
| **Intent Alignment** | Same goals addressed? |
| **Tone Appropriateness** | Right professional tone? |
| **Completeness** | All points addressed? |
| **Coherence** | Well-written and professional? |

### Layer 1: Automated Metrics (Fast, Deterministic)

| Metric | What It Catches |
|--------|-----------------|
| ROUGE-L | Lexical/structural overlap |
| BERTScore | Token-level semantic similarity |
| Sentence Similarity | Overall meaning alignment |
| Key Action Coverage | Completeness check against metadata |

### Layer 2: LLM-as-a-Judge (Nuanced)

Google Gemini scores each reply on all 5 dimensions (1-5 scale) with justifications. This is the primary quality signal.

### Layer 3: Composite Score

```
composite = 0.10 * ROUGE-L
          + 0.20 * BERTScore
          + 0.15 * Sentence Similarity
          + 0.15 * Key Action Coverage
          + 0.40 * LLM Judge (normalized)
```

The LLM judge gets the highest weight (40%) because it's the only metric that captures tone, intent, and nuanced quality. Automated metrics serve as deterministic anchors.

### Validation: Proving Metrics Work

We run controlled degradation tests to verify our metrics aren't just numbers:

1. **Perfect baseline** — reference vs. itself → ~1.0
2. **Paraphrase** — different words, same meaning → high semantic score
3. **Wrong intent** — refund reply for meeting email → low intent score
4. **Incomplete** — missing key actions → low coverage score

These are implemented as automated tests in `tests/test_evaluation.py`.

## How to Run

### Generate a reply for a single email

```bash
python scripts/run_pipeline.py \
  --subject "Meeting Tomorrow" \
  --body "Can we reschedule our 2pm meeting to 4pm?" \
  --from "colleague@company.com"
```

### Run evaluation on the test set

```bash
# Full evaluation with LLM judge
python scripts/run_evaluation.py

# Without LLM judge (faster, automated metrics only)
python scripts/run_evaluation.py --no-judge

# Save reports
python scripts/run_evaluation.py --output-json results.json --output-md results.md
```

### Run the full demo

```bash
python scripts/run_demo.py

# Skip demo emails, just evaluate
python scripts/run_demo.py --eval-only
```

### Validate the dataset

```bash
python data/generate_dataset.py --validate
```

### Run tests

```bash
# Dataset validation
python -m pytest tests/test_dataset.py -v

# Pipeline tests (needs dataset)
python -m pytest tests/test_pipeline.py -v

# Evaluation tests
python -m pytest tests/test_evaluation.py -v

# All tests
python -m pytest tests/ -v
```

## Trade-offs & Limitations

- **Dataset size**: 100 pairs is sufficient for demonstration but not for production. A real system would need 1000+ examples across more categories.
- **Single LLM**: Using Gemini for both generation and judging creates some circularity. A production system should use different models for generation and evaluation.
- **No personalization**: The system generates generic professional replies. Real email assistants would adapt to the user's writing style.
- **English only**: The dataset and evaluation are English-focused.
- **Static retrieval**: The vector store is built once from the training set. A production system would continuously learn from new email-reply pairs.
- **No conversation threading**: Each email is treated independently. Multi-turn email threads would need additional context management.

## Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Google Gemini API (generation + judging) |
| `chromadb` | Local vector database |
| `sentence-transformers` | Embedding generation |
| `rouge-score` | ROUGE-L metric |
| `bert-score` | BERTScore metric |
| `numpy` | Numerical operations |
| `python-dotenv` | Environment variable loading |
| `rich` | Console output formatting |

## AI Tool Usage

This project was built with assistance from AI tools:
- **Claude** (Anthropic) — Used for code generation, architecture design, and documentation
- **Google Gemini** — Used at runtime for email reply generation, dataset expansion, and LLM-as-a-Judge evaluation
