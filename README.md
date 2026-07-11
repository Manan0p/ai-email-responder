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
[Groq API — Llama 3.3 70B] --> Generated Reply
    |
    v
[Evaluation Engine] --> Multi-dimensional accuracy report
```

**Pipeline**: New email → embed → retrieve similar past email-reply pairs → build few-shot prompt → generate reply via Groq (Llama 3.3 70B) → evaluate against reference.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at https://console.groq.com/
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

We hand-author **20 scenario templates** across 5 business email categories, then use an LLM to expand each into 5 realistic variations — yielding **100 diverse pairs**.

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

The `metadata.key_actions` field is critical for evaluation — it defines what a correct reply should accomplish. The model generates these as freeform natural-language phrases (e.g. `["Reschedule", "Confirm Availability"]` or `["Schedule meeting", "Discuss project status"]`) rather than a fixed controlled vocabulary — see the Evaluation section for how coverage is measured against this.

### Dataset Validation

`data/generate_dataset.py --validate` checks structural correctness (required fields, split counts) **and** a content-quality check that flags any entry whose body matches the fallback template pattern verbatim — added after an earlier run silently produced generic stub content on every API failure (see Debugging Journey below). Do not proceed with evaluation unless validation reports zero issues.

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
4. **Generator** (`src/generator.py`) — Builds augmented prompt, calls Groq (`llama-3.3-70b-versatile`)
5. **Pipeline** (`src/pipeline.py`) — Orchestrates load → index → retrieve → generate. Indexes **only the 80 train-split emails**, verified by inspecting `index_dataset(split="train")` directly — the test set is never embedded into the vector store, so retrieval cannot leak test answers into the few-shot context.

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
| Key Action Coverage | Completeness check against metadata, matched at the word level (see note below) |

**Key Action Coverage implementation note:** since `metadata.key_actions` is freeform LLM-generated text rather than a fixed vocabulary, coverage is checked by looking for significant individual words from each action phrase (e.g. "confirm", "reschedule", "meeting") appearing anywhere in the generated reply, rather than requiring the full phrase verbatim. This is intentionally more lenient than exact-phrase matching — a real limitation, not a precision claim.

### Layer 2: LLM-as-a-Judge (Nuanced)

Groq (Llama 3.3 70B) scores each reply on all 5 dimensions (1-5 scale) with justifications. This is the primary quality signal.

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

## Results — Full 20-Example Test Set

**Overall Composite: 0.7423 / 1.00**

| Metric | Score |
|---|---|
| ROUGE-L | 0.341 |
| BERTScore | 0.382 |
| Sentence Similarity | 0.821 |
| Key Action Coverage | 0.958 |
| LLM Judge — Semantic Accuracy | 4.55 / 5 |
| LLM Judge — Intent Alignment | 4.55 / 5 |
| LLM Judge — Tone | 4.80 / 5 |
| LLM Judge — Completeness | 4.35 / 5 |
| LLM Judge — Coherence | 5.00 / 5 |

Individual response composites range from 0.665 to 0.793 across all 20 examples, with no two scores identical — genuine per-response variance, not a flat or templated result.

### By Category

| Category | Composite |
|---|---|
| Customer Support | 0.763 |
| Sales/Partnership | 0.749 |
| HR/Internal | 0.737 |
| Meeting Scheduling | 0.738 |
| Project Updates | 0.725 |

## Debugging Journey

Building this system surfaced three distinct, real bugs across two model-provider migrations — each caught by deliberately checking for suspicious uniformity rather than trusting a plausible-looking number.

**Bug 1 — Dataset generation silently failed on every entry (Gemini era).** The dataset generator hardcoded an incorrect model name, causing every generation call to fail, exhaust retries, and silently fall back to a stub template that echoed the scenario description back as both the "incoming email" and the "reference reply." The dataset still passed schema validation, since validation only checked structure, not content. First evaluation run reported a suspicious flat 0.925 composite with every automated metric at 1.0000 and every judge score at 5.0 — a sign the pipeline was comparing near-identical template text to itself, not real generation to a real reference.

**Bug 2 — Key Action Coverage was silently 0.0 for every response, even after the dataset was fixed.** The coverage metric matched against a curated dictionary of snake_case action identifiers, but real LLM-generated `key_actions` are freeform natural-language phrases (`"Schedule meeting"`, `"Reschedule"`) that never matched. Fixed by falling back to word-level matching against the action phrase itself when no dictionary entry exists.

**Bug 3 — Groq's daily token quota (100K/day, free tier) was exhausted partway through a 100-entry dataset generation run**, causing the same silent-fallback failure mode as Bug 1 to reappear on ~48 entries — this time caught immediately, because the content-quality validation check added after Bug 1 flagged every affected entry by name before any evaluation was run against corrupted data.

The final, verified composite score of **0.7423** reflects a dataset confirmed entry-by-entry to contain real generated content, and an evaluation run confirmed to produce non-flat, individually distinct scores per response — not the first plausible-looking number the pipeline produced.

### The judge catches real errors, not just rubber-stamps

Judge scores show genuine per-response variance (semantic accuracy ranges 4-5 across the 20 test emails, not a flat ceiling), and completeness (4.35) scoring meaningfully lower than coherence (5.00) shows the judge distinguishing between "well-written" and "fully addresses the request" rather than defaulting to uniform high praise.

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

Schema validation checks structure (required fields, split counts) **and** content quality (flags any entry whose body matches the fallback template pattern verbatim). Do not proceed to evaluation unless this reports zero issues.

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
- **Single LLM provider**: Groq (Llama 3.3 70B) is used for dataset generation, reply generation, and judging. This creates real circularity risk — the judge has been shown to catch genuine factual errors and shows non-flat score distribution (see Results), which is a partial mitigation, but a production system should use different model providers for generation and evaluation to fully address this.
- **Free-tier rate/quota limits materially affect reliability**: Groq's free-tier daily token cap is tight enough relative to this dataset's size that generation runs need to be validated for silent fallback failures every time, not just once.
- **No personalization**: The system generates generic professional replies. Real email assistants would adapt to the user's writing style.
- **English only**: The dataset and evaluation are English-focused.
- **Static retrieval**: The vector store is built once from the training set. A production system would continuously learn from new email-reply pairs.
- **No conversation threading**: Each email is treated independently. Multi-turn email threads would need additional context management.
- **Key action matching is word-level, not semantic**: an action is credited as "covered" if a significant word from its description appears anywhere in the reply, which can overcredit replies that use the right vocabulary without truly performing the action. A more robust version would check semantic entailment per action rather than word presence.

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | Groq API (generation + judging) |
| `chromadb` | Local vector database |
| `sentence-transformers` | Embedding generation |
| `rouge-score` | ROUGE-L metric |
| `bert-score` | BERTScore metric |
| `numpy` | Numerical operations |
| `python-dotenv` | Environment variable loading |
| `rich` | Console output formatting |

## AI Tool Usage

This project was built with assistance from AI tools:
- **Groq (Llama 3.3 70B)** — Used at runtime for email reply generation, dataset expansion, and LLM-as-a-Judge evaluation