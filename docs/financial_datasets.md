# Research: HuggingFace Financial Text Datasets

## Executive Summary

Five public, no-auth HuggingFace datasets cover the financial text domain well. For pretraining-style work (raw text density), `ashraq/financial-news-articles` (306k articles, has `text` column) and `PleIAs/SEC` (373k 10-K filings, has `text` column) are the strongest. For instruction/fine-tuning, `sujet-ai/Sujet-Finance-Instruct-177k` aggregates 18 datasets into 177k QA pairs. All five options below require zero authentication and load via standard `load_dataset()`.

---

## Ranked Options

### 1. `ashraq/financial-news-articles` — RECOMMENDED for smoke tests

**Best for:** Pretraining / LM fine-tuning. Has a real `text` column with full article bodies.

```python
from datasets import load_dataset

ds = load_dataset("ashraq/financial-news-articles", split="train")
# or a small slice:
ds = load_dataset("ashraq/financial-news-articles", split="train[:5000]")
```

| Property | Value |
|---|---|
| Rows | 306,000 |
| Columns | `title`, `text`, `url` |
| Download size | ~few hundred MB |
| Auth required | No |
| License | Not specified (scraped news) |
| Text quality | Full article bodies, 0–218k chars per doc |

**Why #1:** Direct `text` column with full article content, not just headlines. 306k examples is large enough to be meaningful, small enough for a quick test run. Single split, no config gymnastics.

---

### 2. `PleIAs/SEC` — RECOMMENDED for domain-rich pretraining

**Best for:** High-quality, long-form financial text (10-K annual reports, 1993–2024). Best raw text quality of the lot.

```python
from datasets import load_dataset

# Streaming for inspection (avoids downloading 5GB)
ds = load_dataset("PleIAs/SEC", split="train", streaming=True)
sample = list(ds.take(500))

# Slice without streaming
ds = load_dataset("PleIAs/SEC", split="train[:1000]")
```

| Property | Value |
|---|---|
| Rows | ~373,000 (first 5GB subset) |
| Columns | `filename`, `id`, `year`, `cik`, `text`, `word_count`, `character_count` |
| Download size | 5GB (first subset) |
| Auth required | No |
| License | CC0-1.0 (public domain) |
| Text quality | Verbatim SEC 10-K filings, avg 34k words/doc |

**Why #2:** Highest text quality for domain pretraining. CC0 license is production-safe. Use streaming or slice syntax for smoke tests — don't pull the full 5GB blindly.

---

### 3. `sujet-ai/Sujet-Finance-Instruct-177k` — RECOMMENDED for instruction tuning

**Best for:** SFT / RLHF alignment on financial tasks. Aggregates 18 source datasets.

```python
from datasets import load_dataset

ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
```

| Property | Value |
|---|---|
| Rows | 177,597 |
| Columns | `inputs`, `answer`, `system_prompt`, `user_prompt`, `task_type`, `dataset`, `index_level`, `conversation_id` |
| Download size | 104 MB (Parquet) |
| Auth required | No |
| License | Apache 2.0 |
| Task types | Sentiment, NER, summarization, QA, classification |

**Why #3:** Best coverage of financial NLP tasks for fine-tuning. 104MB makes it fast to download. The `task_type` column lets you filter to specific subtasks. No `text` column per se — use `inputs` + `answer` for training.

---

### 4. `JanosAudran/financial-reports-sec` — CONSIDER (sentence-level SEC)

**Best for:** Sentence-level classification on 10-K filings with stock return labels.

```python
from datasets import load_dataset

# Small config (~240k sentences)
ds = load_dataset("JanosAudran/financial-reports-sec", "small_lite")

# Large config (~20.5M sentences) — use streaming
ds = load_dataset("JanosAudran/financial-reports-sec", "large_lite", streaming=True)
```

| Property | Value |
|---|---|
| Rows | 240k (`small_lite`) / 20.5M (`large_lite`) |
| Columns | `cik`, `sentence`, `section`, `labels` (1d/5d/30d returns), `filingDate`, `docID` |
| Auth required | No |
| License | Apache 2.0 |
| Text quality | Individual sentences (not full docs) |

**Why #4 not higher:** Text is sentence-fragments, not continuous prose — less useful for LM pretraining. The `sentence` column is the text field. Dataset viewer had timeout issues; load programmatically.

---

### 5. `gbharti/finance-alpaca` — CONSIDER (small, clean QA)

**Best for:** Quick instruction fine-tuning baseline. Small and fast.

```python
from datasets import load_dataset

ds = load_dataset("gbharti/finance-alpaca", split="train")
```

| Property | Value |
|---|---|
| Rows | 68,900 |
| Columns | `instruction`, `input`, `output`, `text` |
| Auth required | No |
| License | MIT |
| Text quality | GPT-3.5 generated + FiQA + Alpaca |

**Why #5:** Has a combined `text` column (convenient). MIT license. But partially synthetic (GPT-3.5), smaller, and less domain-rich than the options above.

---

## Avoid for This Use Case

| Dataset | Reason |
|---|---|
| `takala/financial_phrasebank` | Only 4,840 sentences; sentiment labels only; too small for LM training |
| `ashraq/financial-news` | 1.85M rows but only **headlines** (no article body), not useful for LM pretraining |
| `Josephgflowers/Finance-Instruct-500k` | 518k rows but low provenance transparency; system/user/assistant schema only |
| `arthrod/SEC_filings_1994_2024` | Metadata only (CIK, form type, dates) — no text content |

---

## Quick Smoke Test Recipe

```python
from datasets import load_dataset

# Option A: news articles (fast, ~300MB)
ds = load_dataset("ashraq/financial-news-articles", split="train[:2000]")
texts = ds["text"]

# Option B: SEC filings (streaming, no download)
ds = load_dataset("PleIAs/SEC", split="train", streaming=True)
texts = [ex["text"] for ex in ds.take(200)]

# Option C: instruction pairs (104MB total, then filter)
ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
ds_sentiment = ds.filter(lambda x: x["task_type"] == "sentiment_analysis")
```

---

## Sources

- [ashraq/financial-news-articles](https://huggingface.co/datasets/ashraq/financial-news-articles)
- [PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC)
- [sujet-ai/Sujet-Finance-Instruct-177k](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)
- [JanosAudran/financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)
- [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
- [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)
- [ashraq/financial-news](https://huggingface.co/datasets/ashraq/financial-news)
- [Josephgflowers/Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)
