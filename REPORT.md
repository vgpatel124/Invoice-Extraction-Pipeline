# Invoice Extraction Pipeline: Report

## Summary

- **Pipeline:** `gemini-3.8-flash` reads the original PDF or image. It uses structured output, local validation and one retry with feedback.
- **Accuracy:** on the 10 labelled documents, prompt v1 scores **100% on every field** under the Normalized rules. It was written before looking at any labels and was never tuned. Under Strict rules, "all fields correct" is 67%; every Strict miss is vendor capitalisation. All 66 line-item rows are correct, including a 24-row two-page invoice, a phone photo and two scans.
- **Cost:** **$0.0023 per document** at current prices ($0.0045 from January 2027), retries included. The whole project spent **$0.0600** of the $2.00 budget (Section 4).
- **What did not work:** sending PyMuPDF text instead of the file. Every scan and photo was silently marked "not an invoice".
- **Bonus:** a `needs_review` flag and a Streamlit frontend (`app.py`). The flag catches all 3 silent failures from that experiment. It also flags 4 of 10 clean labelled documents.
- **Outputs:** L01 to L10 come from `gemini-3.8-flash`. U01 to U10 come from `gemini-3.5-flash-lite`, because 3.8 Flash was overloaded (503/504 errors) during the final run (Section 4). Flash-Lite matched 3.8 Flash exactly on the labelled set.
- **Caveat:** 10 labelled documents is a small sample. These numbers show the pipeline handles these layouts. They do not prove it is error-free.

## 1. Approach

- **Model:** `gemini-3.8-flash` through the `google-genai` SDK, one call per document. I chose it because it is the newest Flash model. It reads PDFs and images natively, supports structured output, and costs well under a cent per invoice.
- **Input:** the original PDF or image bytes are sent as they are. Gemini reads PDF pages directly, including any text layer. So digital PDFs, scans and photos all use one code path, with no OCR step. PyMuPDF is only used to check the file, count pages (10-page limit) and detect a text layer.
- **Prompting:** a system prompt in `src/prompt.py` (version `v1`). It does three things:
  1. Asks whether the document is an invoice. It gives a definition and lists look-alikes that are not invoices.
  2. Gives the pack's labelling rules field by field.
  3. Adds a consistency rule: line items + tax should equal the total, but a printed number must never be changed to make it fit.

  The Pydantic model is passed as `response_schema` for structured output.
- **Validation and retry:** the checks are JSON parse, Pydantic, JSON Schema (`schema.json`), totals and sanity. If one fails, the model is called once more. The retry message names the failed check and the numbers, and the thinking level goes from `low` to `medium`. If the output still fails but is schema-valid, it is kept and flagged `needs_review`. Schema-invalid output is never returned.
- **Temperature:** left at the default (1.0). Google advises against lowering it for Gemini 3 models, because it can cause looping and worse output. Reliability comes from structured output, validation and retry instead. This means output is not fully deterministic.
- **API tier:** this submission used a free-tier key, by choice. On the free tier, Google may use inputs to improve its products. Real client invoices should go through a paid key, where inputs are not used. All costs here use paid-tier prices, which is what a client would pay. The actual charge on the free tier was $0.

### What did not work: sending PyMuPDF text instead of the document

**Idea:** text is cheaper than a page image. So send the PDF's text layer as plain text instead of the file. Command: `uv run python eval.py --model gemini-3.5-flash-lite --input-mode text`.

**Why Flash-Lite:** the first try on 3.8 Flash hit the free-tier limit of 20 requests per day after one document (L01, fully correct). The rest of the 3.8 Flash quota was kept for the unlabelled run. Flash-Lite with native input scores 100% Normalized on the labelled set (next subsection), so it is a fair baseline.

| Input (Flash-Lite) | is_invoice | All fields correct (9 invoices) | Line-item rows found | Avg input tokens | Avg cost/doc |
|---|---|---|---|---|---|
| Native file (PDF/JPG bytes) | 10/10 | 9/9 | 66/66 | 1,302 | $0.00139 |
| PyMuPDF text only | 7/10 | 6/9 | 50/66 | 1,031 | $0.00112 |

- **Digital PDFs (6):** text-only was just as accurate, including the 24-row two-page invoice (L04). It used about 20% fewer input tokens.
- **No text layer (3):** L02 (phone photo), L09 and L10 (scans) failed completely. The model got no text, so it returned `is_invoice: false` with every field null.
- **This is a silent failure.** "Not an invoice" is schema-valid and passes every check, so none of the 3 was flagged. In a finance team, these invoices would just be dropped.
- **Conclusion:** the token saving is small and the failure is severe, so the pipeline keeps native input. A text-first design would need an OCR fallback for scans, which is the extra step native input avoids.

### Secondary comparison: 3.8 Flash vs 3.5 Flash-Lite (native input)

| Model | All fields correct | Rows correct | Avg in / out tokens | Avg cost/doc now | Avg cost/doc from Jan 2027 | Typical latency |
|---|---|---|---|---|---|---|
| `gemini-3.8-flash` (thinking `low`) | 9/9 | 66/66 | 1,302 / 343 | $0.00226 | $0.00453 | 3 to 180 s (free tier) |
| `gemini-3.5-flash-lite` (thinking `minimal`) | 9/9 | 66/66 | 1,302 / 402 | $0.00139 | $0.00139 | 1 to 5 s |

- Flash-Lite matched 3.8 Flash on every field, including the phone photo and both scans.
- It costs 38% less now and 69% less from January 2027.
- I expected Flash-Lite to struggle on scans and long tables. On this set, it did not.
- 10 documents is not enough evidence to switch. So 3.8 Flash stays the main model, and Flash-Lite is the first thing to A/B test on more data (Section 7).

## 2. Accuracy

### Matching rules (`src/matching.py`)

Each field is scored two ways. The gap between the two columns shows how much the rules themselves help.

**Strict:** exact equality after trimming whitespace. Numbers must be exactly equal.

**Normalized:**

| Field | Rule |
|---|---|
| Numbers (`tax_total`, `total`, line amounts, quantities) | Strings like "1,200.00" or "₹1,200" are parsed to numbers. Match if within ±0.01 |
| `vendor` | Lowercase, remove punctuation and legal suffixes (pvt, private, ltd, limited, llp, inc, llc, co), collapse spaces. Match if rapidfuzz ratio ≥ 90 |
| `invoice_number` | Exact match, ignoring case and outer spaces. Characters like `/` and `-` are kept |
| `invoice_date`, `currency` | Exact |
| `line_items` | Rows are aligned in printed order (best-match alignment if the counts differ). A row is correct if: description ratio ≥ 85 (ignoring case and spacing), quantity is equal (null must match null), and amount is within ±0.01. The field is correct only if the row count matches and every row is correct |
| Any field | `null` vs a value is always a mismatch |

- **Non-invoices** are scored on `is_invoice`, plus a check that every other field is null. They are left out of the other field averages, so they do not inflate them.
- **"All fields correct"** is counted over invoice documents only.
- A document that ended in a `processing_error` counts as wrong on every field.

### Results: v1 baseline, which is also the final version

Prompt v1 was **not tuned on the labelled set**. It was written from the pack README's labelling rules before any labelled answer was opened. It was not changed after the baseline, because there were no errors to fix.

Run on 2026-11-20 with `gemini-3.8-flash`, thinking `low`, default temperature (`uv run python eval.py`). Raw output: `eval_results/v1_baseline.txt` and `.json`.

| Field | Strict | Normalized |
|---|---|---|
| is_invoice (10 docs) | 10/10 (100%) | 10/10 (100%) |
| vendor | 6/9 (67%) | 9/9 (100%) |
| invoice_number | 9/9 (100%) | 9/9 (100%) |
| invoice_date | 9/9 (100%) | 9/9 (100%) |
| currency | 9/9 (100%) | 9/9 (100%) |
| line_items | 9/9 (100%) | 9/9 (100%) |
| tax_total | 9/9 (100%) | 9/9 (100%) |
| total | 9/9 (100%) | 9/9 (100%) |
| **all fields correct** (9 invoices) | 6/9 (67%) | 9/9 (100%) |

- **Line items:** 66 expected, 66 predicted, 66 correct. Precision, recall and F1 are all 1.000. No missing, extra or wrong rows.
- **Hard cases covered:**
  - a 24-row two-page invoice (L04);
  - rows with no quantity (`null`, not 1);
  - a negative discount row;
  - USD and GBP invoices;
  - invoice numbers with slashes;
  - a JPG phone photo (L02) and two scanned PDFs (L09, L10);
  - one non-invoice (L08, correctly all null).
- **Retries:** none needed. Every first answer passed the totals and sanity checks.
- **Caveat:** 9 invoices and 1 non-invoice is a very small sample. Output also varies at the default temperature, so a fresh run could differ.

### Does the totals check actually fire?

No live document failed it, because all 20 first answers add up. So it was checked in two other ways.

**On real data.** I removed the last line item from the L03 output and passed it to `src/validate.run_checks`. It returns a `totals` issue. This is the exact feedback the retry would send to the model:
```
line items sum to 52,840.00, tax_total is 9,788.40, total is 64,168.40: sum + tax = 62,628.40,
which differs from total by -1,540.00. Re-read the item table (rows on every page, discount and
charge rows, pre-tax amounts) and the tax lines, and copy numbers exactly as printed.
```

**In the test suite** (`uv run pytest`, with model calls scripted, so no API cost). The tests in `tests/test_validate.py` and `tests/test_pipeline.py` check that:
- a totals mismatch reports the numbers;
- a discount with the wrong sign breaks the totals;
- the tolerance scales with large totals;
- a failing first answer triggers exactly one retry, with that feedback, at a higher thinking level;
- an answer that still fails after the retry is kept but flagged `needs_review`;
- a retry that gets worse keeps the better attempt;
- schema-invalid output twice becomes a `processing_error`, and nothing is written.

## 3. Failure analysis

The final setup has no Normalized errors on the labelled set. So this section groups every failure seen anywhere in the project: in Strict scoring, in the experiments and while running. They are ordered by how much damage each could do.

| # | Cause | Where seen | Count | Effect | Caught? |
|---|---|---|---|---|---|
| 1 | Unreadable input marked "not an invoice" | Text-only experiment: L02, L09, L10 | 3 of 10 docs | Whole invoice silently dropped | **No**: schema-valid, all checks pass |
| 2 | Free-tier API limits (504 deadline, 503 overload, 429 daily quota) | Phase 1 test; text-only run on 3.8 Flash; final run of U01, U02 | 2 docs in Phase 1, 9 in the experiment, 2 in the final run | No output for the document | Yes: `processing_error`, exit 1, nothing fake printed |
| 3 | Vendor capitalisation differs from the label | Strict scoring: L05, L07, L09 | 3 of 9 invoices | Strict mismatch only | Handled by the Normalized rule |
| 4 | Response schema rejected by the API | First live call in development | every call | Every extraction would fail | Yes: found on a synthetic invoice before any real call |

**Examples (expected vs got):**

1. **L09** (scanned PDF), text-only mode.
   - Expected: `is_invoice: true`, vendor `Coastal Fresh Seafoods`, total `42682.5`, 5 line items.
   - Got: `is_invoice: false`, every other field `null`.
   - Why: the page has no text layer. The model was sent "(No text could be extracted from this document.)" and correctly said it saw no invoice.
2. **L05**, Strict scoring.
   - Expected vendor: `Orbit Hardware Supplies`. Got: `ORBIT HARDWARE SUPPLIES`.
   - Why: the model copied the header as printed, but the label uses title case. L07 and L09 are the same.
3. **L02**, Phase 1 test.
   - Got: `503 UNAVAILABLE: This model is currently experiencing high demand` after all retries.
   - Result: exit code 1, error JSON on stderr, no output file. The same document worked later and is fully correct.

**Cause details:**

- **1. Unreadable input becomes a "non-invoice".** The checks only test whether an answer is consistent with itself. "Not an invoice, all fields null" is always consistent. In the experiment, the cause was missing text. In real use, a very blurred photo or a blank scan could do the same. It is the only failure that loses an invoice without telling anyone.
- **2. API limits.** The free tier allows 20 requests per day per model. It was also slow: up to 180 s per call, with 504 errors at a 120 s deadline. The fixes: a 300 s timeout, backoff on 5xx errors and per-minute 429s, and immediate failure on a daily-quota 429 (`quota_exhausted`). None of these produce fake data.
- **3. Vendor casing.** This is a labelling style difference, not an extraction error. If a downstream system needs title case, converting an all-caps vendor is a safe extra step. I did not add it, to keep the output "as printed".
- **4. Schema translation.** Pydantic's `extra="forbid"` adds `additionalProperties`, which Gemini's `response_schema` rejects. I removed it from the model sent to Gemini. Extra keys are still rejected locally by the official JSON Schema, and a unit test covers this.

**What to fix first: cause 1.**
- It is the only silent failure, and it hits scans and photos, which are the documents a finance team most wants checked.
- First step, already done: `is_invoice: false` on a document with no text layer now sets `needs_review` (`non_invoice_without_text_layer`, Section 6). This catches all 3 cases from the experiment.
- Next step: get a second model opinion on every `is_invoice: false` (for example Flash-Lite, about $0.0014), and add a low-resolution or low-contrast check for photos.
- Cause 2 comes next and is solved by a paid key. Causes 3 and 4 are already handled.

## 4. Cost per document

**Method.** Token counts come from each response's `usage_metadata` (`prompt_token_count`, `candidates_token_count`, `thoughts_token_count`). They are never estimated. Cost per call:

```
cost = (input_tokens × input_price + (output_tokens + thinking_tokens) × output_price) / 1,000,000
```

- Thinking tokens are billed at the output price.
- Retries are included: each attempt is a separate call, added to the document's cost.
- Cache hits cost nothing. But each document's metadata keeps the cost of the calls that produced it, so the per-document figure does not drop to zero on a cached re-run.

**Prices** (paid tier, standard, USD per 1M tokens). Source: https://aistudio.google.com/docs/pricing (redirected from ai.google.dev/gemini-api/docs/pricing), checked 2026-11-20.

| Model | Input | Output (incl. thinking) |
|---|---|---|
| `gemini-3.8-flash`, through 2026-12-31 (introductory) | $0.75 | $3.75 |
| `gemini-3.8-flash`, from 2027-01-01 | $1.50 | $7.50 |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 |

**Labelled set** (`gemini-3.8-flash`, 10 documents, 10 calls, no retries):

| Metric | Value |
|---|---|
| Avg input tokens / doc | 1,302 |
| Avg output tokens / doc | 305 |
| Avg thinking tokens / doc | 38 |
| **Avg cost / doc, current price** | **$0.00226** |
| Avg cost / doc, from January 2027 | $0.00453 |
| Most expensive doc (L04, 24 rows, 2 pages) | $0.00505 |

At the 2027 price, 1,000 invoices a month cost about $4.50. A retry roughly doubles one document's cost, so the retry rate matters more than the token price.

**Unlabelled set** (`gemini-3.5-flash-lite`, 10 documents, 10 calls, no retries):
- Average **$0.00094 per document**, $0.0094 in total.
- All 10 outputs are schema-valid and pass the totals check.
- 8 are invoices. U06 (a bank statement) and U10 (a cover letter) are marked not an invoice. Together with L08, that makes the pack's 3 non-invoices.
- U02 and U05 are flagged for review as scans or photos.

**Why Flash-Lite for the unlabelled set.** On 2026-11-20, the 3.8 Flash daily quota had reset. But every 3.8 Flash call (U01, U02) failed with `503 UNAVAILABLE` (high demand) or `504 DEADLINE_EXCEEDED` after all 4 backoff attempts, about 10 minutes per document. I stopped the run so failed calls would not use up the quota. Flash-Lite scored the same as 3.8 Flash on the labelled set (Section 1) and took seconds per document. Each document's model is saved in `outputs/_meta/<ID>.json`, and `run_all.py` reuses it on a re-run, so the cache is hit.

**Total project spend:** **$0.0600** of the $2.00 budget, from the budget ledger `.cache/spend.json` at paid-tier prices. This covers development, every eval run, both experiments and all 20 outputs. Failed calls with no usage data cost nothing. The project actually ran on a free-tier key, so nothing was billed. The ledger shows what a paid key would have cost.

## 5. Bad-input handling

**Non-invoices (required).** The prompt defines an invoice and lists look-alikes that are not invoices: purchase orders, quotations, bank statements, cover letters, delivery notes and contracts. The schema check makes sure a non-invoice has every other field `null`. L08, U06 and U10 are all handled this way.

**Extra case chosen: corrupt, empty or unsupported files.** These are caught locally before any API call, so they cost nothing.

| Input | Error type |
|---|---|
| 0-byte file | `empty_file` |
| Truncated or damaged PDF, or an image that does not fully decode | `corrupt_file` |
| Wrong extension, or content that does not match it (for example a text file renamed to `.jpg`) | `unsupported_file` |
| Password-protected PDF | `unsupported_file` |
| More than 10 pages | `too_many_pages` |
| File over 20 MB | `file_too_large` |

Each one gives exit code 1, an error JSON on stderr, nothing on stdout, and a `processing_error` in the metadata. Tests cover them with generated fixtures.

**API failures are also handled:**
- Timeouts, 5xx errors and per-minute rate limits are retried with exponential backoff (4 attempts).
- A daily quota error fails at once as `quota_exhausted`.
- A call that could push spend over the $2 budget is refused as `budget_exceeded`.

## 6. Human-review flag (bonus)

`needs_review` is only set on finished extractions. A failed one is a `processing_error`, never both. The reasons are saved in `review_reasons` in each document's metadata (`src/review.py`).

| Reason | Fires when | Why |
|---|---|---|
| `validation_failed_after_retry` | Totals or sanity checks still fail after the retry | The numbers do not add up as extracted |
| `scan_or_photo` | The input is an image, or a PDF with no text layer | Misreads are much more likely than on digital PDFs |
| `ambiguous_date` | Day and month are both 12 or less (and differ), and the page text does not settle the order. That means there is no month-name version of the date, and no other numeric date on the page that can only be read one way | A swapped day and month is a silent, plausible-looking error |
| `many_line_items` | More than 15 line items | Long tables are where rows get dropped or merged |
| `missing_fields:<names>` | An invoice with an empty header field | The model could not find something every invoice should have |
| `non_invoice_without_text_layer` | `is_invoice: false` on a scan or photo | Section 3, cause 1: an unreadable invoice looks exactly like a non-invoice |

**Does the flag catch the errors?** Measured with `uv run python eval.py` on the labelled set, from cache, at no extra cost.

| Configuration | Flagged, has error | Flagged, no error | Not flagged, has error | Not flagged, no error |
|---|---|---|---|---|
| Production: 3.8 Flash, native input | 0 | 4 | 0 | 6 |
| Text-only experiment (Flash-Lite) | **3** | 1 | **0** | 6 |

- **Production:** there are no errors to catch, so all 4 flags are false positives: L02, L09 and L10 (`scan_or_photo`; L10 also `ambiguous_date`) and L04 (`many_line_items`). A reviewer would check 4 of 10 documents.
- **Text-only experiment:** the flag catches all 3 silently dropped invoices (recall 3/3), with 1 false positive (precision 3/4).
- **Date rule:** only L01 (`2026-04-08`) and L10 (`2026-09-03`) have a day and month that could be swapped. L01's page text confirms the reading, so it is not flagged. L10 is a scan with no text to check against, so it is.
- **Honest caveat:**
  - The `non_invoice_without_text_layer` rule was written after seeing the text-only failure, so its 3/3 is not an independent test. The other rules were planned before any results.
  - With only 3 real errors, these numbers show the flag works as designed, not how well it would generalise.
  - Flagging 40% of clean documents is the price. The flag favours recall, because a missed invoice costs more than a second look.

## 7. Limitations and next steps

- **Small eval set.** 9 invoices and 1 non-invoice cannot tell a 95% pipeline from a 100% one. The first next step is to label more documents, especially poor photos, multi-page invoices and non-invoices that look like invoices.
- **Silent "not an invoice" on unreadable input** (Section 3, cause 1). Scans and photos are now flagged. But a blurry photo with some readable text, or a digital PDF with a garbage text layer, would not be caught.
- **Review flag precision.** 4 of 10 clean documents are flagged. More labelled errors are needed to reduce false positives without losing recall.
- **Flash-Lite may be good enough.** It matched 3.8 Flash here at a third of the 2027 cost and much lower latency. It is worth an A/B test on a bigger labelled set.
- **Free tier.** Slow calls (up to 3 minutes), the 20-requests-per-day limit and data use make it unsuitable for real client documents. Use a paid key.
- **Mixed models in outputs/.** Unlabelled outputs come from Flash-Lite and labelled ones from 3.8 Flash (Section 4). With a reliable paid key, re-run U01 to U10 on 3.8 Flash:
  ```bash
  uv run python run_all.py --only U01 U02 U03 U04 U05 U06 U07 U08 U09 U10 --model gemini-3.8-flash
  ```
- **Non-determinism.** At the default temperature, repeated runs can differ. Only one uncached run was made per setup, to stay within the free-tier quota, so run-to-run variance was not measured.
- **Scope.** One currency per invoice, no special handling for handwriting, a 10-page limit, and no per-line tax breakdown (the schema does not ask for it).

## Tuning log

| Version | Change | Trigger |
|---|---|---|
| v1 | First prompt, written only from the pack README's labelling rules and the brief's totals check. The labelled answers were not opened while writing it. | n/a |
| infra | Request timeout raised from 120 s to 300 s. API backoff raised from 3 to 4 attempts (5 to 60 s waits). Not a prompt change. | Free-tier calls returned 504 and 503 errors on the first Phase 1 test |
| (none) | No prompt change after the v1 baseline. | v1 scored 100% Normalized, so there were no errors to fix. Changing the prompt anyway would only fit it to these 10 documents |
| infra | Daily-quota 429 errors are no longer retried; they fail at once as `quota_exhausted`. Per-minute 429s are still retried. | The first text-only run kept retrying a quota that resets hours later |
| infra | U01 to U10 extracted with `gemini-3.5-flash-lite` instead of `gemini-3.8-flash`. `run_all.py` now reuses each document's saved model unless `--model` is given. Not a prompt change. | 3.8 Flash returned 503/504 overload errors on every call during the final run |
