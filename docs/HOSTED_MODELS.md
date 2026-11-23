# Hosted models: `bleediff gemini`, `bleediff openai`, `bleediff anthropic`

Usage guide for the three hosted adapters (added in v0.5.0; the no-answer split and FLR-v2 diagnostic were added in v0.6.0). They share one engine (`bleediff/hosted.py`); each provider module (`gemini.py`, `openai_chat.py`, `anthropic_messages.py`) holds only that API's request and
response shapes. Analysis is `bleediff/answers.py`. Everything here is exploratory and answer-level.

## 1. Two measurement types: DIRECT_PROBABILITY versus ANSWER_ONLY
| | `DIRECT_PROBABILITY` | `ANSWER_ONLY` |
|---|---|---|
| Commands | `track`, `track --arms`, the v0.2.0 fold | `gemini`, `openai`, `anthropic` |
| Model access | local `llama-server`, raw completion | hosted API, chat message |
| What is measured | probabilities of the letters A-D at the answer position, margins, shifts | the single letter the model returned (or that it returned none) |
| Derived values | log-odds margins, shifts, ranks, leaderboards | counts only: correct / competing / other incorrect / non-answer, newly induced incorrect answers, flips, pass agreement |

No hosted adapter ever produces a probability, margin, confidence or ranking: the Messages API has no logprobs parameter, Gemini rejected `responseLogprobs`, and the one OpenAI model that returned a top-20 list held all four
letters in only 17.5% of responses (those lists are not used). The Atlas reports the two types in separate sections and files and never merges them.

## 2. What counts as an answer: strict parsing, refusals, secondary diagnostics
**Strict parse (primary, always computed).** A reply is an answer only if, after trimming whitespace, it is exactly one of `A`, `B`, `C`, `D`. The saved `status` of every record is the strict result:
`ok`, `blocked`, `invalid`, `empty`, `malformed` or `error` (the request itself failed). Thinking blocks never count as text.

**Provider refusals are not incorrect answers.** A provider decision to return nothing (Anthropic `stop_reason: refusal` with `stop_details.category`; Gemini/OpenAI blocks) is status `blocked`, counted as a non-answer. It
is never added to "incorrect" or "competing", and a category label is provider metadata, not a finding about the prompt. `invalid` (prose, `**B**`, wrong letters) and `empty` (for example a reply that spent all of
`max_tokens` on thinking) are non-answers too.

**Secondary diagnostics are separate and labelled.** Lenient readings, for example rule FLR-v1 (Final-Line Recovery: the last non-empty line is one letter) or unwrapping `**B**`, are never mixed into the strict columns. They were defined after
replies were read, can credit replies the prompt forbade, and are reported in their own labelled columns (see below) and in analysis notes (`docs/EVIDENCE_RECONCILIATION.md`, `research/reconcile_evidence.py`).

**No-answer split and the FLR-v2 diagnostic (offline analysis and Atlas).** `answers.analyze` and the Atlas answer-only tables/CSV split the strict non-answers into *explicit refusals / blocks*, *format-invalid replies with one final letter* (`format_invalid_recoverable`) and *ambiguous or missing* (conflicting standalone letters, prose only, empty, malformed). For the format-invalid replies only, the optional diagnostic `answers.extract_final_letter` (FLR-v2) reports the letter the reply settled on (`flr_v2_correct`/`_competing`/`_other_incorrect`). FLR-v2 is FLR-v1 (historical, unchanged in `research/reconcile_evidence.py`) plus a conflict check: when standalone letters on different lines disagree, FLR-v1 takes the last line, FLR-v2 reports a conflict and picks nothing; on every other reply they agree. It is text-only and deterministic, never applies to refusals, rejects conflicting letters, never changes a stored or recomputed strict status or the strict columns, and produces no probability or confidence. On the saved pilots: Haiku RANDOM_OTHER 12 refusals; Sonnet 60 strict failures, of which COMPETING 32 (diagnostic: 2 correct / 30 competing).

## 3. Live runs and the budget
```
python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-haiku-5-5 --out runs/claude-haiku-5-5 --passes 2 \
    --budget 0.25 --price-in 0.10 --price-out 0.50 --effort low --max-output-tokens 1024
python -m bleediff openai    --prompts examples/openai-answer-only/prompts.json --model gpt-4.1-mini --out runs/gpt41mini --passes 2 --budget 1.00 --price-in 0.40 --price-out 1.60
python -m bleediff gemini    --prompts examples/gemini-answer-only/prompts.json --model gemini-3.8-flash --out runs/flash --passes 2 --budget 1.00 --price-in 0.75 --price-out 3.75 --thinking-level low
```
Keys come only from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`; they are never printed, never written to a record, and are redacted from every error message. Live runs need `--prompts`, `--model`, `--out`
(a **new** directory; evidence is never overwritten), `--budget` and both prices (there are no built-in prices). `--passes` repeats every prompt (1 to 20): repeats are replicate checks, not independent samples.

**Budget = a cap on an estimate, not on your bill.** The guard uses the prices you supply and the usage the provider reports. Before each request, the worst case of the next request (prompt at one token per UTF-8 byte
plus 64, at the input price, plus an output reserve at the output price) must fit under the remaining budget: the reserve is `max_tokens` (Anthropic, a hard cap on thinking plus text),
`max_completion_tokens` (OpenAI, includes reasoning) or `maxOutputTokens` plus a thinking allowance (Gemini), raised to twice the largest output seen so far. Afterwards the real usage is charged; missing or inconsistent usage is charged at the
worst-case bound; a response costing more than its admitted bound stops the run (`COST_BOUND_EXCEEDED`). Not modelled: cache discounts and surcharges, price tiers (for example Haiku 5.5 prompts over 100,000 tokens), later
price changes, taxes. **Retries are accounted conservatively:** a retry is sent only if one more full worst-case attempt still fits under the remaining estimate (otherwise the run stops with `BUDGET_EXHAUSTED`), and every failed attempt that may have been processed (a network error or an HTTP 5xx, including 529) is charged at the worst-case bound, whether or not a later attempt succeeds. Rate limits (429) and other 4xx responses are not charged. The record shows `billable_attempts` and the cost basis. This still bounds an estimate, **not a guaranteed billing limit**: check the provider's dashboard for what you were actually billed.
`--max-retries` (default 2, at most 5) bounds retries; quota/billing errors are never retried.

**Claude settings.** Claude 5.x thinks adaptively and `max_tokens` caps thinking plus text; a tiny cap yields an empty non-answer. Use `--max-output-tokens 1024` and, to limit thinking, `--effort low`. `--thinking`,
`--thinking-budget`, `--temperature`, `--top-p` are sent only if you pass them, and a value the model rejects is reported once, never retried.

## 4. Replay (offline, free)
```
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
python -m bleediff gemini    --replay examples/gemini-answer-only/flash
python -m bleediff openai    --replay examples/openai-answer-only/gpt41mini
```
`--replay` re-reads and re-validates a saved session (schemas, hashes, one record per prompt and pass) and recomputes every table: no key, no network, no model. It refuses to combine with live options. Add `--json-out FILE`
(new file) to keep the analysis. The Atlas takes sessions through an `answer_only_sessions` list (`bleediff atlas --manifest examples/hosted-comparison/atlas-input.json --out runs/hosted-comparison`).

## 5. What is saved, and redaction
Each session directory holds `session.json` (model, endpoint, generation config, prompt-file hash, budget rules, code revision, status), `prompts.json` and `records/<cell>__<condition>__pass<k>.json` (request
parameters without credentials, the strict status, the response, usage, estimated cost). Before a record is written, **opaque thinking signatures, redacted-thinking data and any returned reasoning text are replaced by
placeholders** (`[REDACTED:opaque-thinking-signature]`, `[REDACTED:private-reasoning]`) so evidence can be shared; answers, statuses and usage are untouched. Before sharing a session, also consider that visible reply text
is kept (a refusing or prose reply is evidence).

## 6. What these measurements cannot show
Confidence or probabilities; causes of a refusal; differences attributable to capability (providers differ in hidden reasoning, defaults and chat presentation); anything about prompts other than the frozen ones; independence
(the 16 cells are 4 questions x 4 mappings). See `docs/CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md` and `examples/hosted-comparison/README.md`.
