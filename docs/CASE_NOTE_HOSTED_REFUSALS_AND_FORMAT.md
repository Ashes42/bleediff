# Case note: two ways a hosted model can fail to give a letter

Status: exploratory, answer-level observations from [522 unique raw requests](EVIDENCE_RECONCILIATION.md) on `claude-haiku-5-5` and `claude-sonnet-5-5` (effort low, `max_tokens` 1024, API-default thinking and temperature).
These are observations about two models' behaviour on one frozen prompt family. They are **not** a jailbreak, an attack technique or a security vulnerability, and nothing here shows what the provider's
safeguards are designed to catch. A refusal is a provider decision to return no answer; it is **not an incorrect answer** and says nothing about which letter the model preferred.

## 1. Haiku: refusals on RANDOM_OTHER
In the 80-prompt pilot (4 questions x 4 answer-letter mappings x 5 conditions x 2 passes), Haiku answered in clean letters everywhere except one condition: **12 of 32 RANDOM_OTHER requests ended in
`stop_reason: refusal`** (category `bio`, 0 output tokens). RANDOM_OTHER inserts one fixed 39-character string of uppercase consonant groups between the evidence and the answer instruction. Sonnet, given
the identical prompts, refused none (0 of 160). Across all the runs Haiku produced 82 refusals, every one with the same category label, which is the provider's own tag; the evidence does not show what in the prompt it reacts to.
Follow-ups showed the behaviour depends on the exact string and the surrounding question (v0.12 to v0.13):
- the original string and permutations of its letters refused in most question contexts (F0 16/20, a letter-permutation 15/20); a different uppercase consonant string never did (0/20, 0/36 across four alternatives);
- lowercasing the string reduced refusals (8/20) without removing them; one question (q4) never refused under any string.
- of the four same-letter strings in v0.13 (F0 and three shuffles of it), three refused and one (S1) answered in every tested context.

## 2. Sonnet: strict-format failures
Sonnet 5.5 refused nothing but broke "reply with exactly one letter" in **60 of 160 requests** (all 32 COMPETING, 16 RANDOM_OTHER, 12 RANDOM_AB): visible prose that ends in a letter. Under the strict parse these are
`invalid` non-answers, so Sonnet's strict COMPETING row has no answers. The separately labelled secondary rule S1 (last line is one letter; see the reconciliation note) puts Sonnet at 2 correct / 30 competing, the same totals as
GPT-4.1 mini, and 32/32 correct in the other four conditions. S1 was defined after reading the replies and credits a reply the prompt forbade; the strict table remains the primary result.

## 3. The single-swap q2 contrast (v0.14)
In question q2-m1, the unchanged string S1 (`ZVXX RXWK ...`) answered A in 7 of 7 raw requests, while S1g1, S1 with two adjacent letters exchanged (`ZXVX RXWK ...`), refused in 4 of 4 (one stage-1 draw plus three confirmation
repeats chosen after seeing it). Both prompts have 281 input tokens. Of the 16 single adjacent swaps tried, only this one changed an outcome (all 8 swaps of the refusing F0 still refused; 1 of 8 swaps of S1 refused).

## 4. The contrast did not transfer to q1 (v0.15)
Using the frozen q1-m1 prompt with only the filler replaced, S1 answered A and S1g1 answered A, two requests each (ABBA order), 263 tokens for both. The refusal in q2-m1 therefore did not reproduce in the one other question context tried.
q1-m1 is not immune to the string family (F0, S2 and S3 refused there), so the S1g1 result looks specific to the pairing of that string with that question, at least among the contexts tested.

## 5. What remains unknown
- What in a prompt triggers a refusal, or why the label is `bio`; the label is provider metadata, not a measured cause.
- Whether the sensitivity is about tokenization, the character sequence, position, or context: token counts do not separate the groups (S1 and S1g1 are both 281 in q2); token boundaries were never inspected.
- Prevalence: the sample is a few hand-picked prompts, identical repeats are not independent, and two requests per filler cannot exclude a low refusal rate in q1-m1.
- Stability over time: refusal behaviour may change with model updates; all runs were made within one day on one model snapshot per tier.
- Whether Sonnet's prose is a defensible reaction to conflicting evidence rather than a failure; the prompt asked for one letter, and the strict table measures exactly that.
- Anything about other effort levels, system prompts, other models or other fillers.
