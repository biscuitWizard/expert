# llm-gateway — Design

This document describes how `llm-gateway` integrates with Ollama, runs the tool loop, routes feedback and domain actions, enforces guardrails, and handles summarization.

## 1. Ollama integration

The service uses an HTTP client against Ollama's OpenAI-compatible inference API:

- **Endpoint:** `/v1/chat/completions`.
- **Model selection:** The `model` field is included in every request, configured via `LLM_MODEL` env var (default `qwen3:32b`).
- **Streaming:** Server-Sent Events (SSE) for token or chunk streaming where supported.
- **Configuration:** Base URL, model name, model parameters (for example `temperature`, `top_p`, `max_tokens`).
- **Tools:** Tool or function definitions are supplied in the request body using the API's `tools` / `functions` parameter so the model can emit structured tool calls.

## 2. Tool call loop

After each LLM response in an invocation:

1. **Parse** tool calls from the response (provider-specific JSON or structured deltas, normalized internally).
2. **Route** each tool call:
   - **Feedback tools** (`suppress`, `recall`, `update_goal`, `set_threshold_hint`, `add_goal`) — handled by internal handlers; no external action stream unless specified below for goal routing.
   - **Domain tools** — publish to `actions.{stream_id}` and **wait** for the corresponding result as a new event (or equivalent completion signal) before continuing.
3. **Feed back** tool results to the model as follow-up message(s) in the chat sequence.
4. **Repeat** until the model returns a final assistant message with **no** tool calls.

Throughout this loop, the **server-side `ContextPackage` remains in memory** unchanged for dereferencing (for example `recall(event_indices)` against `recent_events`). It is not re-fetched from external stores mid-loop unless an explicit design adds that; the invariant is retention for the duration of the loop.

## 3. `recall()` data flow

When the model calls `recall(event_indices, reason)`:

1. **Dereference** `event_indices` against `context_package.recent_events` to obtain the corresponding `Event` records, including embeddings as stored in the package.
2. **Build** a `TrainingExample` that includes the event window, embeddings, goal embedding, and **positive** label semantics as defined by the product spec.
3. **Publish** that example to `labels.write`.

The **rendered prompt** visible to the model contains only natural language (spec invariant 4: no raw structured event payloads in the user-facing prompt text). Full typed event data lives **only** on the server-side `ContextPackage` used for dereferencing and label construction.

## 4. `suppress()` data flow

When the model calls `suppress` (parameters per API contract):

1. **Construct** a `TrainingExample` with **negative** label semantics.
2. **Publish** to `labels.write`.
3. **Increment** the per-invocation (or session-scoped, per policy) suppress counter for guardrail accounting.

## 5. Goal and threshold routing

- **`update_goal()` and `add_goal()`** — publish a `GoalUpdateRequest` on `requests.goal_update`.
- **`set_threshold_hint()`** — publish a `ThresholdHint` on the **same** stream `requests.goal_update`, distinguished by **message type** (not by topic).

Downstream consumers must discriminate by message type when handling goal updates versus threshold hints.

## 6. Alignment guardrails (spec Section 7.3)

Guardrails apply **per invocation** within the tool loop:

- **Label write cap:** A maximum number of label writes (`labels.write`) per invocation; attempts beyond `N` are rejected or no-oped per policy.
- **Suppress rate:** If the suppress rate exceeds `MAX_SUPPRESS_RATE`, emitted labels are **quarantined** (or tagged for review) instead of being treated as normal training signal, per implementation.
- **Recall rate:** If recall volume is excessive relative to policy, flag as a **systematic SSM blind spot** (or equivalent telemetry) for operator or automated follow-up.

These checks run as tool results are processed, before or after publish depending on whether the design chooses fail-closed vs. publish-with-flag.

## 7. Post-invocation

After the tool loop completes successfully:

1. **Episode** — Construct an `Episode` summarizing the invocation (metadata, outcomes, pointers as required) and publish to `episodes.write`.
2. **Raw exchange** — Publish the full raw `Exchange` to `events.exchange.{activity_id}` for downstream consumers (for example `rag-service` for session history indexing and retrieval).

Ordering between episode and exchange publishes should be consistent (document in code: typically episode then exchange, or vice versa, but not ambiguous for consumers).

## 8. Summarization

A separate path from interactive tool-loop invocations:

1. **Consume** `requests.summarize` (payload: activity/session identifiers and raw or referenced exchange text, per schema).
2. **Call** Ollama with a dedicated **summarization system prompt** and the raw exchange text (or concatenated transcript).
3. **Publish** the compressed narrative to `results.summarize`.

This use of the LLM is **lower priority** than interactive completion (scheduling, queue depth, or rate limits may deprioritize summarization under load).
