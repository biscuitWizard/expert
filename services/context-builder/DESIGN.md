# Context Builder: Design

This document specifies how the context builder assembles LLM invocation context: stream lookback, RAG integration, prompt template, truncation policy, and invariants for what the model is allowed to see.

## 1. Assembly pipeline

On receiving an **AssembleRequest** from `requests.context`, the service executes the following steps in order.

1. **Stream events (embedded)**: Read recent events from `events.embedded.{stream_id}` using `XREVRANGE`, bounded between `last_fired_seq` and `trigger_event_seq` (inclusive semantics as defined by the stream contract). These events populate **RECENT STREAM** in the rendered prompt and are the events that `recall()` indices refer to. The **ContextPackage** retains full **Event** objects (including embeddings) for **llm-gateway**; the rendered prompt does not expose raw vectors or internal IDs (see section 6).

2. **Top-K episodes**: Issue an asynchronous RAG query for the top-K episodes similar to the trigger event and goal embedding (exact embedding fields follow the **AssembleRequest** and RAG API contract).

3. **Compressed session history**: Retrieve the compressed LLM session history from **rag-service** for **SESSION HISTORY** in the template.

4. **Last M raw exchanges**: Retrieve the last M raw chat exchanges from **rag-service** for **PREVIOUS CHAT ENTRIES**.

5. **Render prompt**: Fill the context package template (section 2) with natural-language renderings only.

6. **Attach ContextPackage**: Include the server-side **ContextPackage** carrying full event payloads (with embeddings) and any other gateway-required fields so **llm-gateway** can execute tools and correlation without re-fetching stream state.

7. **Publish**: Publish the completed package to `packages.ready`.

Failures at any step follow the project's error-handling conventions (logging, dead-letter or retry policy, and whether partial packages are ever emitted); this document does not prescribe operational defaults beyond avoiding unbounded blocking on RAG (section 4).

## 2. Context package template (specification section 6.2)

The LLM-visible prompt body follows this structure. Placeholders are filled after retrieval and event read; scalar formatting (for example `score` as a percentage) is applied at render time.

```
=== ACTIVITY CONTEXT ===
Activity: {activity_name}
Stream: {stream_description}

--- CURRENT GOAL ---
{goal_tree_natural_language}

--- WHAT TRIGGERED THIS ---
The following event crossed the attention threshold (confidence: {score:.0%}):
{trigger_event_natural_language}

--- RECENT STREAM (last {N} events, most recent last) ---
{recent_events_as_prose_or_list}

--- RELEVANT PAST CONTEXT ---
{top_k_retrieved_episodes_as_prose_summaries}

--- SESSION HISTORY ---
{compressed_llm_history}

--- PREVIOUS CHAT ENTRIES ---
{n_most_recent_raw_exchanges}

=== END CONTEXT ===
```

**N**, **K**, and **M** are the counts after any truncation (section 3). The template is normative for section ordering and headings; minor wording adjustments for clarity are allowed only if they remain semantically equivalent and spec-compliant.

## 3. Truncation priority (specification section 6.3)

If the assembled context would exceed the configured LLM context limit, reduce material in the following order. Items listed first are removed or reduced first (the last items in the list are the last to be touched).

1. **Previous chat entries**: Decrease M (drop older exchanges first within the "last M" window).
2. **Relevant past context**: Decrease K (drop lower-ranked retrieved episodes first).
3. **Recent stream events**: Decrease N (drop older events in the recent window first while preserving ordering: most recent last).
4. **Session history**: Apply further compression or shorten the compressed history per **rag-service** capabilities and policy.
5. **Goal description**: Do not truncate.
6. **Trigger event**: Do not truncate.

Goal and trigger sections are mandatory in full unless the specification explicitly allows an error path when even minimal renderings do not fit; the implementation should prefer failing closed with a clear error over silently dropping goal or trigger content.

## 4. RAG query pattern

Interaction with **rag-service** uses Redis Streams for request and response:

- Publish the query to `queries.rag` with a unique **request_id** (or equivalent correlation identifier) and all parameters required by **rag-service**.
- Consume `results.rag`, selecting the message whose **request_id** matches the in-flight request.
- Enforce a **timeout** on each wait so assembly does not block indefinitely if **rag-service** is slow or drops a reply.

Multiple concurrent assemble operations must correlate strictly by **request_id** to avoid cross-talk. Retries, idempotency, and partial-result behavior are implementation-defined but must preserve the timeout invariant.

## 5. Event lookback via XREVRANGE

The embedded event stream name is `events.embedded.{stream_id}`. The lookback window runs backward from `trigger_event_seq` toward `last_fired_seq`, inclusive according to the product definition of those sequence bounds.

Events in this window:

- Appear in **RECENT STREAM** as prose or a list, subject to truncation (section 3).
- Define the index space for `recall()` and related attention tools that refer to positions in the recent stream.

The **ContextPackage** always retains complete **Event** records (including embeddings) for downstream use by **llm-gateway** even when the rendered prompt shortens or omits prose for budget reasons.

## 6. Natural language rendering (specification invariants 4 and 11)

The rendered prompt is **natural language only** with respect to sensitive machine artifacts:

- **Never** include embedding vectors, raw similarity scores (except where the template explicitly formats a confidence, such as `{score:.0%}` for the trigger), or opaque internal identifiers in the prose sections unless the specification explicitly calls for a human-safe label.
- **Never** expose retrieval keys, episode IDs, or database identifiers to the model in the prompt body.
- Goal hierarchy is rendered as indented natural language. Retrieved episodes appear as prose summaries produced from retrieval results, not as raw index rows.

The server-side **ContextPackage** may still carry structured IDs and embeddings for gateway and tooling; that data is not copied verbatim into the LLM prompt.
