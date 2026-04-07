# Context Builder

Assembly service for the Expert attention fabric. The context builder consumes assemble requests from the orchestrator, gathers recent stream events, retrieval-augmented context, and session history, then produces a structured natural-language prompt plus a server-side context package for **llm-gateway**.

## Responsibility

- **Assemble context packages**: Build the full invocation context for each LLM call from stream events, RAG results, and session state.
- **Consume assemble requests**: Read **AssembleRequest** messages from the orchestrator on `requests.context`.
- **Event lookback**: Read recent events via Redis `XREVRANGE` on per-stream embedded event streams for the window relevant to the invocation.
- **Retrieval**: Query **rag-service** for top episodes and patterns aligned with the trigger event and goal embedding.
- **Session history**: Retrieve compressed history and recent raw exchanges from **rag-service** as defined by the assembly contract.
- **Prompt rendering**: Render a structured natural-language prompt from assembled facts; apply truncation priority when the result exceeds the configured LLM context budget.
- **Publish completed packages**: Emit finished **ContextPackage** payloads (including full event data for the gateway) on `packages.ready`.

## Owns

- Prompt template (layout and section ordering per specification)
- Truncation logic and context budget enforcement
- Natural-language rendering rules (what appears in the LLM-visible prompt versus server-side-only data)

## Consumes

| Source | Role |
|--------|------|
| `requests.context` | Assemble requests from the orchestrator (trigger, goal, stream, sequence bounds, tool set references, etc.) |

## Publishes

| Destination | Role |
|-------------|------|
| `packages.ready` | Completed context packages ready for **llm-gateway** (structured prompt plus server-side **ContextPackage** with full events and embeddings) |

## Dependencies

- **Redis**: Stream reads via `XREVRANGE` for event lookback on `events.embedded.{stream_id}`.
- **rag-service**: Episodes, patterns, compressed session history, and raw exchanges via asynchronous query/result streams (`queries.rag` / `results.rag`).

## Stateful

No. Assembly is a pure function of the request, Redis stream slices, and RAG responses; no durable service-owned state is required beyond process-local caches if any are introduced for performance.

## Related documentation

See [DESIGN.md](./DESIGN.md) for the assembly pipeline, template, truncation order, RAG request pattern, and rendering invariants.
