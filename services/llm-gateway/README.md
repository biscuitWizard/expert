# llm-gateway

## Responsibility

The `llm-gateway` service manages LLM invocations through llamacpp’s HTTP API. It consumes context packages, submits prompts with per-activity tool sets, handles streaming responses, and parses tool calls. It retains a server-side `ContextPackage` for the duration of the tool loop so `recall()` can dereference event indices. The service routes domain tool calls, feedback, and goal or threshold updates; enforces alignment guardrails; publishes raw exchanges after each invocation; and handles session history summarization requests.

## Ownership

- llamacpp HTTP client
- Tool call parsing
- Tool execution routing
- Alignment guardrails
- Server-side `ContextPackage` retention during the tool loop

## Messaging

| Direction | Topic |
|-----------|--------|
| Consumes | `packages.ready`, `requests.summarize` |
| Publishes | `actions.{stream_id}`, `labels.write`, `episodes.write`, `events.exchange.{activity_id}`, `requests.goal_update`, `results.summarize` |

## Dependencies

- **llamacpp** — remote inference over HTTP (completion or OpenAI-compatible chat completions).

## State model

**Stateless per invocation.** Each invocation is independent; the `ContextPackage` held during a tool loop is ephemeral and scoped to that loop only.
