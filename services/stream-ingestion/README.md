# stream-ingestion

## Responsibility

The **stream-ingestion** service is the bidirectional adapter layer between external event sources and the internal event bus. It normalizes inbound raw data into the canonical [`Event`](../../crates/expert-types/src/event.rs) format used across Expert. It also executes outbound domain actions: sending telnet commands, placing trades, querying APIs, or any other tool-specific interaction with the external system.

## Ownership

This service owns stream registration, adapter lifecycle management, event normalization, and outbound action execution.

## Integration

**Consumes from:** Redis Streams on `actions.{stream_id}` — outbound domain tool calls emitted by **llm-gateway** (LLM tool invocations that must be applied to the live source).

**Publishes to:** Redis Streams on `events.raw.{stream_id}` — normalized canonical events for downstream processing (encoding, routing, agent consumption).

**Depends on:** Redis for stream buffers, consumer groups, sequence counters, and coordination.

## State model

The service is **stateless** at the process level: durable ordering is delegated to Redis. Per-stream sequence counters are persisted in Redis. Connection and protocol state exist only within each running adapter instance (not replicated across replicas of this service).

## Specification reference

Canonical event shape and fields are defined in **section 9.3 (Event schema)** of the Expert specification. Implementation details and adapter contracts are documented in [DESIGN.md](./DESIGN.md).
