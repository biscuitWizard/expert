# discord-adapter

## Responsibility

The **discord-adapter** service is a bidirectional adapter between Discord (connected as a user account) and the internal event bus. It normalizes inbound Discord events (messages, DMs, friend requests, guild joins) into the canonical [`Event`](../../crates/expert-types/src/event.rs) format and executes outbound domain actions from the LLM tool loop.

## Ownership

This service owns Discord authentication, Gateway connection management, event normalization, and outbound action execution for the Discord domain.

## Integration

**Consumes from:** Redis Streams on `actions.{stream_id}` — outbound domain tool calls emitted by **llm-gateway**.

**Publishes to:** Redis Streams on `events.raw.{stream_id}` — normalized canonical events for downstream processing.

**Bootstraps:** On startup, creates an activity via the **orchestrator** REST API with Discord-specific tool definitions and default goals.

**Depends on:** Redis for stream buffers, Orchestrator for activity lifecycle, Discord Gateway + REST API for external communication.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | Yes (or email/pass) | Discord user token |
| `DISCORD_EMAIL` | No | Email for login fallback |
| `DISCORD_PASSWORD` | No | Password for login fallback |
| `DISCORD_USERNAME` | Yes | Bot's own display name |
| `STREAM_ID` | No | Stream ID override (default: `discord`) |
| `REDIS_URL` | No | Redis connection URL |
| `ORCHESTRATOR_URL` | No | Orchestrator REST API URL |
| `DISCORD_GUILD_IDS` | No | Comma-separated guild IDs to filter (empty = all) |

## Running

```bash
# Via cargo (local dev)
DISCORD_TOKEN=<token> DISCORD_USERNAME=<name> cargo run -p discord-adapter

# Via docker compose
docker compose up discord-adapter
```
