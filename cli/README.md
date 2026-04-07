# expert-cli

Command-line interface for the Expert attention fabric system. Thin Python client that calls the orchestrator's REST API.

## Installation

```bash
cd cli
pip install -e .
```

## Usage

```bash
expert status                        # system health, worker count, queue depth
expert activity list                 # list all activities with lifecycle state
expert activity create \
  --stream-id mud-01 \
  --domain mud \
  --goal "Alert on combat threats"   # create activity with a root goal
expert activity inspect <id>         # detailed state, scores, firing history
expert activity suspend <id>         # serialize state, remove from worker
expert activity resume <id>          # restore to a worker
expert activity terminate <id>       # discard state, preserve goals in RAG
expert goal add <activity-id> \
  --parent <goal-id> \
  --description "Health drops below 30%" \
  --aggregation max                  # add a child goal to an activity
expert goal list <activity-id>       # show goal tree for an activity
```

## Configuration

The CLI reads the orchestrator URL from the `EXPERT_ORCHESTRATOR_URL` environment variable (default: `http://localhost:8081`).

## Dependencies

- Python >= 3.11
- [typer](https://typer.tiangolo.com/) -- CLI framework
- [httpx](https://www.python-httpx.org/) -- async HTTP client
- [rich](https://rich.readthedocs.io/) -- terminal formatting

## Future

Natural path to an LLM management agent layer: an LLM with tool definitions mapping to the orchestrator API, translating natural language commands to API calls. The orchestrator REST API is the real interface; this CLI and any future agent are clients of it.
