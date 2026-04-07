import json
import os
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="expert", help="Expert attention fabric CLI")
console = Console()

BASE_URL = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:3000")


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


@app.command("create-activity")
def create_activity(
    stream_id: str = typer.Option(..., "--stream-id", "-s", help="Stream ID to monitor"),
    goal: list[str] = typer.Option(..., "--goal", "-g", help="Goal description (repeatable)"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain name"),
):
    """Create a new activity with goals."""
    goals = [{"name": g.split(":")[0].strip() if ":" in g else g[:30], "description": g} for g in goal]

    body = {
        "stream_id": stream_id,
        "goals": goals,
    }
    if domain:
        body["domain"] = domain

    with _client() as client:
        resp = client.post("/activities", json=body)

    if resp.status_code == 201:
        data = resp.json()
        console.print(f"[green]Activity created:[/green] {data['activity_id']}")
        console.print(f"  Stream: {data['stream_id']}")
        console.print(f"  Domain: {data['domain']}")
        console.print(f"  Goals:  {data['goal_count']}")
    else:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")


@app.command("list")
def list_activities():
    """List all activities."""
    with _client() as client:
        resp = client.get("/activities")

    if resp.status_code != 200:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")
        return

    activities = resp.json()
    if not activities:
        console.print("[dim]No activities found.[/dim]")
        return

    table = Table(title="Activities")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Stream", style="green")
    table.add_column("Domain")
    table.add_column("State", style="yellow")
    table.add_column("Goals", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Invocations", justify="right")

    for a in activities:
        table.add_row(
            a["activity_id"][:12],
            a["stream_id"],
            a["domain"],
            a["lifecycle_state"],
            str(a["goal_count"]),
            str(a["event_count"]),
            str(a["invocation_count"]),
        )

    console.print(table)


@app.command("status")
def status(activity_id: str = typer.Argument(..., help="Activity ID")):
    """Show detailed activity status."""
    with _client() as client:
        resp = client.get(f"/activities/{activity_id}")

    if resp.status_code == 404:
        console.print(f"[red]Activity not found:[/red] {activity_id}")
        return
    if resp.status_code != 200:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")
        return

    data = resp.json()
    console.print(f"[bold]Activity:[/bold] {data['activity_id']}")
    console.print(f"  Stream:      {data['stream_id']}")
    console.print(f"  Domain:      {data['domain']}")
    console.print(f"  State:       [yellow]{data['lifecycle_state']}[/yellow]")
    console.print(f"  Events:      {data['event_count']}")
    console.print(f"  Invocations: {data['invocation_count']}")
    console.print(f"  Suppresses:  {data['suppress_count']}")
    console.print(f"  Recalls:     {data['recall_count']}")
    console.print(f"  Thresholds:  {data['theta']}")

    console.print("\n[bold]Goals:[/bold]")
    for g in data.get("goals", []):
        console.print(f"  - {g['name']} (v{g['version']}): {g['description']}")


@app.command("delete")
def delete_activity(activity_id: str = typer.Argument(..., help="Activity ID")):
    """Delete an activity."""
    with _client() as client:
        resp = client.delete(f"/activities/{activity_id}")

    if resp.status_code == 204:
        console.print(f"[green]Activity deleted:[/green] {activity_id}")
    elif resp.status_code == 404:
        console.print(f"[red]Activity not found:[/red] {activity_id}")
    else:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")


@app.command("health")
def health():
    """Check orchestrator health."""
    with _client() as client:
        try:
            resp = client.get("/health")
            if resp.status_code == 200:
                console.print(f"[green]Orchestrator:[/green] healthy ({BASE_URL})")
            else:
                console.print(f"[red]Orchestrator:[/red] unhealthy (status {resp.status_code})")
        except httpx.ConnectError:
            console.print(f"[red]Orchestrator:[/red] unreachable ({BASE_URL})")


if __name__ == "__main__":
    app()
