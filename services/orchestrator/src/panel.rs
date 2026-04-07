use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Form, Json, Router, extract::Request, middleware};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScrollPointsBuilder;
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tracing::warn;

use crate::{AppState, ModelWarmupStatus};

static PANEL_HTML: &str = include_str!("panel.html");

pub fn router(state: Arc<AppState>, qdrant: Option<Qdrant>) -> Router {
    let session_token = uuid::Uuid::new_v4().to_string();

    let panel_state = Arc::new(PanelState {
        app: state.clone(),
        qdrant,
        session_token,
    });

    let api_routes = Router::new()
        .route("/panel/api/status", get(system_status))
        .route("/panel/api/activities", get(list_activities))
        .route("/panel/api/activities/{id}", get(get_activity))
        .route("/panel/api/qdrant/collections", get(list_collections))
        .route("/panel/api/qdrant/collections/{name}", get(collection_info))
        .route(
            "/panel/api/qdrant/collections/{name}/scroll",
            get(scroll_points),
        )
        .route_layer(middleware::from_fn_with_state(
            panel_state.clone(),
            token_auth_middleware,
        ))
        .with_state(panel_state.clone());

    Router::new()
        .route("/panel/auth", post(handle_login))
        .route("/panel", get(serve_panel))
        .route("/panel/sse", get(sse_handler))
        .with_state(panel_state)
        .merge(api_routes)
}

struct PanelState {
    app: Arc<AppState>,
    qdrant: Option<Qdrant>,
    session_token: String,
}

fn check_token(state: &PanelState, headers: &HeaderMap, uri: &axum::http::Uri) -> bool {
    if let Some(token_header) = headers.get("x-panel-token") {
        if let Ok(token) = token_header.to_str() {
            if token == state.session_token {
                return true;
            }
        }
    }
    if let Some(query) = uri.query() {
        for param in query.split('&') {
            if let Some(val) = param.strip_prefix("token=") {
                if val == state.session_token {
                    return true;
                }
            }
        }
    }
    false
}

async fn token_auth_middleware(
    State(state): State<Arc<PanelState>>,
    headers: HeaderMap,
    request: Request,
    next: middleware::Next,
) -> impl IntoResponse {
    if check_token(&state, &headers, request.uri()) {
        return next.run(request).await.into_response();
    }
    StatusCode::UNAUTHORIZED.into_response()
}

#[derive(Deserialize)]
struct LoginForm {
    username: String,
    password: String,
}

async fn handle_login(
    State(state): State<Arc<PanelState>>,
    Form(form): Form<LoginForm>,
) -> impl IntoResponse {
    let (Some(expected_user), Some(expected_pass)) =
        (&state.app.config.panel_user, &state.app.config.panel_pass)
    else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };

    if form.username == *expected_user && form.password == *expected_pass {
        (
            StatusCode::FOUND,
            [(
                header::LOCATION,
                format!("/panel?token={}", state.session_token),
            )],
        )
            .into_response()
    } else {
        Html(login_page(Some("Invalid credentials"))).into_response()
    }
}

fn login_page(error: Option<&str>) -> String {
    let error_html = error
        .map(|e| {
            format!("<div style=\"color:#f85149;margin-bottom:16px;font-size:13px\">{e}</div>")
        })
        .unwrap_or_default();
    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Expert Debug Panel - Login</title>
<style>
body {{ background:#0e1117; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }}
.login {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:32px; width:320px; }}
h2 {{ font-size:18px; margin-bottom:20px; font-weight:600; }}
label {{ display:block; font-size:12px; color:#8b949e; margin-bottom:4px; }}
input {{ width:100%; padding:8px 10px; background:#0e1117; border:1px solid #30363d; border-radius:4px; color:#e6edf3; font-size:13px; margin-bottom:14px; box-sizing:border-box; }}
input:focus {{ border-color:#58a6ff; outline:none; }}
button {{ width:100%; padding:8px; background:#238636; border:1px solid #2ea043; border-radius:4px; color:#fff; font-size:13px; font-weight:600; cursor:pointer; }}
button:hover {{ background:#2ea043; }}
</style></head><body>
<div class="login">
<h2>Expert Debug Panel</h2>
{error_html}
<form method="POST" action="/panel/auth">
<label for="username">Username</label>
<input type="text" id="username" name="username" autofocus>
<label for="password">Password</label>
<input type="password" id="password" name="password">
<button type="submit">Sign in</button>
</form>
</div></body></html>"#
    )
}

async fn serve_panel(
    State(state): State<Arc<PanelState>>,
    headers: HeaderMap,
    request: Request,
) -> impl IntoResponse {
    if !check_token(&state, &headers, request.uri()) {
        return Html(login_page(None)).into_response();
    }

    let html = PANEL_HTML.replace(
        "<!--SESSION_TOKEN-->",
        &format!(
            "<meta name=\"panel-token\" content=\"{}\">",
            state.session_token
        ),
    );
    Html(html).into_response()
}

async fn sse_handler(
    State(state): State<Arc<PanelState>>,
    headers: HeaderMap,
    request: Request,
) -> impl IntoResponse {
    if !check_token(&state, &headers, request.uri()) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let rx = state.app.event_log.subscribe();

    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_default();
            let sse_event: Result<Event, std::convert::Infallible> =
                Ok(Event::default().data(data));
            Some(sse_event)
        }
        Err(_) => None,
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

async fn list_activities(State(state): State<Arc<PanelState>>) -> impl IntoResponse {
    let registry = state.app.registry.read().await;
    let activities: Vec<serde_json::Value> = registry
        .list()
        .iter()
        .map(|m| {
            serde_json::json!({
                "activity_id": m.state.activity_id,
                "stream_id": m.state.stream_id,
                "domain": m.state.domain,
                "lifecycle_state": format!("{:?}", m.state.lifecycle_state),
                "goals": m.goals.iter().map(|g| serde_json::json!({
                    "id": g.id,
                    "name": g.name,
                    "description": g.description,
                    "version": g.version,
                    "active": g.active,
                })).collect::<Vec<_>>(),
                "theta": m.state.theta,
                "event_count": m.state.event_count,
                "invocation_count": m.state.invocation_count,
                "suppress_count": m.state.suppress_count,
                "recall_count": m.state.recall_count,
                "created_at": m.state.created_at,
                "last_active": m.state.last_active,
                "worker_id": m.worker_id,
                "has_pending_fire": m.pending_fire.is_some(),
            })
        })
        .collect();
    Json(activities)
}

async fn get_activity(
    State(state): State<Arc<PanelState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let registry = state.app.registry.read().await;
    match registry.get(&id) {
        Some(m) => {
            let resp = serde_json::json!({
                "activity_id": m.state.activity_id,
                "stream_id": m.state.stream_id,
                "domain": m.state.domain,
                "lifecycle_state": format!("{:?}", m.state.lifecycle_state),
                "goals": m.goals.iter().map(|g| serde_json::json!({
                    "id": g.id,
                    "name": g.name,
                    "description": g.description,
                    "version": g.version,
                    "active": g.active,
                    "parent_id": g.parent_id,
                    "aggregation": format!("{:?}", g.aggregation),
                    "domain": g.domain,
                })).collect::<Vec<_>>(),
                "theta": m.state.theta,
                "event_count": m.state.event_count,
                "invocation_count": m.state.invocation_count,
                "suppress_count": m.state.suppress_count,
                "recall_count": m.state.recall_count,
                "created_at": m.state.created_at,
                "last_active": m.state.last_active,
                "worker_id": m.worker_id,
                "has_pending_fire": m.pending_fire.is_some(),
                "firing_history": m.state.firing_history,
                "refractory_until": m.state.refractory_until,
                "h_norm": m.state.h.iter().map(|x| x * x).sum::<f32>().sqrt(),
            });
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn list_collections(State(state): State<Arc<PanelState>>) -> impl IntoResponse {
    let Some(ref qdrant) = state.qdrant else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Qdrant not configured"})),
        )
            .into_response();
    };

    match qdrant.list_collections().await {
        Ok(response) => {
            let names: Vec<String> = response
                .collections
                .iter()
                .map(|c| c.name.clone())
                .collect();
            Json(serde_json::json!({ "collections": names })).into_response()
        }
        Err(e) => {
            warn!(error = %e, "qdrant list_collections failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("{e}")})),
            )
                .into_response()
        }
    }
}

async fn collection_info(
    State(state): State<Arc<PanelState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let Some(ref qdrant) = state.qdrant else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Qdrant not configured"})),
        )
            .into_response();
    };

    match qdrant.collection_info(&name).await {
        Ok(info) => {
            let result = &info.result;
            let resp = serde_json::json!({
                "name": name,
                "status": format!("{:?}", result.as_ref().map(|r| r.status)),
                "points_count": result.as_ref().and_then(|r| r.points_count),
                "indexed_vectors_count": result.as_ref().and_then(|r| r.indexed_vectors_count),
                "segments_count": result.as_ref().map(|r| r.segments_count),
            });
            Json(resp).into_response()
        }
        Err(e) => {
            warn!(error = %e, collection = %name, "qdrant collection_info failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("{e}")})),
            )
                .into_response()
        }
    }
}

#[derive(Deserialize)]
struct ScrollParams {
    offset: Option<String>,
    limit: Option<u32>,
}

async fn scroll_points(
    State(state): State<Arc<PanelState>>,
    Path(name): Path<String>,
    Query(params): Query<ScrollParams>,
) -> impl IntoResponse {
    let Some(ref qdrant) = state.qdrant else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Qdrant not configured"})),
        )
            .into_response();
    };

    let limit = params.limit.unwrap_or(20).min(100);

    let mut builder = ScrollPointsBuilder::new(&name)
        .limit(limit)
        .with_payload(true);

    if let Some(ref offset_id) = params.offset {
        use qdrant_client::qdrant::PointId;
        use qdrant_client::qdrant::point_id::PointIdOptions;
        builder = builder.offset(PointId {
            point_id_options: Some(PointIdOptions::Uuid(offset_id.clone())),
        });
    }

    match qdrant.scroll(builder).await {
        Ok(response) => {
            let points: Vec<serde_json::Value> = response
                .result
                .iter()
                .map(|p| {
                    let id = p.id.as_ref().map(|pid| match &pid.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u.clone(),
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => {
                            n.to_string()
                        }
                        None => String::new(),
                    });

                    let payload: serde_json::Map<String, serde_json::Value> = p
                        .payload
                        .iter()
                        .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                        .collect();

                    serde_json::json!({
                        "id": id,
                        "payload": payload,
                    })
                })
                .collect();

            let next_offset =
                response
                    .next_page_offset
                    .as_ref()
                    .map(|pid| match &pid.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u.clone(),
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => {
                            n.to_string()
                        }
                        None => String::new(),
                    });

            Json(serde_json::json!({
                "points": points,
                "next_offset": next_offset,
            }))
            .into_response()
        }
        Err(e) => {
            warn!(error = %e, collection = %name, "qdrant scroll failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("{e}")})),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// System status endpoint
// ---------------------------------------------------------------------------

const CHECK_TIMEOUT: Duration = Duration::from_millis(2000);

async fn system_status(State(state): State<Arc<PanelState>>) -> impl IntoResponse {
    let config = &state.app.config;

    let (redis, qdrant, postgres, ollama_ps, system, gpus) = tokio::join!(
        check_redis(&config.redis_url),
        check_qdrant(&state.qdrant),
        check_tcp_from_url(&config.postgres_url),
        check_ollama_ps(&config.ollama_url),
        read_proc_metrics(),
        read_gpu_metrics(),
    );

    let warmup = state.app.warmup_status.read().await;
    let ollama = build_ollama_status(ollama_ps, &warmup, config);

    Json(serde_json::json!({
        "services": {
            "redis": redis,
            "qdrant": qdrant,
            "postgres": postgres,
            "ollama": ollama,
        },
        "system": system,
        "gpus": gpus,
    }))
}

async fn check_redis(url: &str) -> serde_json::Value {
    let start = Instant::now();
    let host_port = url
        .strip_prefix("redis://")
        .unwrap_or(url)
        .trim_end_matches('/');
    let addr = if !host_port.contains(':') {
        format!("{host_port}:6379")
    } else {
        host_port.to_string()
    };

    match tokio::time::timeout(CHECK_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(_stream)) => serde_json::json!({
            "status": "up",
            "latency_ms": start.elapsed().as_millis(),
        }),
        Ok(Err(e)) => serde_json::json!({
            "status": "down",
            "error": format!("{e}"),
        }),
        Err(_) => serde_json::json!({
            "status": "down",
            "error": "timeout",
        }),
    }
}

async fn check_qdrant(client: &Option<Qdrant>) -> serde_json::Value {
    let Some(client) = client else {
        return serde_json::json!({"status": "unconfigured"});
    };
    let start = Instant::now();
    match tokio::time::timeout(CHECK_TIMEOUT, client.health_check()).await {
        Ok(Ok(_)) => serde_json::json!({
            "status": "up",
            "latency_ms": start.elapsed().as_millis(),
        }),
        Ok(Err(e)) => serde_json::json!({
            "status": "down",
            "error": format!("{e}"),
        }),
        Err(_) => serde_json::json!({
            "status": "down",
            "error": "timeout",
        }),
    }
}

fn parse_host_port(url: &str) -> Option<String> {
    // Handles postgres://user:pass@host:port/db and http://host:port/path
    let without_scheme = url.find("://").map(|i| &url[i + 3..]).unwrap_or(url);
    // Strip userinfo
    let after_auth = without_scheme
        .find('@')
        .map(|i| &without_scheme[i + 1..])
        .unwrap_or(without_scheme);
    // Take host:port (before any / or ?)
    let host_port = after_auth.split('/').next()?;
    let host_port = host_port.split('?').next()?;
    Some(host_port.to_string())
}

fn default_port_for_scheme(url: &str) -> u16 {
    if url.starts_with("postgres") {
        5432
    } else if url.starts_with("http://") {
        80
    } else if url.starts_with("https://") {
        443
    } else {
        0
    }
}

async fn check_tcp_from_url(url: &str) -> serde_json::Value {
    let Some(host_port) = parse_host_port(url) else {
        return serde_json::json!({"status": "unconfigured", "error": "cannot parse URL"});
    };

    let addr = if !host_port.contains(':') {
        format!("{}:{}", host_port, default_port_for_scheme(url))
    } else {
        host_port
    };

    let start = Instant::now();
    match tokio::time::timeout(CHECK_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => serde_json::json!({
            "status": "up",
            "latency_ms": start.elapsed().as_millis(),
        }),
        Ok(Err(e)) => serde_json::json!({
            "status": "down",
            "error": format!("{e}"),
        }),
        Err(_) => serde_json::json!({
            "status": "down",
            "error": "timeout",
        }),
    }
}

async fn check_ollama_ps(base_url: &str) -> serde_json::Value {
    let url = format!("{}/api/ps", base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let start = Instant::now();

    match tokio::time::timeout(CHECK_TIMEOUT, client.get(&url).send()).await {
        Ok(Ok(resp)) => match resp.json::<serde_json::Value>().await {
            Ok(body) => {
                let latency = start.elapsed().as_millis();
                let mut loaded = serde_json::Map::new();

                if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
                    for m in models {
                        let name = m
                            .get("name")
                            .or_else(|| m.get("model"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("unknown");
                        let size_vram = m.get("size_vram").and_then(|v| v.as_u64()).unwrap_or(0);
                        loaded.insert(
                            name.to_string(),
                            serde_json::json!({ "vram_bytes": size_vram }),
                        );
                    }
                }

                serde_json::json!({
                    "status": "up",
                    "latency_ms": latency,
                    "loaded_models": loaded,
                })
            }
            Err(e) => serde_json::json!({
                "status": "up",
                "latency_ms": start.elapsed().as_millis(),
                "error": format!("parse error: {e}"),
            }),
        },
        Ok(Err(e)) => serde_json::json!({
            "status": "down",
            "error": format!("{e}"),
        }),
        Err(_) => serde_json::json!({
            "status": "down",
            "error": "timeout",
        }),
    }
}

fn build_ollama_status(
    ps: serde_json::Value,
    warmup: &std::collections::HashMap<String, ModelWarmupStatus>,
    config: &expert_config::Config,
) -> serde_json::Value {
    let status = ps.get("status").and_then(|s| s.as_str()).unwrap_or("down");
    let latency = ps.get("latency_ms").cloned();

    let loaded = ps
        .get("loaded_models")
        .and_then(|m| m.as_object())
        .cloned()
        .unwrap_or_default();

    let model_status = |name: &str| -> serde_json::Value {
        let warmup_str = match warmup.get(name) {
            Some(ModelWarmupStatus::Warm) => "warm",
            Some(ModelWarmupStatus::Warming) => "warming",
            Some(ModelWarmupStatus::Error(_)) => "error",
            Some(ModelWarmupStatus::Cold) | None => "cold",
        };

        let vram_bytes = loaded
            .get(name)
            .and_then(|v| v.get("vram_bytes"))
            .and_then(|v| v.as_u64());

        let mut obj = serde_json::json!({ "status": warmup_str, "name": name });
        if let Some(vram) = vram_bytes {
            obj["vram_bytes"] = serde_json::json!(vram);
        }
        if let Some(ModelWarmupStatus::Error(msg)) = warmup.get(name) {
            obj["error"] = serde_json::json!(msg);
        }
        obj
    };

    let mut result = serde_json::json!({
        "status": status,
        "models": {
            "llm": model_status(&config.llm_model),
            "embeddings": model_status(&config.embeddings_model),
        },
    });

    if let Some(lat) = latency {
        result["latency_ms"] = lat;
    }

    result
}

async fn read_gpu_metrics() -> serde_json::Value {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,nounits,noheader",
        ])
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return serde_json::json!([]),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let gpus: Vec<serde_json::Value> = stdout
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() < 6 {
                return None;
            }
            Some(serde_json::json!({
                "index": fields[0].parse::<u32>().unwrap_or(0),
                "name": fields[1],
                "vram_used_mb": fields[2].parse::<u64>().unwrap_or(0),
                "vram_total_mb": fields[3].parse::<u64>().unwrap_or(0),
                "utilization_pct": fields[4].parse::<u32>().unwrap_or(0),
                "temp_c": fields[5].parse::<u32>().unwrap_or(0),
            }))
        })
        .collect();

    serde_json::json!(gpus)
}

async fn read_proc_metrics() -> serde_json::Value {
    let mut result = serde_json::json!({});

    // /proc/loadavg -> "0.45 0.38 0.30 1/423 12345"
    if let Ok(data) = tokio::fs::read_to_string("/proc/loadavg").await {
        let parts: Vec<&str> = data.split_whitespace().collect();
        if parts.len() >= 3 {
            result["load_1m"] = serde_json::json!(parts[0].parse::<f64>().unwrap_or(0.0));
            result["load_5m"] = serde_json::json!(parts[1].parse::<f64>().unwrap_or(0.0));
            result["load_15m"] = serde_json::json!(parts[2].parse::<f64>().unwrap_or(0.0));
        }
    }

    // CPU core count from /proc/stat (count "cpu\d" lines)
    if let Ok(data) = tokio::fs::read_to_string("/proc/stat").await {
        let cores = data
            .lines()
            .filter(|l| {
                l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(|b| b.is_ascii_digit())
            })
            .count();
        result["cpu_cores"] = serde_json::json!(cores);
    }

    // /proc/meminfo
    if let Ok(data) = tokio::fs::read_to_string("/proc/meminfo").await {
        let mut total_kb: u64 = 0;
        let mut available_kb: u64 = 0;
        let mut buffers_kb: u64 = 0;
        let mut cached_kb: u64 = 0;
        let mut free_kb: u64 = 0;
        let mut swap_total_kb: u64 = 0;
        let mut swap_free_kb: u64 = 0;

        for line in data.lines() {
            let parse_val = |l: &str| -> u64 {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            };
            if line.starts_with("MemTotal:") {
                total_kb = parse_val(line);
            } else if line.starts_with("MemAvailable:") {
                available_kb = parse_val(line);
            } else if line.starts_with("MemFree:") {
                free_kb = parse_val(line);
            } else if line.starts_with("Buffers:") {
                buffers_kb = parse_val(line);
            } else if line.starts_with("Cached:") && !line.starts_with("CachedSwap") {
                cached_kb = parse_val(line);
            } else if line.starts_with("SwapTotal:") {
                swap_total_kb = parse_val(line);
            } else if line.starts_with("SwapFree:") {
                swap_free_kb = parse_val(line);
            }
        }

        if available_kb == 0 {
            available_kb = free_kb + buffers_kb + cached_kb;
        }

        let used_kb = total_kb.saturating_sub(available_kb);
        result["mem_total_mb"] = serde_json::json!(total_kb / 1024);
        result["mem_used_mb"] = serde_json::json!(used_kb / 1024);
        result["mem_available_mb"] = serde_json::json!(available_kb / 1024);
        result["swap_total_mb"] = serde_json::json!(swap_total_kb / 1024);
        result["swap_used_mb"] = serde_json::json!((swap_total_kb - swap_free_kb) / 1024);
    }

    // /proc/net/dev -> cumulative bytes for rate computation on the frontend
    if let Ok(data) = tokio::fs::read_to_string("/proc/net/dev").await {
        let mut rx_total: u64 = 0;
        let mut tx_total: u64 = 0;
        for line in data.lines().skip(2) {
            let line = line.trim();
            if line.starts_with("lo:") {
                continue;
            }
            if let Some((_iface, rest)) = line.split_once(':') {
                let fields: Vec<&str> = rest.split_whitespace().collect();
                if fields.len() >= 10 {
                    rx_total += fields[0].parse::<u64>().unwrap_or(0);
                    tx_total += fields[8].parse::<u64>().unwrap_or(0);
                }
            }
        }
        result["net_rx_bytes"] = serde_json::json!(rx_total);
        result["net_tx_bytes"] = serde_json::json!(tx_total);
    }

    // Uptime
    if let Ok(data) = tokio::fs::read_to_string("/proc/uptime").await {
        if let Some(secs_str) = data.split_whitespace().next() {
            if let Ok(secs) = secs_str.parse::<f64>() {
                result["uptime_secs"] = serde_json::json!(secs as u64);
            }
        }
    }

    result
}

fn qdrant_value_to_json(v: &qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;
    match &v.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::json!(i),
        Some(Kind::DoubleValue(d)) => serde_json::json!(d),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            let items: Vec<serde_json::Value> =
                list.values.iter().map(qdrant_value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Some(Kind::StructValue(s)) => {
            let map: serde_json::Map<String, serde_json::Value> = s
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        None => serde_json::Value::Null,
    }
}
