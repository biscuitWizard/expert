use std::sync::Arc;
use tokio::sync::RwLock;
use axum::Router;
use axum::routing::{delete, get, post};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use expert_redis::{StreamProducer, StateStore};
use expert_tests::*;

struct TestAppState {
    registry: RwLock<Registry>,
    #[allow(dead_code)]
    producer: RwLock<StreamProducer>,
    #[allow(dead_code)]
    state_store: RwLock<StateStore>,
}

struct Registry {
    activities: std::collections::HashMap<String, ActivityEntry>,
}

#[derive(Clone)]
struct ActivityEntry {
    activity_id: String,
    stream_id: String,
    domain: String,
    goal_count: usize,
}

#[derive(Serialize)]
struct ActivityResponse {
    activity_id: String,
    stream_id: String,
    domain: String,
    goal_count: usize,
}

#[derive(Deserialize)]
struct CreateReq {
    stream_id: String,
    domain: Option<String>,
    goals: Vec<GoalInput>,
}

#[derive(Deserialize)]
struct GoalInput {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    description: String,
}

fn test_router(state: Arc<TestAppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/activities", post(create_activity))
        .route("/activities", get(list_activities))
        .route("/activities/{id}", get(get_activity))
        .route("/activities/{id}", delete(delete_activity))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn create_activity(
    State(state): State<Arc<TestAppState>>,
    Json(req): Json<CreateReq>,
) -> impl IntoResponse {
    let activity_id = uuid::Uuid::new_v4().to_string();
    let domain = req.domain.unwrap_or("default".to_string());
    let entry = ActivityEntry {
        activity_id: activity_id.clone(),
        stream_id: req.stream_id.clone(),
        domain: domain.clone(),
        goal_count: req.goals.len(),
    };
    state.registry.write().await.activities.insert(activity_id.clone(), entry.clone());

    let resp = ActivityResponse {
        activity_id: entry.activity_id,
        stream_id: entry.stream_id,
        domain: entry.domain,
        goal_count: entry.goal_count,
    };
    (StatusCode::CREATED, Json(resp)).into_response()
}

async fn list_activities(State(state): State<Arc<TestAppState>>) -> impl IntoResponse {
    let reg = state.registry.read().await;
    let list: Vec<ActivityResponse> = reg.activities.values().map(|e| ActivityResponse {
        activity_id: e.activity_id.clone(),
        stream_id: e.stream_id.clone(),
        domain: e.domain.clone(),
        goal_count: e.goal_count,
    }).collect();
    Json(list)
}

async fn get_activity(
    State(state): State<Arc<TestAppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let reg = state.registry.read().await;
    match reg.activities.get(&id) {
        Some(e) => {
            let resp = ActivityResponse {
                activity_id: e.activity_id.clone(),
                stream_id: e.stream_id.clone(),
                domain: e.domain.clone(),
                goal_count: e.goal_count,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn delete_activity(
    State(state): State<Arc<TestAppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut reg = state.registry.write().await;
    match reg.activities.remove(&id) {
        Some(_) => StatusCode::NO_CONTENT,
        None => StatusCode::NOT_FOUND,
    }
}

async fn setup() -> (reqwest::Client, String) {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let state = Arc::new(TestAppState {
        registry: RwLock::new(Registry { activities: std::collections::HashMap::new() }),
        producer: RwLock::new(StreamProducer::new(conn.clone(), 1000)),
        state_store: RwLock::new(StateStore::new(conn)),
    });

    let app = test_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");
    (client, base)
}

#[tokio::test]
async fn test_health_endpoint() {
    let (client, base) = setup().await;
    let resp = client.get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn test_create_list_delete_activity() {
    let (client, base) = setup().await;

    let body = serde_json::json!({
        "stream_id": "s1",
        "domain": "test",
        "goals": [{"name": "g1", "description": "goal 1"}]
    });

    let create_resp = client.post(format!("{base}/activities"))
        .json(&body)
        .send().await.unwrap();
    assert_eq!(create_resp.status(), 201);
    let created: serde_json::Value = create_resp.json().await.unwrap();
    let activity_id = created["activity_id"].as_str().unwrap().to_string();

    let list_resp = client.get(format!("{base}/activities")).send().await.unwrap();
    assert_eq!(list_resp.status(), 200);
    let list: Vec<serde_json::Value> = list_resp.json().await.unwrap();
    assert_eq!(list.len(), 1);

    let del_resp = client.delete(format!("{base}/activities/{activity_id}"))
        .send().await.unwrap();
    assert_eq!(del_resp.status(), 204);

    let list_resp2 = client.get(format!("{base}/activities")).send().await.unwrap();
    let list2: Vec<serde_json::Value> = list_resp2.json().await.unwrap();
    assert_eq!(list2.len(), 0);
}

#[tokio::test]
async fn test_get_unknown_activity_404() {
    let (client, base) = setup().await;
    let resp = client.get(format!("{base}/activities/nonexistent-id"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_delete_unknown_activity_404() {
    let (client, base) = setup().await;
    let resp = client.delete(format!("{base}/activities/nonexistent-id"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}
