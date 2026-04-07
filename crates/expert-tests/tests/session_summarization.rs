//! Type-level serde tests for session summarization and exchange payloads (no Redis).

use expert_types::context::{ActivityExchange, Exchange, ToolCall};
use expert_types::signals::{SummarizeRequest, SummarizeResult};

fn sample_exchange() -> Exchange {
    Exchange {
        timestamp: 1_700_000_000_000,
        rendered_prompt: "prompt text".to_string(),
        response: "assistant reply".to_string(),
        tool_calls: vec![ToolCall {
            tool_name: "recall".to_string(),
            arguments: serde_json::json!({ "indices": [0, 1] }),
            result: Some(serde_json::json!({ "ok": true })),
        }],
    }
}

#[test]
fn activity_exchange_serde_roundtrip() {
    let original = ActivityExchange {
        activity_id: "activity-uuid".to_string(),
        exchange: sample_exchange(),
    };

    let json = serde_json::to_string(&original).expect("serialize ActivityExchange");
    let back: ActivityExchange = serde_json::from_str(&json).expect("deserialize ActivityExchange");

    assert_eq!(back.activity_id, original.activity_id);
    assert_eq!(back.exchange.timestamp, original.exchange.timestamp);
    assert_eq!(
        back.exchange.rendered_prompt,
        original.exchange.rendered_prompt
    );
    assert_eq!(back.exchange.response, original.exchange.response);
    assert_eq!(
        back.exchange.tool_calls.len(),
        original.exchange.tool_calls.len()
    );
    assert_eq!(
        back.exchange.tool_calls[0].tool_name,
        original.exchange.tool_calls[0].tool_name
    );
}

#[test]
fn summarize_request_result_serde_roundtrip() {
    let req = SummarizeRequest {
        activity_id: "act-1".to_string(),
        session_id: "sess-9".to_string(),
        raw_text: "long session transcript ...".to_string(),
    };

    let req_json = serde_json::to_string(&req).expect("serialize SummarizeRequest");
    let req_back: SummarizeRequest =
        serde_json::from_str(&req_json).expect("deserialize SummarizeRequest");
    assert_eq!(req_back.activity_id, req.activity_id);
    assert_eq!(req_back.session_id, req.session_id);
    assert_eq!(req_back.raw_text, req.raw_text);

    let res = SummarizeResult {
        activity_id: req.activity_id.clone(),
        session_id: req.session_id.clone(),
        compressed_narrative: "short summary".to_string(),
    };

    let res_json = serde_json::to_string(&res).expect("serialize SummarizeResult");
    let res_back: SummarizeResult =
        serde_json::from_str(&res_json).expect("deserialize SummarizeResult");
    assert_eq!(res_back.activity_id, res.activity_id);
    assert_eq!(res_back.session_id, res.session_id);
    assert_eq!(res_back.compressed_narrative, res.compressed_narrative);
}

#[test]
fn exchange_roundtrip_as_json_string() {
    let original = sample_exchange();
    let json_string = serde_json::to_string(&original).expect("serialize Exchange");

    let back: Exchange = serde_json::from_str(&json_string).expect("deserialize Exchange");

    let again = serde_json::to_string(&back).expect("re-serialize");
    assert_eq!(json_string, again, "stable JSON roundtrip");
}
