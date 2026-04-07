use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Comparison operators for field predicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    Eq,
    Ne,
    In,
    NotIn,
    Exists,
    NotExists,
    Contains,
    Matches,
}

/// A single predicate comparing a metadata field to a value.
///
/// The `field` string supports dot-delimited paths for nested JSON objects
/// (e.g. `"sender.name"` traverses `metadata["sender"]["name"]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPredicate {
    pub field: String,
    pub op: FilterOp,
    #[serde(default)]
    pub value: Value,

    /// Compiled regex (populated lazily on first `Matches` evaluation).
    #[serde(skip)]
    compiled_regex: OnceLock<Option<Regex>>,
}

impl PartialEq for FieldPredicate {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.op == other.op && self.value == other.value
    }
}

impl FieldPredicate {
    pub fn new(field: impl Into<String>, op: FilterOp, value: Value) -> Self {
        Self {
            field: field.into(),
            op,
            value,
            compiled_regex: OnceLock::new(),
        }
    }

    fn compiled_regex(&self) -> Option<&Regex> {
        self.compiled_regex
            .get_or_init(|| self.value.as_str().and_then(|pat| Regex::new(pat).ok()))
            .as_ref()
    }
}

/// Composable event filter tree.
///
/// Activities declare filters over `Event.metadata` fields. The SSM worker
/// evaluates the filter at fan-out time to decide whether an event is
/// delivered to an activity.
///
/// `EventFilter::All` (the default) matches every event, preserving
/// backward compatibility for activities created without a filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EventFilter {
    /// Composite: all children must match.
    #[serde(rename = "and")]
    And { and: Vec<EventFilter> },
    /// Composite: at least one child must match.
    #[serde(rename = "or")]
    Or { or: Vec<EventFilter> },
    /// Negate a child filter.
    #[serde(rename = "not")]
    Not { not: Box<EventFilter> },
    /// Leaf predicate on a metadata field.
    Field(FieldPredicate),
    /// Match all events (no filtering).
    All,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self::All
    }
}

impl EventFilter {
    /// Evaluate this filter against an event's metadata map.
    /// Returns `true` if the event should be delivered to the activity.
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            Self::All => true,
            Self::And { and } => and.iter().all(|child| child.matches(metadata)),
            Self::Or { or } => or.iter().any(|child| child.matches(metadata)),
            Self::Not { not } => !not.matches(metadata),
            Self::Field(pred) => eval_predicate(pred, metadata),
        }
    }

    /// Validate filter structure, returning a list of errors (empty = valid).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        validate_recursive(self, 0, &mut errors, &mut 0);
        errors
    }
}

const MAX_DEPTH: usize = 8;
const MAX_PREDICATES: usize = 64;

fn validate_recursive(
    filter: &EventFilter,
    depth: usize,
    errors: &mut Vec<String>,
    predicate_count: &mut usize,
) {
    if depth > MAX_DEPTH {
        errors.push(format!(
            "filter nesting exceeds maximum depth of {MAX_DEPTH}"
        ));
        return;
    }

    match filter {
        EventFilter::All => {}
        EventFilter::And { and } => {
            if and.is_empty() {
                errors.push("'and' filter must contain at least one child".into());
            }
            for child in and {
                validate_recursive(child, depth + 1, errors, predicate_count);
            }
        }
        EventFilter::Or { or } => {
            if or.is_empty() {
                errors.push("'or' filter must contain at least one child".into());
            }
            for child in or {
                validate_recursive(child, depth + 1, errors, predicate_count);
            }
        }
        EventFilter::Not { not } => {
            validate_recursive(not, depth + 1, errors, predicate_count);
        }
        EventFilter::Field(pred) => {
            *predicate_count += 1;
            if *predicate_count > MAX_PREDICATES {
                errors.push(format!(
                    "filter exceeds maximum of {MAX_PREDICATES} predicates"
                ));
                return;
            }
            validate_predicate(pred, errors);
        }
    }
}

fn validate_predicate(pred: &FieldPredicate, errors: &mut Vec<String>) {
    if pred.field.is_empty() {
        errors.push("field name must not be empty".into());
    }

    match pred.op {
        FilterOp::Eq | FilterOp::Ne | FilterOp::Contains => {
            if !matches!(
                pred.value,
                Value::String(_) | Value::Number(_) | Value::Bool(_)
            ) {
                errors.push(format!(
                    "field '{}': {:?} requires a string, number, or bool value",
                    pred.field, pred.op
                ));
            }
        }
        FilterOp::In | FilterOp::NotIn => {
            if !pred.value.is_array() {
                errors.push(format!(
                    "field '{}': {:?} requires an array value",
                    pred.field, pred.op
                ));
            }
        }
        FilterOp::Exists | FilterOp::NotExists => {
            // value is ignored
        }
        FilterOp::Matches => match pred.value.as_str() {
            Some(pattern) => {
                if Regex::new(pattern).is_err() {
                    errors.push(format!(
                        "field '{}': invalid regex pattern '{}'",
                        pred.field, pattern
                    ));
                }
            }
            None => {
                errors.push(format!(
                    "field '{}': 'matches' requires a string regex pattern",
                    pred.field
                ));
            }
        },
    }
}

/// Resolve a dot-delimited field path against the metadata map.
fn resolve_field<'a>(field: &str, metadata: &'a HashMap<String, Value>) -> Option<&'a Value> {
    let mut parts = field.splitn(2, '.');

    let root_key = parts.next()?;
    let root_val = metadata.get(root_key)?;

    match parts.next() {
        None => Some(root_val),
        Some(rest) => resolve_nested(rest, root_val),
    }
}

fn resolve_nested<'a>(path: &str, val: &'a Value) -> Option<&'a Value> {
    let mut current = val;
    for segment in path.split('.') {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

fn eval_predicate(pred: &FieldPredicate, metadata: &HashMap<String, Value>) -> bool {
    match pred.op {
        FilterOp::Exists => resolve_field(&pred.field, metadata).is_some(),
        FilterOp::NotExists => resolve_field(&pred.field, metadata).is_none(),
        _ => {
            let field_val = match resolve_field(&pred.field, metadata) {
                Some(v) => v,
                None => return false,
            };
            match pred.op {
                FilterOp::Eq => values_equal(field_val, &pred.value),
                FilterOp::Ne => !values_equal(field_val, &pred.value),
                FilterOp::In => match pred.value.as_array() {
                    Some(arr) => arr.iter().any(|v| values_equal(field_val, v)),
                    None => false,
                },
                FilterOp::NotIn => match pred.value.as_array() {
                    Some(arr) => !arr.iter().any(|v| values_equal(field_val, v)),
                    None => true,
                },
                FilterOp::Contains => match (field_val.as_str(), pred.value.as_str()) {
                    (Some(haystack), Some(needle)) => haystack.contains(needle),
                    _ => false,
                },
                FilterOp::Matches => match field_val.as_str() {
                    Some(text) => pred.compiled_regex().map_or(false, |re| re.is_match(text)),
                    None => false,
                },
                FilterOp::Exists | FilterOp::NotExists => unreachable!(),
            }
        }
    }
}

/// Compare two JSON values with cross-type numeric coercion.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            a.as_f64().zip(b.as_f64()).map_or(false, |(x, y)| x == y)
        }
        _ => a == b,
    }
}

impl fmt::Display for EventFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "*"),
            Self::And { and } => {
                let parts: Vec<String> = and.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", parts.join(" AND "))
            }
            Self::Or { or } => {
                let parts: Vec<String> = or.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", parts.join(" OR "))
            }
            Self::Not { not } => write!(f, "NOT {not}"),
            Self::Field(pred) => write!(f, "{} {:?} {}", pred.field, pred.op, pred.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_all_matches_everything() {
        let filter = EventFilter::All;
        assert!(filter.matches(&HashMap::new()));
        assert!(filter.matches(&meta(&[("x", Value::from("y"))])));
    }

    #[test]
    fn test_eq_string() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "channel_id",
            FilterOp::Eq,
            Value::from("dm-alice"),
        ));
        assert!(filter.matches(&meta(&[("channel_id", Value::from("dm-alice"))])));
        assert!(!filter.matches(&meta(&[("channel_id", Value::from("dm-bob"))])));
        assert!(!filter.matches(&HashMap::new()));
    }

    #[test]
    fn test_ne() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "author",
            FilterOp::Ne,
            Value::from("bot"),
        ));
        assert!(filter.matches(&meta(&[("author", Value::from("human"))])));
        assert!(!filter.matches(&meta(&[("author", Value::from("bot"))])));
        // Missing field: returns false for Ne (because the base eval returns false)
        assert!(!filter.matches(&HashMap::new()));
    }

    #[test]
    fn test_in_operator() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "symbol",
            FilterOp::In,
            Value::from(vec!["AAPL", "MSFT", "GOOG"]),
        ));
        assert!(filter.matches(&meta(&[("symbol", Value::from("AAPL"))])));
        assert!(filter.matches(&meta(&[("symbol", Value::from("GOOG"))])));
        assert!(!filter.matches(&meta(&[("symbol", Value::from("TSLA"))])));
    }

    #[test]
    fn test_not_in() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "event_type",
            FilterOp::NotIn,
            Value::from(vec!["typing", "presence"]),
        ));
        assert!(filter.matches(&meta(&[("event_type", Value::from("message"))])));
        assert!(!filter.matches(&meta(&[("event_type", Value::from("typing"))])));
    }

    #[test]
    fn test_exists_and_not_exists() {
        let exists = EventFilter::Field(FieldPredicate::new(
            "channel_id",
            FilterOp::Exists,
            Value::Null,
        ));
        let not_exists = EventFilter::Field(FieldPredicate::new(
            "channel_id",
            FilterOp::NotExists,
            Value::Null,
        ));

        let with = meta(&[("channel_id", Value::from("abc"))]);
        let without = HashMap::new();

        assert!(exists.matches(&with));
        assert!(!exists.matches(&without));
        assert!(!not_exists.matches(&with));
        assert!(not_exists.matches(&without));
    }

    #[test]
    fn test_contains() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "raw_text",
            FilterOp::Contains,
            Value::from("attacks"),
        ));
        assert!(filter.matches(&meta(&[(
            "raw_text",
            Value::from("The goblin attacks you!")
        )])));
        assert!(!filter.matches(&meta(&[("raw_text", Value::from("Nothing happens."))])));
    }

    #[test]
    fn test_matches_regex() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "line",
            FilterOp::Matches,
            Value::from(r"^\d+ damage"),
        ));
        assert!(filter.matches(&meta(&[("line", Value::from("42 damage dealt"))])));
        assert!(!filter.matches(&meta(&[("line", Value::from("no damage"))])));
    }

    #[test]
    fn test_and_composition() {
        let filter = EventFilter::And {
            and: vec![
                EventFilter::Field(FieldPredicate::new(
                    "channel_id",
                    FilterOp::Eq,
                    Value::from("dm-alice"),
                )),
                EventFilter::Field(FieldPredicate::new(
                    "author",
                    FilterOp::Ne,
                    Value::from("bot"),
                )),
            ],
        };
        let m = meta(&[
            ("channel_id", Value::from("dm-alice")),
            ("author", Value::from("alice")),
        ]);
        assert!(filter.matches(&m));

        let m2 = meta(&[
            ("channel_id", Value::from("dm-alice")),
            ("author", Value::from("bot")),
        ]);
        assert!(!filter.matches(&m2));

        let m3 = meta(&[
            ("channel_id", Value::from("dm-bob")),
            ("author", Value::from("alice")),
        ]);
        assert!(!filter.matches(&m3));
    }

    #[test]
    fn test_or_composition() {
        let filter = EventFilter::Or {
            or: vec![
                EventFilter::Field(FieldPredicate::new(
                    "event_type",
                    FilterOp::Eq,
                    Value::from("combat"),
                )),
                EventFilter::Field(FieldPredicate::new(
                    "event_type",
                    FilterOp::Eq,
                    Value::from("quest"),
                )),
            ],
        };
        assert!(filter.matches(&meta(&[("event_type", Value::from("combat"))])));
        assert!(filter.matches(&meta(&[("event_type", Value::from("quest"))])));
        assert!(!filter.matches(&meta(&[("event_type", Value::from("chat"))])));
    }

    #[test]
    fn test_not() {
        let filter = EventFilter::Not {
            not: Box::new(EventFilter::Field(FieldPredicate::new(
                "event_type",
                FilterOp::Eq,
                Value::from("typing"),
            ))),
        };
        assert!(filter.matches(&meta(&[("event_type", Value::from("message"))])));
        assert!(!filter.matches(&meta(&[("event_type", Value::from("typing"))])));
    }

    #[test]
    fn test_dot_path_resolution() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "sender.name",
            FilterOp::Eq,
            Value::from("alice"),
        ));
        let m = meta(&[("sender", serde_json::json!({"name": "alice", "id": "123"}))]);
        assert!(filter.matches(&m));

        let m2 = meta(&[("sender", serde_json::json!({"name": "bob", "id": "456"}))]);
        assert!(!filter.matches(&m2));
    }

    #[test]
    fn test_numeric_cross_type_eq() {
        let filter =
            EventFilter::Field(FieldPredicate::new("count", FilterOp::Eq, Value::from(42)));
        assert!(filter.matches(&meta(&[("count", Value::from(42))])));
        assert!(filter.matches(&meta(&[("count", Value::from(42.0))])));
        assert!(!filter.matches(&meta(&[("count", Value::from(43))])));
    }

    #[test]
    fn test_validation_valid_filter() {
        let filter = EventFilter::And {
            and: vec![EventFilter::Field(FieldPredicate::new(
                "x",
                FilterOp::Eq,
                Value::from("y"),
            ))],
        };
        assert!(filter.validate().is_empty());
    }

    #[test]
    fn test_validation_empty_and() {
        let filter = EventFilter::And { and: vec![] };
        let errs = filter.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("at least one child"));
    }

    #[test]
    fn test_validation_empty_field_name() {
        let filter = EventFilter::Field(FieldPredicate::new("", FilterOp::Eq, Value::from("x")));
        let errs = filter.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("field name"));
    }

    #[test]
    fn test_validation_in_requires_array() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "x",
            FilterOp::In,
            Value::from("not-an-array"),
        ));
        let errs = filter.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("array"));
    }

    #[test]
    fn test_validation_matches_bad_regex() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "x",
            FilterOp::Matches,
            Value::from("[invalid"),
        ));
        let errs = filter.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("regex"));
    }

    #[test]
    fn test_serde_roundtrip_field() {
        let filter = EventFilter::Field(FieldPredicate::new(
            "channel_id",
            FilterOp::Eq,
            Value::from("abc"),
        ));
        let json = serde_json::to_string(&filter).unwrap();
        let restored: EventFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(filter, restored);
    }

    #[test]
    fn test_serde_roundtrip_and() {
        let filter = EventFilter::And {
            and: vec![
                EventFilter::Field(FieldPredicate::new("a", FilterOp::Eq, Value::from("1"))),
                EventFilter::Field(FieldPredicate::new("b", FilterOp::Ne, Value::from("2"))),
            ],
        };
        let json = serde_json::to_string(&filter).unwrap();
        let restored: EventFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(filter, restored);
    }

    #[test]
    fn test_serde_deserialize_all_from_null() {
        let filter: EventFilter = serde_json::from_str("null").unwrap_or_default();
        assert_eq!(filter, EventFilter::All);
    }

    #[test]
    fn test_display() {
        let filter = EventFilter::And {
            and: vec![
                EventFilter::Field(FieldPredicate::new(
                    "channel",
                    FilterOp::Eq,
                    Value::from("abc"),
                )),
                EventFilter::Field(FieldPredicate::new(
                    "type",
                    FilterOp::Ne,
                    Value::from("typing"),
                )),
            ],
        };
        let s = filter.to_string();
        assert!(s.contains("AND"));
        assert!(s.contains("channel"));
    }
}
