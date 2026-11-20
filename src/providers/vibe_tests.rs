use super::*;

const CONTEXT_WINDOW: u64 = 200_000;
const OBSERVED_AT: i64 = 1_700_000_000_000;

fn payload(json: &str) -> Vec<u8> {
    json.as_bytes().to_vec()
}

fn one_snapshot(observation: VibeObservation) -> QuotaSnapshotInput {
    let [snapshot] = observation.snapshots.try_into().expect("one snapshot");
    snapshot
}

#[test]
fn parses_post_llm_usage_with_model_scope() {
    let observation = parse_hook_payload(
        &payload(
            r#"{
                "hook_event_name": "post_llm",
                "session_id": "00000000-0000-0000-0000-000000000000",
                "parent_session_id": null,
                "transcript_path": "C:\\Users\\me\\.vibe\\logs\\session\\session.json",
                "cwd": "C:\\repo",
                "agent_name": "accept-edits",
                "turn_id": "turn-1",
                "model": "mistral-medium-3-5",
                "response_text": "should be discarded",
                "usage": {"input_tokens": 150000, "output_tokens": 10000, "total_tokens": 160000}
            }"#,
        ),
        OBSERVED_AT,
        CONTEXT_WINDOW,
    )
    .unwrap();
    assert_eq!(observation.rejected_usages, 0);
    let snapshot = one_snapshot(observation);
    assert_eq!(snapshot.provider, Provider::MistralVibe);
    assert_eq!(snapshot.scope_key, "mistral-medium-3-5");
    assert_eq!(snapshot.window_kind, WindowKind::Other);
    assert_eq!(snapshot.limit_kind, LimitKind::Limited);
    assert_eq!(snapshot.window_duration_seconds, None);
    assert_eq!(snapshot.used_percent, 80.0);
    assert_eq!(snapshot.resets_at, None);
    assert_eq!(snapshot.observed_at, OBSERVED_AT);
    assert_eq!(snapshot.availability, Availability::Unknown);
    assert_eq!(snapshot.quality, Quality::Partial);
}

#[test]
fn uses_default_scope_without_model() {
    let observation = parse_hook_payload(
        &payload(r#"{"hook_event_name": "post_llm", "usage": {"total_tokens": 2000}}"#),
        OBSERVED_AT,
        CONTEXT_WINDOW,
    )
    .unwrap();
    let snapshot = one_snapshot(observation);
    assert_eq!(snapshot.scope_key, "session");
    assert_eq!(snapshot.used_percent, 1.0);
}

#[test]
fn clamps_percent_above_the_window() {
    let observation = parse_hook_payload(
        &payload(r#"{"hook_event_name": "post_llm", "usage": {"total_tokens": 999999999}}"#),
        OBSERVED_AT,
        CONTEXT_WINDOW,
    )
    .unwrap();
    assert_eq!(one_snapshot(observation).used_percent, 100.0);
}

#[test]
fn missing_or_null_usage_is_an_empty_observation() {
    for json in [
        r#"{"hook_event_name": "post_llm"}"#,
        r#"{"hook_event_name": "post_llm", "usage": null}"#,
    ] {
        let observation = parse_hook_payload(&payload(json), OBSERVED_AT, CONTEXT_WINDOW).unwrap();
        assert!(observation.snapshots.is_empty());
        assert_eq!(observation.rejected_usages, 0);
    }
}

#[test]
fn usage_without_total_is_rejected() {
    let observation = parse_hook_payload(
        &payload(r#"{"hook_event_name": "post_llm", "usage": {"input_tokens": 5}}"#),
        OBSERVED_AT,
        CONTEXT_WINDOW,
    )
    .unwrap();
    assert!(observation.snapshots.is_empty());
    assert_eq!(observation.rejected_usages, 1);
}

#[test]
fn rejects_other_hook_events() {
    for json in [
        r#"{"hook_event_name": "post_agent", "usage": {"total_tokens": 5}}"#,
        r#"{"hook_event_name": "pre_tool", "tool_name": "file_system.bash"}"#,
    ] {
        assert_eq!(
            parse_hook_payload(&payload(json), OBSERVED_AT, CONTEXT_WINDOW),
            Err(VibeParseError::UnexpectedEvent)
        );
    }
    assert_eq!(
        parse_hook_payload(&payload(r#"{"usage": {}}"#), OBSERVED_AT, CONTEXT_WINDOW),
        Err(VibeParseError::UnexpectedEvent)
    );
}

#[test]
fn rejects_invalid_model_keys() {
    for json in [
        r#"{"hook_event_name": "post_llm", "model": "bad model!", "usage": {"total_tokens": 5}}"#,
        r#"{"hook_event_name": "post_llm", "model": "", "usage": {"total_tokens": 5}}"#,
        r#"{"hook_event_name": "post_llm", "model": 42, "usage": {"total_tokens": 5}}"#,
    ] {
        let observation = parse_hook_payload(&payload(json), OBSERVED_AT, CONTEXT_WINDOW).unwrap();
        assert!(observation.snapshots.is_empty());
        assert_eq!(observation.rejected_usages, 1);
    }
}

#[test]
fn allows_common_model_key_shapes() {
    for model in ["mistral-medium-3.5", "devstral", "my_local-model.v2"] {
        let json = format!(
            r#"{{"hook_event_name": "post_llm", "model": "{model}", "usage": {{"total_tokens": 100}}}}"#
        );
        let observation = parse_hook_payload(&payload(&json), OBSERVED_AT, CONTEXT_WINDOW).unwrap();
        assert_eq!(one_snapshot(observation).scope_key, model);
    }
}

#[test]
fn rejects_invalid_documents() {
    assert_eq!(
        parse_hook_payload(&payload("{"), OBSERVED_AT, CONTEXT_WINDOW),
        Err(VibeParseError::InvalidDocument)
    );
    assert_eq!(
        parse_hook_payload(&payload(r#"[1, 2]"#), OBSERVED_AT, CONTEXT_WINDOW),
        Err(VibeParseError::InvalidDocument)
    );
}

#[test]
fn rejects_oversized_input() {
    let json = format!(
        r#"{{"hook_event_name": "post_llm", "pad": "{}", "usage": {{"total_tokens": 1}}}}"#,
        "a".repeat(MAX_STATUS_INPUT_BYTES)
    );
    assert_eq!(
        parse_hook_payload(&payload(&json), OBSERVED_AT, CONTEXT_WINDOW),
        Err(VibeParseError::Oversized)
    );
}

#[test]
fn rejects_out_of_range_context_windows() {
    for window in [
        0,
        MIN_CONTEXT_WINDOW_TOKENS - 1,
        MAX_CONTEXT_WINDOW_TOKENS + 1,
    ] {
        assert_eq!(
            parse_hook_payload(
                &payload(r#"{"hook_event_name": "post_llm", "usage": {"total_tokens": 1}}"#),
                OBSERVED_AT,
                window
            ),
            Err(VibeParseError::InvalidContextWindow)
        );
    }
}

#[test]
fn zero_tokens_are_accepted() {
    let observation = parse_hook_payload(
        &payload(r#"{"hook_event_name": "post_llm", "usage": {"total_tokens": 0}}"#),
        OBSERVED_AT,
        CONTEXT_WINDOW,
    )
    .unwrap();
    assert_eq!(one_snapshot(observation).used_percent, 0.0);
}
