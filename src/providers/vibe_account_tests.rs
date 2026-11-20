use std::{collections::VecDeque, path::Path, time::Duration};

use serde_json::{Value, json};

use super::*;

const NOW: i64 = 1_700_000_000_000;
const TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Default)]
struct FakeTransport {
    incoming: VecDeque<Result<Vec<u8>, TransportError>>,
    sent: Vec<Value>,
    shutdown: bool,
}

impl FakeTransport {
    fn with_messages(messages: impl IntoIterator<Item = Value>) -> Self {
        Self {
            incoming: messages
                .into_iter()
                .map(|message| Ok(serde_json::to_vec(&message).unwrap()))
                .collect(),
            ..Self::default()
        }
    }
}

impl LineTransport for FakeTransport {
    fn send_line(&mut self, line: &[u8]) -> Result<(), TransportError> {
        self.sent.push(serde_json::from_slice(line).unwrap());
        Ok(())
    }

    fn receive_line(&mut self, _timeout: Duration) -> Result<Vec<u8>, TransportError> {
        self.incoming
            .pop_front()
            .unwrap_or(Err(TransportError::Timeout))
    }

    fn shutdown(&mut self) {
        self.shutdown = true;
    }
}

fn workdir() -> &'static Path {
    Path::new("tmp-workdir")
}

fn started() -> Value {
    json!({"id":2,"result":{"state":{"session":{"id":"0f6c1d2e-aaaa-bbbb-cccc-123456789abc",
        "title":"private title"},"history":[{"text":"private"}]}}})
}

fn account(budget: Value) -> Value {
    json!({"id":3,"result":{"account":{
        "status":"ready",
        "plan":{"kind":"chat","name":"PRIVATE_PLAN","title":"Private"},
        "apiKey":{"preview":"sk-PRIVATE","scope":"user","accessScope":"x"},
        "budget":budget
    }}})
}

fn script(budget: Value) -> FakeTransport {
    FakeTransport::with_messages([
        json!({"id":1,"result":{"serverInfo":{"name":"vibe","version":"3.1.3"},"processId":1}}),
        json!({"method":"session/event","params":{"noise":true}}),
        started(),
        account(budget),
    ])
}

#[test]
fn handshake_is_exact_and_only_read_only_methods_are_sent() {
    let mut transport = script(json!({
        "usagePercentage":20.8,"resetAt":"2026-12-01T00:00:00Z","paygEnabled":false
    }));
    let observation = query_with_transport(&mut transport, workdir(), NOW, TIMEOUT).unwrap();

    let methods: Vec<_> = transport
        .sent
        .iter()
        .map(|message| message["method"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        methods,
        ["initialize", "initialized", "session/start", "account/read"]
    );
    assert_eq!(transport.sent[1]["params"], json!({}));
    assert_eq!(transport.sent[2]["params"]["kind"], "ephemeral");
    assert_eq!(transport.sent[2]["params"]["sessionOnly"], true);
    assert_eq!(
        transport.sent[3]["params"]["sessionId"],
        "0f6c1d2e-aaaa-bbbb-cccc-123456789abc"
    );
    assert!(transport.shutdown);

    assert_eq!(observation.snapshots.len(), 1);
    let row = &observation.snapshots[0];
    assert_eq!(row.provider, Provider::MistralVibe);
    assert_eq!(row.scope_key, BUDGET_SCOPE_KEY);
    assert_eq!(row.window_kind, WindowKind::Monthly);
    assert_eq!(row.used_percent, 20.8);
    assert_eq!(row.resets_at, Some(1_796_083_200));
    assert_eq!(row.availability, Availability::Allowed);
    assert_eq!(row.quality, Quality::Fresh);
}

#[test]
fn normalized_row_never_contains_plan_key_or_session_data() {
    let mut transport = script(json!({"usagePercentage":5.0,"resetAt":null}));
    let observation = query_with_transport(&mut transport, workdir(), NOW, TIMEOUT).unwrap();
    let rendered = format!("{observation:?}");
    for secret in ["PRIVATE", "sk-", "0f6c1d2e", "private"] {
        assert!(!rendered.contains(secret), "leaked {secret}");
    }
    assert_eq!(observation.snapshots[0].quality, Quality::Partial);
    assert_eq!(observation.snapshots[0].resets_at, None);
}

#[test]
fn exhausted_budget_is_blocked_without_payg_and_allowed_with_it() {
    let mut transport = script(json!({"usagePercentage":130.0,"paygEnabled":false}));
    let row = query_with_transport(&mut transport, workdir(), NOW, TIMEOUT)
        .unwrap()
        .snapshots
        .remove(0);
    assert_eq!(row.used_percent, 100.0);
    assert_eq!(row.availability, Availability::Blocked);

    let mut transport = script(json!({"usagePercentage":130.0,"paygEnabled":true}));
    let row = query_with_transport(&mut transport, workdir(), NOW, TIMEOUT)
        .unwrap()
        .snapshots
        .remove(0);
    assert_eq!(row.availability, Availability::Allowed);
}

#[test]
fn missing_budget_or_unready_account_is_unavailable() {
    let mut transport = script(Value::Null);
    assert_eq!(
        query_with_transport(&mut transport, workdir(), NOW, TIMEOUT),
        Err(VibeAccountError::Unavailable)
    );

    let mut transport = FakeTransport::with_messages([
        json!({"id":1,"result":{}}),
        started(),
        json!({"id":3,"result":{"account":{"status":"unavailable"}}}),
    ]);
    assert_eq!(
        query_with_transport(&mut transport, workdir(), NOW, TIMEOUT),
        Err(VibeAccountError::Unavailable)
    );
}

#[test]
fn invalid_numbers_and_session_ids_are_protocol_errors() {
    let mut transport = script(json!({"usagePercentage":-1.0}));
    assert_eq!(
        query_with_transport(&mut transport, workdir(), NOW, TIMEOUT),
        Err(VibeAccountError::Protocol {
            method: "account/read"
        })
    );

    let mut transport = FakeTransport::with_messages([
        json!({"id":1,"result":{}}),
        json!({"id":2,"result":{"state":{"session":{"id":"bad\nid"}}}}),
    ]);
    assert_eq!(
        query_with_transport(&mut transport, workdir(), NOW, TIMEOUT),
        Err(VibeAccountError::Protocol {
            method: "session/start"
        })
    );
    assert!(transport.sent.iter().all(|m| m["method"] != "account/read"));
}

#[test]
fn rpc_errors_and_timeouts_are_sanitized() {
    let mut transport = FakeTransport::with_messages([
        json!({"id":1,"result":{}}),
        json!({"id":2,"error":{"code":-1,"message":"secret detail"}}),
    ]);
    let error = query_with_transport(&mut transport, workdir(), NOW, TIMEOUT).unwrap_err();
    assert_eq!(
        error,
        VibeAccountError::Rpc {
            method: "session/start"
        }
    );
    assert!(!error.to_string().contains("secret"));
    assert_eq!(error.class(), "rpc");

    let mut transport = FakeTransport::with_messages([json!({"id":1,"result":{}})]);
    assert_eq!(
        query_with_transport(&mut transport, workdir(), NOW, TIMEOUT),
        Err(VibeAccountError::Timeout {
            method: "session/start"
        })
    );
    assert!(transport.shutdown);
}

#[test]
fn missing_executable_is_a_spawn_error() {
    let adapter = VibeAccountAdapter::new(Some("definitely-not-a-vibe-app-server-binary".into()));
    assert_eq!(adapter.read_budget(NOW), Err(VibeAccountError::Spawn));
}
