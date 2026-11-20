//! Read-only Mistral Vibe subscription budget acquisition.
//!
//! Vibe publishes its monthly Vibe Code allowance only through its own local
//! app server (`account/read`). This adapter directly spawns
//! `vibe-app-server` (no shell), performs `initialize`/`initialized`, opens one
//! throwaway `ephemeral` session (which Vibe neither persists nor counts as a
//! new session), sends `account/read`, and shuts the child down.
//!
//! Only `account.status` and `account.budget.{usagePercentage, resetAt,
//! paygEnabled}` are deserialized. Plan names, API-key previews, identity,
//! and the transient session identifier are never retained or persisted.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{json, value::RawValue};
use thiserror::Error;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use uuid::Uuid;

use crate::{
    domain::{
        Availability, LimitKind, Provider, Quality, QuotaSnapshot, QuotaSnapshotInput, WindowKind,
    },
    providers::codex::{ChildTransport, LineTransport, MAX_PROTOCOL_LINE_BYTES, TransportError},
};

/// Program spawned when no explicit path is configured; resolved via `PATH`.
pub const DEFAULT_APP_SERVER: &str = "vibe-app-server";
/// Scope key of the single monthly Vibe Code allowance row.
pub const BUDGET_SCOPE_KEY: &str = "vibe_code";
/// `usagePercentage` is already expressed on a 0–100 scale.
const USAGE_PERCENT_SCALE: f64 = 1.0;
const MAX_MESSAGES_PER_REQUEST: usize = 256;
const MAX_SESSION_ID_BYTES: usize = 128;
const INITIALIZE_ID: u64 = 1;
const SESSION_START_ID: u64 = 2;
const ACCOUNT_READ_ID: u64 = 3;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, PartialEq)]
pub struct VibeAccountObservation {
    pub snapshots: Vec<QuotaSnapshotInput>,
}

/// Sanitized failure classes; variants never carry provider output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VibeAccountError {
    #[error("unable to start the Mistral Vibe app server")]
    Spawn,
    #[error("Mistral Vibe {method} failed at the transport boundary")]
    Transport { method: &'static str },
    #[error("Mistral Vibe {method} timed out")]
    Timeout { method: &'static str },
    #[error("Mistral Vibe {method} response violated the protocol")]
    Protocol { method: &'static str },
    #[error("Mistral Vibe {method} request was rejected")]
    Rpc { method: &'static str },
    #[error("Mistral Vibe reported no subscription budget")]
    Unavailable,
}

impl VibeAccountError {
    pub const fn class(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::Transport { .. } => "transport",
            Self::Timeout { .. } => "timeout",
            Self::Protocol { .. } => "protocol",
            Self::Rpc { .. } => "rpc",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VibeAccountAdapter {
    executable: PathBuf,
    timeout: Duration,
}

impl VibeAccountAdapter {
    pub fn new(executable: Option<PathBuf>) -> Self {
        Self {
            executable: executable.unwrap_or_else(|| PathBuf::from(DEFAULT_APP_SERVER)),
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn read_budget(
        &self,
        observed_at: i64,
    ) -> Result<VibeAccountObservation, VibeAccountError> {
        // A neutral working directory keeps project-level Vibe configuration
        // and hooks of the dashboard's own cwd out of the read.
        let workdir = std::env::temp_dir();
        let mut transport = ChildTransport::spawn_with(&self.executable, &[], Some(&workdir))
            .map_err(|_| VibeAccountError::Spawn)?
            .with_shutdown_grace(SHUTDOWN_GRACE);
        query_with_transport(&mut transport, &workdir, observed_at, self.timeout)
    }
}

pub fn query_with_transport<T: LineTransport>(
    transport: &mut T,
    workdir: &Path,
    observed_at: i64,
    timeout: Duration,
) -> Result<VibeAccountObservation, VibeAccountError> {
    let result = query_inner(transport, workdir, observed_at, timeout);
    transport.shutdown();
    result
}

#[derive(Deserialize)]
struct SessionStartResult {
    state: SessionState,
}

#[derive(Deserialize)]
struct SessionState {
    session: SessionRef,
}

#[derive(Deserialize)]
struct SessionRef {
    id: String,
}

#[derive(Deserialize)]
struct AccountReadResult {
    account: AccountView,
}

#[derive(Deserialize)]
struct AccountView {
    status: String,
    budget: Option<BudgetView>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BudgetView {
    usage_percentage: f64,
    reset_at: Option<String>,
    payg_enabled: Option<bool>,
}

#[derive(Deserialize)]
struct Envelope<'a> {
    #[serde(borrow)]
    id: Option<&'a RawValue>,
    #[serde(borrow)]
    error: Option<&'a RawValue>,
    #[serde(borrow)]
    result: Option<&'a RawValue>,
}

fn query_inner<T: LineTransport>(
    transport: &mut T,
    workdir: &Path,
    observed_at: i64,
    timeout: Duration,
) -> Result<VibeAccountObservation, VibeAccountError> {
    let workdir = workdir.to_str().ok_or(VibeAccountError::Spawn)?;
    request::<_, serde::de::IgnoredAny>(
        transport,
        "initialize",
        INITIALIZE_ID,
        json!({"clientInfo":{
            "name":"agent_usage_dashboard","title":"Agent Usage Dashboard",
            "version":env!("CARGO_PKG_VERSION")
        }}),
        timeout,
    )?;
    send(
        transport,
        "initialized",
        &json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
    )?;
    let started: SessionStartResult = request(
        transport,
        "session/start",
        SESSION_START_ID,
        json!({"kind":"ephemeral","sessionOnly":true,"agentConfig":{"cwd":workdir}}),
        timeout,
    )?;
    let session_id = started.state.session.id;
    if !is_valid_session_id(&session_id) {
        return Err(VibeAccountError::Protocol {
            method: "session/start",
        });
    }
    let account: AccountReadResult = request(
        transport,
        "account/read",
        ACCOUNT_READ_ID,
        json!({ "sessionId": session_id }),
        timeout,
    )?;
    parse_account(account.account, observed_at)
}

fn parse_account(
    account: AccountView,
    observed_at: i64,
) -> Result<VibeAccountObservation, VibeAccountError> {
    const METHOD: &str = "account/read";
    if account.status != "ready" {
        return Err(VibeAccountError::Unavailable);
    }
    let budget = account.budget.ok_or(VibeAccountError::Unavailable)?;
    let raw_percent = budget.usage_percentage * USAGE_PERCENT_SCALE;
    if !raw_percent.is_finite() || raw_percent < 0.0 {
        return Err(VibeAccountError::Protocol { method: METHOD });
    }
    let payg = budget.payg_enabled.unwrap_or(false);
    let resets_at = budget.reset_at.as_deref().and_then(parse_reset);
    // Past 100% pay-as-you-go bills instead of pausing; the subscription bar
    // stays full either way, and only a non-PAYG exhausted budget is blocked.
    let availability = if raw_percent >= 100.0 && !payg {
        Availability::Blocked
    } else {
        Availability::Allowed
    };
    let input = QuotaSnapshotInput {
        provider: Provider::MistralVibe,
        scope_key: BUDGET_SCOPE_KEY.to_owned(),
        window_kind: WindowKind::Monthly,
        limit_kind: LimitKind::Limited,
        window_duration_seconds: None,
        used_percent: raw_percent.min(100.0),
        resets_at,
        observed_at,
        availability,
        source_version: None,
        quality: if resets_at.is_some() {
            Quality::Fresh
        } else {
            Quality::Partial
        },
        source_sequence: Uuid::new_v4(),
    };
    QuotaSnapshot::new(input.clone()).map_err(|_| VibeAccountError::Protocol { method: METHOD })?;
    Ok(VibeAccountObservation {
        snapshots: vec![input],
    })
}

fn parse_reset(value: &str) -> Option<i64> {
    let seconds = OffsetDateTime::parse(value, &Rfc3339)
        .ok()?
        .unix_timestamp();
    (seconds >= 0).then_some(seconds)
}

fn is_valid_session_id(id: &str) -> bool {
    (1..=MAX_SESSION_ID_BYTES).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn send<T: LineTransport>(
    transport: &mut T,
    method: &'static str,
    message: &serde_json::Value,
) -> Result<(), VibeAccountError> {
    let bytes = serde_json::to_vec(message).map_err(|_| VibeAccountError::Protocol { method })?;
    transport
        .send_line(&bytes)
        .map_err(|_| VibeAccountError::Transport { method })
}

fn request<T: LineTransport, R: DeserializeOwned>(
    transport: &mut T,
    method: &'static str,
    request_id: u64,
    params: serde_json::Value,
    timeout: Duration,
) -> Result<R, VibeAccountError> {
    send(
        transport,
        method,
        &json!({"jsonrpc":"2.0","id":request_id,"method":method,"params":params}),
    )?;
    let deadline = Instant::now() + timeout;
    for _ in 0..MAX_MESSAGES_PER_REQUEST {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(VibeAccountError::Timeout { method });
        }
        let line = transport
            .receive_line(remaining)
            .map_err(|error| match error {
                TransportError::Timeout => VibeAccountError::Timeout { method },
                _ => VibeAccountError::Transport { method },
            })?;
        if line.len() > MAX_PROTOCOL_LINE_BYTES {
            return Err(VibeAccountError::Protocol { method });
        }
        let envelope: Envelope<'_> =
            serde_json::from_slice(&line).map_err(|_| VibeAccountError::Protocol { method })?;
        if !envelope.id.is_some_and(|id| id_matches(id, request_id)) {
            continue;
        }
        if envelope.error.is_some() {
            return Err(VibeAccountError::Rpc { method });
        }
        let result = envelope
            .result
            .ok_or(VibeAccountError::Protocol { method })?;
        return serde_json::from_str(result.get())
            .map_err(|_| VibeAccountError::Protocol { method });
    }
    Err(VibeAccountError::Protocol { method })
}

fn id_matches(id: &RawValue, expected: u64) -> bool {
    let text = id.get().trim().trim_matches('"');
    text.parse::<u64>().ok() == Some(expected)
}

#[cfg(test)]
#[path = "vibe_account_tests.rs"]
mod tests;
