use serde::{Deserialize, de::IgnoredAny};
use thiserror::Error;
use uuid::Uuid;

use crate::domain::{
    Availability, LimitKind, Provider, Quality, QuotaSnapshot, QuotaSnapshotInput, WindowKind,
};

/// Maximum size of one caller-supplied Mistral Vibe hook payload.
pub const MAX_STATUS_INPUT_BYTES: usize = 1024 * 1024;
/// Smallest context window the ingest accepts for token normalization.
pub const MIN_CONTEXT_WINDOW_TOKENS: u64 = 4_096;
/// Largest context window the ingest accepts for token normalization.
pub const MAX_CONTEXT_WINDOW_TOKENS: u64 = 10_000_000;

const MAX_MODEL_KEY_BYTES: usize = 64;
const DEFAULT_SCOPE_KEY: &str = "session";

/// Permit-listed normalized result of parsing one Mistral Vibe `post_llm`
/// hook payload.
///
/// The original payload and every unrelated field (session and conversation
/// identifiers, prompts, transcripts, response text) are discarded before this
/// value crosses the provider boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeObservation {
    pub snapshots: Vec<QuotaSnapshotInput>,
    pub rejected_usages: usize,
}

/// Sanitized Mistral Vibe parser failures.
///
/// Variants deliberately carry no deserializer messages or input-derived data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VibeParseError {
    #[error("Mistral Vibe hook input exceeds the size limit")]
    Oversized,
    #[error("Mistral Vibe hook input is not one valid JSON object")]
    InvalidDocument,
    #[error("Mistral Vibe hook input has an unexpected event")]
    UnexpectedEvent,
    #[error("Mistral Vibe context window token setting is out of range")]
    InvalidContextWindow,
}

#[derive(Deserialize)]
struct HookEnvelope {
    hook_event_name: Option<String>,
    model: Option<ModelField>,
    usage: Option<UsageField>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelField {
    Text(String),
    Invalid(IgnoredAny),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum UsageField {
    Tokens(UsageTokens),
    Invalid(IgnoredAny),
}

#[derive(Deserialize)]
struct UsageTokens {
    total_tokens: Option<u64>,
}

/// Parses one bounded Mistral Vibe hook payload.
///
/// Only `post_llm` events carry token usage. A missing or null `usage` member,
/// or one with no usable usage object, is an observation with no snapshots.
/// No field other than the event name, an optional model alias, and usage
/// token totals is read from the document.
pub fn parse_hook_payload(
    input: &[u8],
    observed_at: i64,
    context_window_tokens: u64,
) -> Result<VibeObservation, VibeParseError> {
    if !(MIN_CONTEXT_WINDOW_TOKENS..=MAX_CONTEXT_WINDOW_TOKENS).contains(&context_window_tokens) {
        return Err(VibeParseError::InvalidContextWindow);
    }
    if input.len() > MAX_STATUS_INPUT_BYTES {
        return Err(VibeParseError::Oversized);
    }
    let envelope: HookEnvelope =
        serde_json::from_slice(input).map_err(|_| VibeParseError::InvalidDocument)?;
    if envelope.hook_event_name.as_deref() != Some("post_llm") {
        return Err(VibeParseError::UnexpectedEvent);
    }

    let source_sequence = Uuid::new_v4();
    let mut observation = VibeObservation {
        snapshots: Vec::new(),
        rejected_usages: 0,
    };

    let Some(usage) = envelope.usage else {
        return Ok(observation);
    };
    let tokens = match usage {
        UsageField::Tokens(tokens) => tokens,
        UsageField::Invalid(_) => {
            observation.rejected_usages += 1;
            return Ok(observation);
        }
    };
    let Some(total_tokens) = tokens.total_tokens else {
        observation.rejected_usages += 1;
        return Ok(observation);
    };

    let Some(scope_key) = resolve_scope_key(envelope.model.as_ref()) else {
        observation.rejected_usages += 1;
        return Ok(observation);
    };

    if let Some(snapshot) = snapshot(
        scope_key,
        total_tokens,
        context_window_tokens,
        observed_at,
        source_sequence,
    ) {
        observation.snapshots.push(snapshot);
    } else {
        observation.rejected_usages += 1;
    }
    Ok(observation)
}

fn resolve_scope_key(model: Option<&ModelField>) -> Option<String> {
    match model {
        None => Some(DEFAULT_SCOPE_KEY.to_owned()),
        Some(ModelField::Text(model)) if is_valid_model_key(model) => Some(model.to_owned()),
        Some(&ModelField::Text(_)) | Some(&ModelField::Invalid(_)) => None,
    }
}

fn snapshot(
    scope_key: String,
    total_tokens: u64,
    context_window_tokens: u64,
    observed_at: i64,
    source_sequence: Uuid,
) -> Option<QuotaSnapshotInput> {
    // The percentage is clamped so a payload that claims more tokens than the
    // configured window holds still shows a filled bar instead of being
    // rejected as an invalid subscription window.
    let input = QuotaSnapshotInput {
        provider: Provider::MistralVibe,
        scope_key,
        window_kind: WindowKind::Other,
        limit_kind: LimitKind::Limited,
        window_duration_seconds: None,
        used_percent: ((total_tokens as f64 / context_window_tokens as f64) * 100.0)
            .clamp(0.0, 100.0),
        resets_at: None,
        observed_at,
        availability: Availability::Unknown,
        source_version: None,
        quality: Quality::Partial,
        source_sequence,
    };
    QuotaSnapshot::new(input.clone()).ok().map(|_| input)
}

fn is_valid_model_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    (1..=MAX_MODEL_KEY_BYTES).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[cfg(test)]
#[path = "vibe_tests.rs"]
mod tests;
