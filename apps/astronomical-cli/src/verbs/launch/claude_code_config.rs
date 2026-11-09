//! Builds the process-scoped environment that points one Claude Code session
//! at the local Anthropic-compatible endpoint.
//!
//! Claude Code reads its endpoint, credential, models, and context window
//! only from its environment and settings files, so this launch sets them in
//! the launched process alone and never writes `~/.claude`. Variable names
//! follow Claude Code's documented model configuration
//! (https://code.claude.com/docs/en/model-config and
//! https://code.claude.com/docs/en/env-vars).

use std::net::SocketAddr;

use super::chat_models::InstalledChatModel;

/// Where Claude Code sends Messages requests.
pub const ANTHROPIC_BASE_URL_VARIABLE: &str = "ANTHROPIC_BASE_URL";
/// Sent as a bearer header. The loopback daemon does not authenticate, so the
/// value is a fixed placeholder; `ANTHROPIC_API_KEY` is avoided because an
/// interactive session stops to ask the user to approve a custom key.
pub const ANTHROPIC_AUTH_TOKEN_VARIABLE: &str = "ANTHROPIC_AUTH_TOKEN";
pub const LOCAL_AUTH_TOKEN_PLACEHOLDER: &str = "astronomical-local";
/// The session's main model.
pub const ANTHROPIC_MODEL_VARIABLE: &str = "ANTHROPIC_MODEL";
/// The context window Claude Code assumes. It cannot recognize a local model
/// name, so without this it assumes 200K tokens and compacts at the wrong
/// boundary. The value applies on its own because local model identifiers
/// never start with `claude-` and carry no `[1m]` suffix.
pub const CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE: &str = "CLAUDE_CODE_MAX_CONTEXT_TOKENS";
/// Claude Code reaches beyond the main model through its family aliases
/// (`haiku` for background work, any alias a subagent names) and a subagent
/// default. Each is pinned to the local model, so no request ever names an
/// Anthropic model the local server cannot serve.
pub const LOCAL_MODEL_ROUTING_VARIABLES: [&str; 5] = [
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
];

/// The environment for one Claude Code session on `chat_model`. A model whose
/// window the daemon does not advertise gets no window variable, leaving
/// Claude Code's own default in place rather than inventing one.
pub fn claude_code_environment(
    bind_address: SocketAddr,
    chat_model: &InstalledChatModel,
) -> Vec<(String, String)> {
    let mut session_environment = vec![
        (
            ANTHROPIC_BASE_URL_VARIABLE.to_owned(),
            format!("http://{bind_address}"),
        ),
        (
            ANTHROPIC_AUTH_TOKEN_VARIABLE.to_owned(),
            LOCAL_AUTH_TOKEN_PLACEHOLDER.to_owned(),
        ),
        (
            ANTHROPIC_MODEL_VARIABLE.to_owned(),
            chat_model.model_id.clone(),
        ),
    ];
    for alias_variable in LOCAL_MODEL_ROUTING_VARIABLES {
        session_environment.push((alias_variable.to_owned(), chat_model.model_id.clone()));
    }
    if let Some(context_window_tokens) = chat_model.context_window_tokens {
        session_environment.push((
            CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE.to_owned(),
            context_window_tokens.to_string(),
        ));
    }
    session_environment
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_AUTH_TOKEN_VARIABLE, ANTHROPIC_BASE_URL_VARIABLE, ANTHROPIC_MODEL_VARIABLE,
        CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE, LOCAL_MODEL_ROUTING_VARIABLES,
        claude_code_environment,
    };
    use crate::verbs::launch::chat_models::InstalledChatModel;

    fn environment_value<'a>(
        session_environment: &'a [(String, String)],
        variable_name: &str,
    ) -> Option<&'a str> {
        for (environment_name, environment_value) in session_environment {
            if environment_name == variable_name {
                return Some(environment_value);
            }
        }
        None
    }

    fn local_model(context_window_tokens: Option<u32>) -> InstalledChatModel {
        InstalledChatModel {
            model_id: "local-sonnet-free-model".to_owned(),
            context_window_tokens,
        }
    }

    #[test]
    fn should_point_the_session_at_the_loopback_instance_with_its_model() {
        let session_environment = claude_code_environment(
            "127.0.0.1:6733".parse().expect("loopback address"),
            &local_model(Some(262_144)),
        );

        assert_eq!(
            environment_value(&session_environment, ANTHROPIC_BASE_URL_VARIABLE),
            Some("http://127.0.0.1:6733")
        );
        assert_eq!(
            environment_value(&session_environment, ANTHROPIC_MODEL_VARIABLE),
            Some("local-sonnet-free-model")
        );
        assert!(environment_value(&session_environment, ANTHROPIC_AUTH_TOKEN_VARIABLE).is_some());
        assert!(environment_value(&session_environment, "ANTHROPIC_API_KEY").is_none());
    }

    #[test]
    fn should_pin_every_model_alias_to_the_local_model() {
        let session_environment = claude_code_environment(
            "127.0.0.1:6733".parse().expect("loopback address"),
            &local_model(Some(262_144)),
        );

        for alias_variable in LOCAL_MODEL_ROUTING_VARIABLES {
            assert_eq!(
                environment_value(&session_environment, alias_variable),
                Some("local-sonnet-free-model"),
                "{alias_variable} must name the local model"
            );
        }
    }

    #[test]
    fn should_declare_the_advertised_context_window() {
        let session_environment = claude_code_environment(
            "127.0.0.1:6733".parse().expect("loopback address"),
            &local_model(Some(32_768)),
        );

        assert_eq!(
            environment_value(
                &session_environment,
                CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE
            ),
            Some("32768")
        );
    }

    #[test]
    fn should_leave_the_window_unset_when_the_daemon_advertises_none() {
        let session_environment = claude_code_environment(
            "127.0.0.1:6733".parse().expect("loopback address"),
            &local_model(None),
        );

        assert!(
            environment_value(
                &session_environment,
                CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE
            )
            .is_none()
        );
    }
}
