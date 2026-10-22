//! Builds process-scoped Codex CLI configuration.
//!
//! Generates a throwaway `CODEX_HOME` holding one `config.toml`, following
//! Codex's documented `CODEX_HOME` environment variable and custom
//! `model_providers` table (https://developers.openai.com/codex/config-reference).
//! The user's `~/.codex` is never read or written: the generated directory is
//! the only configuration the launched session sees. `wire_api = "responses"`
//! because the Codex config reference declares `responses` the only supported
//! wire protocol, and the Astronomical daemon serves `/v1/responses` alongside
//! `/v1/chat/completions`. No `env_key` is set: the endpoint is loopback-only
//! and performs no authentication.

use std::{io, net::SocketAddr, path::Path};

use tempfile::TempDir;

use crate::models::LibraryChatModel;

pub const CODEX_HOME_VARIABLE: &str = "CODEX_HOME";
const CODEX_PROVIDER_ID: &str = "astronomical";

/// Creates the per-session `CODEX_HOME` with the generated `config.toml` and
/// returns its path as the environment value for the launched process.
///
/// The directory is deliberately disowned rather than deleted on drop: the
/// Codex session keeps writing into it (history, logs) for as long as it
/// runs, and this CLI execs into the harness so it has no exit hook left to
/// clean up with. The operating system's temporary-directory reaping owns
/// the file's eventual removal. The disown happens only after the config
/// write and the environment-value validation succeed, so a failed launch
/// never leaks an ownerless directory.
pub fn codex_process_home(
    bind_address: SocketAddr,
    chat_model: &LibraryChatModel,
) -> io::Result<String> {
    let codex_home = TempDir::with_prefix("astronomical-codex-home-")?;
    write_codex_config(codex_home.path(), bind_address, chat_model)?;
    // Environment values must be UTF-8 text; a non-UTF-8 temporary path
    // cannot name the generated home, so launch fails closed instead of
    // pointing Codex at a path that does not exist. Validating before the
    // disown keeps the still-owned directory's drop cleanup on the error
    // path.
    let validated_codex_home_path = codex_home
        .path()
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "CODEX_HOME path is not UTF-8"))?
        .to_owned();
    let disowned_codex_home = codex_home.keep();
    // The disowned path is byte-identical to the validated one, so this
    // conversion cannot fail; fall back to the validated text regardless.
    Ok(disowned_codex_home
        .into_os_string()
        .into_string()
        .unwrap_or(validated_codex_home_path))
}

fn write_codex_config(
    codex_home_path: &Path,
    bind_address: SocketAddr,
    chat_model: &LibraryChatModel,
) -> io::Result<()> {
    std::fs::write(
        codex_home_path.join("config.toml"),
        codex_config_toml(bind_address, chat_model),
    )
}

pub fn codex_config_toml(bind_address: SocketAddr, chat_model: &LibraryChatModel) -> String {
    let openai_compatible_base_url = format!("http://{bind_address}/v1");
    format!(
        "model = {}\n\
         model_provider = \"{CODEX_PROVIDER_ID}\"\n\
         \n\
         [model_providers.{CODEX_PROVIDER_ID}]\n\
         name = \"Astronomical\"\n\
         base_url = {}\n\
         wire_api = \"responses\"\n",
        toml_basic_string(&chat_model.model_id),
        toml_basic_string(&openai_compatible_base_url),
    )
}

/// Quotes a TOML basic string, escaping everything TOML forbids raw inside
/// one: the two characters TOML requires a backslash for, and the control
/// characters TOML only accepts as escapes. Model ids come from the daemon's
/// `/v1/models` document rather than a fixed catalog, so the escaping keeps
/// a hostile or accidental id from producing an unparseable config.
fn toml_basic_string(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for character in value.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\u{8}' => quoted.push_str("\\b"),
            '\u{9}' => quoted.push_str("\\t"),
            '\u{a}' => quoted.push_str("\\n"),
            '\u{c}' => quoted.push_str("\\f"),
            '\u{d}' => quoted.push_str("\\r"),
            control if control <= '\u{1f}' || control == '\u{7f}' => {
                quoted.push_str(&format!("\\u{:04X}", control as u32));
            }
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod codex_config_tests {
    use super::toml_basic_string;

    #[test]
    fn should_quote_a_plain_value_unchanged() {
        assert_eq!(
            toml_basic_string("library-chat-model"),
            "\"library-chat-model\""
        );
    }

    #[test]
    fn should_escape_the_characters_toml_requires_escapes_for() {
        assert_eq!(
            toml_basic_string("say \"hi\"\\done\r\nstop"),
            "\"say \\\"hi\\\"\\\\done\\r\\nstop\""
        );
    }

    #[test]
    fn should_escape_control_characters_as_unicode_escapes() {
        assert_eq!(toml_basic_string("a\u{1}b\u{7f}"), "\"a\\u0001b\\u007F\"");
    }
}
