//! The published local API sample must describe the interface the app actually serves.
//!
//! The page is the artifact a visitor copies into a coding agent, so a value that
//! drifts from the product is worse than no value at all: a path the app does not
//! route fails somewhere far from the page that published it. The ports the page
//! names are a configuration value, so that half of the contract lives with the
//! configuration crate that owns them.

use std::{collections::BTreeSet, net::SocketAddr, path::PathBuf};

use astronomical_daemon::ChatGenerationStreamEvent;
use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use ipc_protocol::ChatGenerationCompletionReason;
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::support::{MODEL_ID, ScriptedExecutor};

const LOCAL_API_PAGE_RELATIVE_PATH: &str = "../../site/local-api.html";
const LOOPBACK_ORIGIN_PREFIX: &str = "http://127.0.0.1:";
const ENDPOINT_PATH_PREFIX: &str = "/v1/";
const MODEL_IDENTIFIER_PLACEHOLDER: &str = "<identifier from /v1/models>";

fn local_api_page() -> String {
    let page_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(LOCAL_API_PAGE_RELATIVE_PATH);
    std::fs::read_to_string(&page_path).expect("the local API page should be readable")
}

/// Every versioned endpoint path the page names, wherever it names it: the
/// documented list, a command, or guidance about an identifier.
fn published_endpoint_paths(page: &str) -> BTreeSet<String> {
    endpoint_paths_after_prefix(page, ENDPOINT_PATH_PREFIX)
}

/// Pure: every distinct path in `published_text` that starts with
/// `path_prefix` and continues with at least one path character.
fn endpoint_paths_after_prefix(published_text: &str, path_prefix: &str) -> BTreeSet<String> {
    let mut endpoint_paths = BTreeSet::new();
    for (path_start, _) in published_text.match_indices(path_prefix) {
        let path_segment = leading_path_segment(&published_text[path_start + path_prefix.len()..]);
        if !path_segment.is_empty() {
            endpoint_paths.insert(format!("{path_prefix}{path_segment}"));
        }
    }
    endpoint_paths
}

/// Pure: the run of endpoint path characters `text` starts with.
fn leading_path_segment(text: &str) -> String {
    let mut path_segment = String::new();
    for path_character in text.chars() {
        if !is_endpoint_path_character(path_character) {
            break;
        }
        path_segment.push(path_character);
    }
    path_segment
}

/// Lowercase letters, `/`, and `_`: every character the served paths use,
/// including `/v1/messages/count_tokens`.
fn is_endpoint_path_character(path_character: char) -> bool {
    path_character.is_ascii_lowercase() || path_character == '/' || path_character == '_'
}

fn first_loopback_origin(published_text: &str) -> Option<String> {
    let origin_start = published_text.find(LOOPBACK_ORIGIN_PREFIX)?;
    let after_origin = &published_text[origin_start + LOOPBACK_ORIGIN_PREFIX.len()..];
    let port_digit_count = after_origin
        .chars()
        .take_while(char::is_ascii_digit)
        .count();
    if port_digit_count == 0 {
        return None;
    }
    Some(format!(
        "{LOOPBACK_ORIGIN_PREFIX}{}",
        &after_origin[..port_digit_count]
    ))
}

/// The page's copyable blocks, HTML-unescaped: a visitor copies the decoded text,
/// not the entity references that carry it.
fn published_snippet_blocks(page: &str) -> Vec<String> {
    page.split("<pre class=\"doc-snippet\"")
        .skip(1)
        .filter_map(|block| block.split("</pre>").next())
        .filter_map(|block| {
            let code_start = block.find("<code>")? + "<code>".len();
            let code_end = block[code_start..].find("</code>")? + code_start;
            Some(unescape_html(&block[code_start..code_end]))
        })
        .collect()
}

fn unescape_html(encoded_text: &str) -> String {
    encoded_text
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

#[tokio::test]
async fn should_serve_every_endpoint_path_the_local_api_page_documents() {
    let documented_paths = published_endpoint_paths(&local_api_page());
    assert!(
        !documented_paths.is_empty(),
        "the local API page should document at least one endpoint path"
    );

    for documented_path in documented_paths {
        let request = if documented_path == "/v1/models" {
            Request::builder().uri(&documented_path).body(Body::empty())
        } else {
            Request::builder()
                .method("POST")
                .uri(&documented_path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(r#"{{"model":"{MODEL_ID}"}}"#)))
        }
        .expect("a documented request should be a valid request");

        let response = astronomical_daemon::build_application(ScriptedExecutor::ready(Vec::new()))
            .oneshot(request)
            .await
            .expect("the application should answer a documented path");

        assert_ne!(
            response.status(),
            StatusCode::NOT_FOUND,
            "the local API page documents {documented_path}, which the application does not serve"
        );
    }
}

#[tokio::test]
async fn should_answer_the_verification_command_the_local_api_page_publishes() {
    let published_command = published_command_for_path("/chat/completions");

    let completion = run_published_command(&published_command).await;

    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["choices"][0]["message"]["content"], "done");
}

#[tokio::test]
async fn should_answer_the_messages_verification_command_the_local_api_page_publishes() {
    let published_command = published_command_for_path("/v1/messages \\");

    let message = run_published_command(&published_command).await;

    assert_eq!(message["type"], "message");
    assert_eq!(message["content"][0]["text"], "done");
    assert_eq!(message["stop_reason"], "end_turn");
}

/// The first published `curl` command that targets `path_fragment`.
fn published_command_for_path(path_fragment: &str) -> String {
    for published_block in published_snippet_blocks(&local_api_page()) {
        if published_block.contains("curl ") && published_block.contains(path_fragment) {
            return published_block;
        }
    }
    panic!("the local API page should publish a curl command for {path_fragment}");
}

/// Runs a published command the way a reader does: their own identifier
/// pasted in, against the port the page names (here, a scripted app bound to
/// a free loopback port), and returns the JSON it prints.
async fn run_published_command(published_command: &str) -> serde_json::Value {
    let bound_origin = serve_scripted_application().await;
    let published_origin = first_loopback_origin(published_command)
        .expect("the published command should name a loopback address");
    let live_command = published_command
        .replace(&published_origin, &bound_origin)
        .replace(MODEL_IDENTIFIER_PLACEHOLDER, MODEL_ID);
    let command_output = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(&live_command)
        .output()
        .await
        .expect("bash should execute the published command");
    assert!(
        command_output.status.success(),
        "the published command should succeed, stderr was: {}",
        String::from_utf8_lossy(&command_output.stderr)
    );
    match serde_json::from_slice(&command_output.stdout) {
        Ok(printed_document) => printed_document,
        Err(json_error) => panic!(
            "the published command should print JSON, it printed {:?} ({json_error})",
            String::from_utf8_lossy(&command_output.stdout)
        ),
    }
}

/// A scripted app that answers one generation with "done", served on a free
/// loopback port; returns its origin.
async fn serve_scripted_application() -> String {
    let application = astronomical_daemon::build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("the test should bind a loopback listener");
    let bound_origin = format!(
        "http://{}",
        listener
            .local_addr()
            .expect("the listener should report its address")
    );
    tokio::spawn(async move {
        let _ = axum::serve(listener, application).await;
    });
    bound_origin
}

#[test]
fn should_publish_provider_configuration_that_parses_once_the_identifier_is_pasted() {
    let page = local_api_page();
    let mut provider_configurations = 0;
    for snippet_block in published_snippet_blocks(&page) {
        if !snippet_block.trim_start().starts_with('{') {
            continue;
        }
        let pasted_block = snippet_block.replace(MODEL_IDENTIFIER_PLACEHOLDER, MODEL_ID);
        let document: serde_json::Value = match serde_json::from_str(&pasted_block) {
            Ok(document) => document,
            Err(source) => panic!(
                "a published configuration must be valid JSON once the identifier is pasted: {source}"
            ),
        };

        assert!(
            document.get("provider").is_none(),
            "OpenCode 2 reads providers, package and settings; the earlier provider shape is deprecated"
        );
        let provider = document
            .get("providers")
            .and_then(|providers| providers.get("astronomical"))
            .unwrap_or(&serde_json::Value::Null);
        if provider.as_object().is_none() {
            continue;
        }
        // OpenCode keeps a provider's endpoint and credential under `settings`;
        // Pi keeps them on the provider itself.
        let is_opencode = provider.get("settings").is_some();
        provider_configurations += 1;

        assert_provider_names_loopback_endpoint_and_placeholder_credential(provider, is_opencode);

        if is_opencode {
            assert_opencode_provider_package_and_model(provider);
        } else {
            assert_pi_provider_api_and_model(provider);
        }
    }

    assert_eq!(
        provider_configurations, 2,
        "the page should publish one opencode and one Pi provider configuration"
    );
}

fn assert_provider_names_loopback_endpoint_and_placeholder_credential(
    provider: &serde_json::Value,
    is_opencode: bool,
) {
    let published_base_url_text = if is_opencode {
        provider["settings"]["baseURL"].as_str()
    } else {
        provider["baseUrl"].as_str()
    };
    let published_base_url =
        published_base_url_text.expect("a published provider must name a base URL");
    assert!(
        published_base_url.starts_with("http://127.0.0.1:"),
        "the published base URL must stay a loopback address"
    );
    assert!(
        published_base_url.ends_with("/v1"),
        "the published base URL must carry the versioned path prefix the app serves"
    );

    let published_credential = if is_opencode {
        provider["settings"]["apiKey"].as_str()
    } else {
        provider["apiKey"].as_str()
    }
    .expect("a published provider must carry the credential field a client requires");
    assert!(
        !published_credential.is_empty(),
        "agents refuse a provider with an empty credential field, so the recipe must carry the placeholder"
    );
    assert!(
        !published_credential.starts_with("sk-"),
        "the published credential must never read as a secret"
    );
}

fn assert_opencode_provider_package_and_model(provider: &serde_json::Value) {
    assert_eq!(
        provider["package"],
        "@opencode/ai/providers/openai-compatible"
    );
    assert_eq!(
        provider["models"][MODEL_ID],
        serde_json::json!({"name": MODEL_ID}),
        "the published model entry must be the identifier a reader pastes"
    );
}

fn assert_pi_provider_api_and_model(provider: &serde_json::Value) {
    assert_eq!(provider["api"], "openai-completions");
    for suppressed_option in [
        "supportsStore",
        "supportsStrictMode",
        "supportsDeveloperRole",
    ] {
        assert_eq!(
            provider["compat"][suppressed_option], false,
            "the Pi recipe must suppress {suppressed_option}, which this endpoint answers with a JSON error"
        );
    }
    assert_eq!(
        provider["models"][0],
        serde_json::json!({"id": MODEL_ID, "name": MODEL_ID}),
        "the published model entry must be the identifier a reader pastes"
    );
}

const OPENCODE_PLUGIN_MANIFEST_RELATIVE_PATH: &str = "../../integrations/opencode/package.json";
const OPENCODE_PLUGIN_INSTALL_COMMAND: &str = "opencode plugin add ";

/// Pure: every package name the published text tells a reader to install with
/// `opencode plugin add`, in page order.
fn opencode_plugin_install_names(published_text: &str) -> Vec<String> {
    let mut install_names = Vec::new();
    for (command_start, _) in published_text.match_indices(OPENCODE_PLUGIN_INSTALL_COMMAND) {
        let after_command =
            &published_text[command_start + OPENCODE_PLUGIN_INSTALL_COMMAND.len()..];
        let mut install_name = String::new();
        for name_character in after_command.chars() {
            if !(name_character.is_ascii_alphanumeric() || "-_@/.".contains(name_character)) {
                break;
            }
            install_name.push(name_character);
        }
        // A name ending a sentence is followed by its full stop, which is not
        // part of an npm name.
        install_names.push(install_name.trim_end_matches('.').to_owned());
    }
    install_names
}

/// Every published place that tells a reader to install the plugin: the
/// website pages, Observatory's Connect page, and the READMEs and docs a
/// reader lands on from GitHub. The home page only mentions the plugin and
/// links to the local API page, so it carries no install line to check.
const OPENCODE_PLUGIN_INSTALL_PUBLICATIONS: [&str; 6] = [
    "../../site/local-api.html",
    "../../site/support.html",
    "console/index.html",
    "../../README.md",
    "../../docs/how-it-works.md",
    "../../integrations/opencode/README.md",
];

/// The page tells visitors which npm package to install; a name that drifts
/// from the plugin's own manifest would install nothing, or someone else's
/// package.
#[test]
fn should_publish_the_opencode_plugin_install_name_the_plugin_manifest_carries() {
    let manifest_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(OPENCODE_PLUGIN_MANIFEST_RELATIVE_PATH);
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("the plugin manifest should be readable"),
    )
    .expect("the plugin manifest should be JSON");
    let package_name = manifest["name"]
        .as_str()
        .expect("the plugin manifest should name its package");

    for publication_relative_path in OPENCODE_PLUGIN_INSTALL_PUBLICATIONS {
        let publication_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(publication_relative_path);
        let publication_text =
            std::fs::read_to_string(&publication_path).unwrap_or_else(|failure| {
                panic!("{publication_relative_path} should be readable: {failure}")
            });
        let install_names = opencode_plugin_install_names(&publication_text);
        assert!(
            !install_names.is_empty(),
            "{publication_relative_path} should tell OpenCode users how to install the plugin"
        );
        for install_name in install_names {
            assert_eq!(
                install_name, package_name,
                "{publication_relative_path} names a package the plugin is not published as"
            );
        }
    }
}

#[test]
fn should_read_the_package_name_after_the_install_command() {
    assert_eq!(
        opencode_plugin_install_names("run opencode plugin add opencode-astronomical</code> now"),
        vec!["opencode-astronomical".to_owned()]
    );
    assert_eq!(
        opencode_plugin_install_names("opencode plugin add @scope/name@1.2.0\n"),
        vec!["@scope/name@1.2.0".to_owned()]
    );
    assert_eq!(
        opencode_plugin_install_names("then run opencode plugin add opencode-astronomical."),
        vec!["opencode-astronomical".to_owned()],
        "a full stop ending the sentence is not part of the name"
    );
    assert!(opencode_plugin_install_names("no install line here").is_empty());
}
