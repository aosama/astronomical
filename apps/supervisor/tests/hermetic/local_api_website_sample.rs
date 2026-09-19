//! The published local API sample must describe the interface the app actually serves.
//!
//! The page is the artifact a visitor copies into a coding agent, so a value that
//! drifts from the product is worse than no value at all: a path the app does not
//! route fails somewhere far from the page that published it. The ports the page
//! names are a configuration value, so that half of the contract lives with the
//! configuration crate that owns them.

use std::{collections::BTreeSet, net::SocketAddr, path::PathBuf, process::Command};

use astronomical_ipc_protocol::ChatGenerationCompletionReason;
use astronomical_supervisor::{ChatGenerationStreamEvent, build_application};
use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::common::{MODEL_ID, ScriptedExecutor};

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
    page.match_indices(ENDPOINT_PATH_PREFIX)
        .map(|(path_start, path_prefix)| {
            let after_prefix = &page[path_start + path_prefix.len()..];
            let path_segment: String = after_prefix
                .chars()
                .take_while(|character| character.is_ascii_lowercase() || *character == '/')
                .collect();
            (path_prefix, path_segment)
        })
        .filter(|(_, path_segment)| !path_segment.is_empty())
        .map(|(path_prefix, path_segment)| format!("{path_prefix}{path_segment}"))
        .collect()
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

        let response = build_application(ScriptedExecutor::ready(Vec::new()))
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
    let published_command = published_snippet_blocks(&local_api_page())
        .into_iter()
        .find(|block| block.contains("curl ") && block.contains("/chat/completions"))
        .expect("the local API page should publish a chat completion command");

    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
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

    // A reader pastes their own identifier into the command and runs it against
    // the port the page names, so the test performs exactly those two substitutions.
    let published_origin = first_loopback_origin(&published_command)
        .expect("the published command should name a loopback address");
    let live_command = published_command
        .replace(&published_origin, &bound_origin)
        .replace(MODEL_IDENTIFIER_PLACEHOLDER, MODEL_ID);

    let command_output = tokio::task::spawn_blocking(move || {
        Command::new("bash").arg("-c").arg(&live_command).output()
    })
    .await
    .expect("the published command should run")
    .expect("bash should execute the published command");

    assert!(
        command_output.status.success(),
        "the published command should succeed, stderr was: {}",
        String::from_utf8_lossy(&command_output.stderr)
    );
    let completion: serde_json::Value = serde_json::from_slice(&command_output.stdout)
        .unwrap_or_else(|source| {
            panic!(
                "the published command should print a JSON completion, it printed {:?} ({source})",
                String::from_utf8_lossy(&command_output.stdout)
            )
        });
    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["choices"][0]["message"]["content"], "done");
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

        let (provider, is_opencode) = if document.get("provider").is_some() {
            (&document["provider"]["astronomical"], true)
        } else {
            (
                document
                    .get("providers")
                    .and_then(|providers| providers.get("astronomical"))
                    .unwrap_or(&serde_json::Value::Null),
                false,
            )
        };
        if provider.as_object().is_none() {
            continue;
        }
        provider_configurations += 1;

        let opencode_base_url = if is_opencode {
            provider["options"]["baseURL"].as_str()
        } else {
            provider["baseUrl"].as_str()
        };
        let published_base_url =
            opencode_base_url.expect("a published provider must name a base URL");
        assert!(
            published_base_url.starts_with("http://127.0.0.1:"),
            "the published base URL must stay a loopback address"
        );
        assert!(
            published_base_url.ends_with("/v1"),
            "the published base URL must carry the versioned path prefix the app serves"
        );

        let published_credential = if is_opencode {
            provider["options"]["apiKey"].as_str()
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

        if is_opencode {
            assert_eq!(provider["npm"], "@ai-sdk/openai-compatible");
            assert_eq!(
                provider["models"][MODEL_ID],
                serde_json::json!({"name": MODEL_ID}),
                "the published model entry must be the identifier a reader pastes"
            );
        } else {
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
    }

    assert_eq!(
        provider_configurations, 2,
        "the page should publish one opencode and one Pi provider configuration"
    );
}
