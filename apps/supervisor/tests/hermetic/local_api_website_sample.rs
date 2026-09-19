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
