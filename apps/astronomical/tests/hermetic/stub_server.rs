//! Loopback Astronomical stand-in for launch journeys. Serves `/v1/status` and
//! `/v1/models` on an ephemeral port so tests never touch a real instance.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    thread,
};

pub struct StubAstronomical {
    pub bind_address: SocketAddr,
}

impl StubAstronomical {
    pub fn spawn(status_body: &str, models_body: &str) -> Self {
        Self::spawn_with_models_status(status_body, models_body, "200 OK")
    }

    pub fn spawn_with_models_status(
        status_body: &str,
        models_body: &str,
        models_status_line: &'static str,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("ephemeral loopback listener");
        let bind_address = listener.local_addr().expect("listener address");
        let status_body = status_body.to_owned();
        let models_body = models_body.to_owned();
        thread::spawn(move || {
            for incoming in listener.incoming() {
                let Ok(tcp_stream) = incoming else {
                    continue;
                };
                let _ = respond(tcp_stream, &status_body, &models_body, models_status_line);
            }
        });
        Self { bind_address }
    }
}

fn respond(
    mut tcp_stream: TcpStream,
    status_body: &str,
    models_body: &str,
    models_status_line: &str,
) -> std::io::Result<()> {
    let mut request_bytes = [0_u8; 4096];
    let bytes_read = tcp_stream.read(&mut request_bytes)?;
    let request_text = String::from_utf8_lossy(&request_bytes[..bytes_read]);
    let request_path = request_text.split_whitespace().nth(1).unwrap_or("/");
    let response_body = if request_path.starts_with("/v1/status") {
        status_body
    } else if request_path.starts_with("/v1/models") {
        models_body
    } else {
        "{\"error\":\"unknown\"}"
    };
    let status_code = if request_path.starts_with("/v1/status") {
        "200 OK"
    } else if request_path.starts_with("/v1/models") {
        models_status_line
    } else {
        "404 Not Found"
    };
    let response = format!(
        "HTTP/1.1 {status_code}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
        response_body.len()
    );
    tcp_stream.write_all(response.as_bytes())
}

pub fn ready_status_json() -> &'static str {
    r#"{"application":{"channel":"stable","channel_display_name":"Stable"},"status":"ready","activity":"idle"}"#
}

pub fn chat_models_json(model_entries: &[(&str, u32)]) -> String {
    let models = model_entries
        .iter()
        .map(|(model_id, context_window)| {
            format!(
                r#"{{"id":"{model_id}","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/chat/completions"],"output_modalities":["text"],"context_window":{context_window}}}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"object":"list","data":[{models}]}}"#)
}

pub fn image_model_json(model_id: &str) -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"{model_id}","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/images/generations"],"output_modalities":["image"]}}]}}"#
    )
}

pub fn embedding_model_json(model_id: &str) -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"{model_id}","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/embeddings"],"output_modalities":["text"]}}]}}"#
    )
}

pub fn mixed_library_json() -> String {
    r#"{"object":"list","data":[
        {"id":"library-image-model","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/images/generations"],"output_modalities":["image"]},
        {"id":"library-chat-model","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/chat/completions"],"output_modalities":["text"],"context_window":131072},
        {"id":"library-embedding-model","object":"model","owned_by":"astronomical","supported_endpoints":["/v1/embeddings"],"output_modalities":["text"]}
    ]}"#
    .to_owned()
}
