//! GPU acceptance for the native OpenAI embeddings endpoint on a live ModernBERT model.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;

use super::openai_rest::{E2E_TIMEOUT, get_endpoint, stop_serving_rest_server};
use crate::support::http::send_http_request;
use crate::support::serving_rest::launch_serving_rest_server_for_embedding_model;

const EMBEDDINGS_MODEL_LEAF_ID: &str = "nomicai-modernbert-embed-base-8bit";
const NATIVE_VECTOR_WIDTH: usize = 768;
const TRUNCATED_VECTOR_WIDTH: usize = 256;
const ROMEO_AND_JULIET_LINES: [&str; 2] = [
    "O Romeo, Romeo, wherefore art thou Romeo?",
    "Two households, both alike in dignity.",
];
const UNRELATED_FINANCE_LINE: &str =
    "The quarterly revenue report shows strong EBITDA growth and dividend guidance.";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads nomicai-modernbert-embed-base-8bit on GPU and exercises embeddings over public REST"]
async fn should_embed_romeo_and_juliet_lines_through_public_rest() {
    tokio::time::timeout(E2E_TIMEOUT, run_embeddings_gpu_journey())
        .await
        .expect("the embeddings GPU journey must finish within 115 seconds");
}

async fn run_embeddings_gpu_journey() {
    let selected_model = configured_embeddings_model();
    eprintln!(
        "[embeddings-rest] status=progress phase=launch model={}",
        selected_model.model_id
    );
    let maximum_input_tokens = match &selected_model.capabilities {
        astronomical_config::ModelCapabilities::Embeddings(embedding_capabilities) => {
            embedding_capabilities.max_input_tokens
        }
        other_capabilities => {
            panic!("expected embedding capability, got {other_capabilities:?}")
        }
    };
    let rest_server = launch_serving_rest_server_for_embedding_model(
        &selected_model.model_id,
        selected_model.model_directory.clone(),
        NATIVE_VECTOR_WIDTH as u32,
        maximum_input_tokens,
    )
    .await;
    let server_address = rest_server.server_address;

    eprintln!("[embeddings-rest 1/6] status=progress phase=models_capability");
    let models_response = get_endpoint(server_address, "/v1/models").await;
    assert_http_ok(&models_response);
    let models_document = http_json_body(&models_response);
    let advertised_model = advertised_model_document(&models_document, &selected_model.model_id);
    assert_eq!(
        advertised_model["supported_endpoints"],
        serde_json::json!(["/v1/embeddings"]),
        "embedding models must advertise only /v1/embeddings"
    );
    assert_eq!(
        advertised_model["output_modalities"],
        serde_json::json!(["embedding"])
    );
    eprintln!("[embeddings-rest 1/6] status=success phase=models_capability");

    eprintln!("[embeddings-rest 2/6] status=progress phase=single_float");
    let single_response = post_embeddings(
        server_address,
        embeddings_request_body(&selected_model.model_id, single_input(), None, false),
    )
    .await;
    let single_document = assert_embeddings_list(&single_response, 1, NATIVE_VECTOR_WIDTH, false);
    assert_unit_norm(&single_document, 0, NATIVE_VECTOR_WIDTH);
    assert_eq!(
        single_document["model"].as_str(),
        Some(selected_model.model_id.as_str()),
        "response model must echo the resolved discovered identity"
    );
    eprintln!("[embeddings-rest 2/6] status=success phase=single_float");

    eprintln!("[embeddings-rest 3/6] status=progress phase=array_order");
    let array_response = post_embeddings(
        server_address,
        embeddings_request_body(&selected_model.model_id, array_input(), None, false),
    )
    .await;
    let array_document = assert_embeddings_list(&array_response, 2, NATIVE_VECTOR_WIDTH, false);
    assert_vector_order(&array_document, 2, NATIVE_VECTOR_WIDTH);
    eprintln!("[embeddings-rest 3/6] status=success phase=array_order");

    eprintln!("[embeddings-rest 4/6] status=progress phase=base64_roundtrip");
    let base64_response = post_embeddings(
        server_address,
        embeddings_request_body(&selected_model.model_id, single_input(), None, true),
    )
    .await;
    let base64_document = assert_embeddings_list(&base64_response, 1, 0, true);
    assert_base64_decodes_to_float(&base64_document, NATIVE_VECTOR_WIDTH);
    eprintln!("[embeddings-rest 4/6] status=success phase=base64_roundtrip");

    eprintln!("[embeddings-rest 5/6] status=progress phase=truncated_dimensions");
    let truncated_response = post_embeddings(
        server_address,
        embeddings_request_body(
            &selected_model.model_id,
            single_input(),
            Some(TRUNCATED_VECTOR_WIDTH),
            false,
        ),
    )
    .await;
    let truncated_document =
        assert_embeddings_list(&truncated_response, 1, TRUNCATED_VECTOR_WIDTH, false);
    assert_unit_norm(&truncated_document, 0, TRUNCATED_VECTOR_WIDTH);
    eprintln!("[embeddings-rest 5/6] status=success phase=truncated_dimensions");

    eprintln!("[embeddings-rest 6/6] status=progress phase=semantic_similarity");
    let similarity_response = post_embeddings(
        server_address,
        embeddings_request_body(
            &selected_model.model_id,
            serde_json::json!([
                ROMEO_AND_JULIET_LINES[0],
                ROMEO_AND_JULIET_LINES[1],
                UNRELATED_FINANCE_LINE,
            ]),
            None,
            false,
        ),
    )
    .await;
    let similarity_document =
        assert_embeddings_list(&similarity_response, 3, NATIVE_VECTOR_WIDTH, false);
    let play_similarity = cosine_similarity(&similarity_document, 0, 1);
    let finance_similarity = cosine_similarity(&similarity_document, 0, 2);
    assert!(
        play_similarity > finance_similarity + 0.05,
        "both Romeo and Juliet lines must embed closer to each other than to an unrelated finance line: play={play_similarity} finance={finance_similarity}"
    );
    eprintln!(
        "[embeddings-rest 6/6] status=success phase=semantic_similarity play={play_similarity:.3} finance={finance_similarity:.3}"
    );

    stop_serving_rest_server(rest_server).await;
    eprintln!(
        "[embeddings-rest] status=success model={}",
        selected_model.model_id
    );
}

fn configured_embeddings_model() -> astronomical_config::DiscoveredModel {
    let discovered_models = crate::support::configured_discovered_models();
    let selected_model = discovered_models
        .into_iter()
        .find(|discovered_model| discovered_model.model_id == EMBEDDINGS_MODEL_LEAF_ID)
        .unwrap_or_else(|| {
            panic!(
                "Development discovery must include {EMBEDDINGS_MODEL_LEAF_ID} for embeddings acceptance"
            )
        });
    eprintln!(
        "[embeddings-rest] status=diagnostic phase=discovery capabilities={:?} family={:?}",
        selected_model.capabilities, selected_model.model_family
    );
    assert!(
        matches!(
            selected_model.capabilities,
            astronomical_config::ModelCapabilities::Embeddings(_)
        ),
        "{EMBEDDINGS_MODEL_LEAF_ID} must be discovered as an embedding model"
    );
    selected_model
}

fn embeddings_request_body(
    model_id: &str,
    input_json: Value,
    dimensions: Option<usize>,
    use_base64: bool,
) -> String {
    let mut request = serde_json::json!({
        "model": model_id,
        "input": input_json,
    });
    if use_base64 {
        request["encoding_format"] = serde_json::json!("base64");
    }
    if let Some(dimensions) = dimensions {
        request["dimensions"] = serde_json::json!(dimensions);
    }
    request.to_string()
}

fn single_input() -> Value {
    serde_json::json!(ROMEO_AND_JULIET_LINES[0])
}

fn array_input() -> Value {
    serde_json::json!(ROMEO_AND_JULIET_LINES)
}

async fn post_embeddings(server_address: std::net::SocketAddr, request_body: String) -> String {
    let request_text = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    );
    send_http_request(server_address, request_text).await
}

fn assert_embeddings_list(
    http_response: &str,
    expected_rows: usize,
    expected_width: usize,
    base64_encoded: bool,
) -> Value {
    assert_http_ok(http_response);
    let response_document = http_json_body(http_response);
    assert_eq!(response_document["object"], "list");
    let embedding_rows = response_document["data"]
        .as_array()
        .expect("embedding rows should be an array");
    assert_eq!(embedding_rows.len(), expected_rows, "one vector per input");
    for (row_index, embedding_row) in embedding_rows.iter().enumerate() {
        assert_eq!(embedding_row["object"], "embedding");
        assert_eq!(embedding_row["index"], row_index);
        if base64_encoded {
            assert!(
                embedding_row["embedding"].as_str().is_some(),
                "base64 rows must be strings"
            );
        } else if expected_width > 0 {
            let components = embedding_row["embedding"]
                .as_array()
                .expect("float rows must be arrays");
            assert_eq!(components.len(), expected_width);
            assert!(
                components.iter().all(|component| component.is_number()
                    && component.as_f64().unwrap_or(0.0).is_finite()),
                "float components must be finite numbers"
            );
        }
    }
    response_document
}

fn assert_unit_norm(response_document: &Value, row_index: usize, expected_width: usize) {
    let components = response_document["data"][row_index]["embedding"]
        .as_array()
        .expect("float rows must be arrays");
    assert_eq!(components.len(), expected_width);
    let squared_sum = components
        .iter()
        .map(|component| component.as_f64().unwrap_or(0.0) * component.as_f64().unwrap_or(0.0))
        .sum::<f64>();
    assert!(
        (squared_sum - 1.0).abs() < 1e-3,
        "embedding rows must be unit-normalized, got squared_sum={squared_sum}"
    );
}

fn assert_vector_order(response_document: &Value, expected_rows: usize, expected_width: usize) {
    assert_eq!(
        response_document["data"].as_array().expect("rows").len(),
        expected_rows
    );
    for embedding_row in response_document["data"].as_array().expect("rows") {
        assert_eq!(
            embedding_row["embedding"].as_array().expect("floats").len(),
            expected_width
        );
    }
}

fn assert_base64_decodes_to_float(response_document: &Value, expected_width: usize) {
    let encoded = response_document["data"][0]["embedding"]
        .as_str()
        .expect("base64 rows must be strings");
    let decoded_bytes = STANDARD
        .decode(encoded)
        .expect("base64 embedding rows must decode");
    assert_eq!(decoded_bytes.len(), expected_width * 4);
    let components = decoded_bytes
        .chunks_exact(4)
        .map(|component_bytes| {
            f32::from_le_bytes(component_bytes.try_into().expect("4-byte chunk"))
        })
        .collect::<Vec<_>>();
    let squared_sum = components
        .iter()
        .map(|component| f64::from(*component) * f64::from(*component))
        .sum::<f64>();
    assert!(
        (squared_sum - 1.0).abs() < 1e-3,
        "decoded base64 rows must match the float contract unit norm, got {squared_sum}"
    );
}

fn cosine_similarity(response_document: &Value, first_row: usize, second_row: usize) -> f64 {
    let first_components = float_components(response_document, first_row);
    let second_components = float_components(response_document, second_row);
    let dot_product: f64 = first_components
        .iter()
        .zip(second_components.iter())
        .map(|(first, second)| first * second)
        .sum();
    dot_product
}

fn float_components(response_document: &Value, row_index: usize) -> Vec<f64> {
    response_document["data"][row_index]["embedding"]
        .as_array()
        .expect("float rows must be arrays")
        .iter()
        .map(|component| component.as_f64().unwrap_or(0.0))
        .collect()
}

fn advertised_model_document<'models_document>(
    models_document: &'models_document Value,
    model_id: &str,
) -> &'models_document Value {
    models_document["data"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|advertised_model| advertised_model["id"].as_str() == Some(model_id))
        .unwrap_or_else(|| panic!("GET /v1/models must advertise {model_id}"))
}

fn assert_http_ok(http_response: &str) {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {http_response}"
    );
}

fn http_json_body(http_response: &str) -> Value {
    let response_body = http_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .trim_start_matches('\u{feff}');
    serde_json::from_str(response_body).unwrap_or_else(|json_error| {
        panic!("HTTP body should be JSON ({json_error}): {http_response}")
    })
}
