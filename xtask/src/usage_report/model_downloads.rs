//! Hugging Face download counts for every model in the download catalog.
//! Hugging Face counts requests for a model's files over a rolling 30 days,
//! not people, and the maintainer's own downloads count too, so the figures
//! show which models draw interest rather than how many people use them.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloads {
    pub repository_id: String,
    pub downloads_last_30_days: Option<u64>,
    pub downloads_all_time: Option<u64>,
    pub like_count: Option<u64>,
}

/// The `huggingface_id` of every catalog entry, in catalog order, each once.
pub fn catalog_repository_ids(catalog_json_text: &str) -> Result<Vec<String>, String> {
    let catalog: Value = serde_json::from_str(catalog_json_text)
        .map_err(|parse_error| format!("download catalog is not valid JSON: {parse_error}"))?;
    let catalog_entries = catalog["entries"]
        .as_array()
        .ok_or("download catalog has no entries list")?;
    let mut repository_ids: Vec<String> = Vec::with_capacity(catalog_entries.len());
    for catalog_entry in catalog_entries {
        let Some(repository_id) = catalog_entry["huggingface_id"].as_str() else {
            continue;
        };
        if !repository_ids
            .iter()
            .any(|known_id| known_id == repository_id)
        {
            repository_ids.push(repository_id.to_owned());
        }
    }
    Ok(repository_ids)
}

/// Reads one model's counts from the Hugging Face models API response. A
/// response that is not JSON, or lacks a field, leaves that count unknown
/// rather than reporting zero, because zero would read as "nobody".
pub fn parse_model_downloads(repository_id: &str, model_response_text: &str) -> ModelDownloads {
    let model_response: Value = serde_json::from_str(model_response_text).unwrap_or(Value::Null);
    ModelDownloads {
        repository_id: repository_id.to_owned(),
        downloads_last_30_days: model_response["downloads"].as_u64(),
        downloads_all_time: model_response["downloadsAllTime"].as_u64(),
        like_count: model_response["likes"].as_u64(),
    }
}

/// Most downloaded in the last 30 days first; unknown counts sort last, and
/// ties keep catalog order.
pub fn sort_by_recent_downloads(model_downloads: &mut [ModelDownloads]) {
    model_downloads.sort_by_key(|model| std::cmp::Reverse(model.downloads_last_30_days));
}
