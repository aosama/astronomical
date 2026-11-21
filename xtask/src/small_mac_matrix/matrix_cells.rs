//! The matrix's plan and results, with no network: which Macs are simulated,
//! what one cell's outcome was, and the table that reports them.

/// The Development instance's local address; Stable is never the default.
pub const DEVELOPMENT_BASE_URL: &str = "http://127.0.0.1:6733";

/// The Mac sizes simulated when none are named: the sizes sold below 48 GB.
pub const DEFAULT_MAC_MEMORY_GB: [u64; 4] = [8, 16, 24, 32];

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixOptions {
    pub base_url: String,
    pub model_ids: Vec<String>,
    pub mac_memory_gb: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CellOutcome {
    Replied {
        seconds_to_reply: f64,
    },
    /// The engine refused the memory limit itself, for example below a floor.
    LimitRefused,
    /// The reply failed; the engine's error code says how.
    Failed {
        error_code: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatrixCell {
    pub model_id: String,
    pub mac_memory_gb: u64,
    pub memory_limit_gb: u64,
    pub outcome: CellOutcome,
}

/// The memory limit a Mac of `mac_memory_gb` is simulated with, in whole GB
/// of 2^30 bytes. macOS lets the graphics chip use about two thirds of a
/// small Mac's memory; this is that share rounded down, an approximation of
/// the limit such a Mac reports, not a measurement of one.
pub fn simulated_memory_limit_gb(mac_memory_gb: u64) -> u64 {
    mac_memory_gb * 2 / 3
}

pub fn parse_matrix_options(command_arguments: &[String]) -> Result<MatrixOptions, String> {
    let mut matrix_options = MatrixOptions {
        base_url: DEVELOPMENT_BASE_URL.to_owned(),
        model_ids: Vec::new(),
        mac_memory_gb: Vec::new(),
    };
    let mut remaining_arguments = command_arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        match argument.as_str() {
            "--base-url" => {
                matrix_options.base_url = remaining_arguments
                    .next()
                    .ok_or("--base-url requires a URL")?
                    .clone();
            }
            "--model" => matrix_options.model_ids.push(
                remaining_arguments
                    .next()
                    .ok_or("--model requires a model id")?
                    .clone(),
            ),
            "--mac-gb" => {
                let mac_memory_text = remaining_arguments
                    .next()
                    .ok_or("--mac-gb requires a number")?;
                let mac_memory_gb: u64 = mac_memory_text.parse().map_err(|_| {
                    format!("--mac-gb needs a whole number of GB, got {mac_memory_text}")
                })?;
                if mac_memory_gb < 8 {
                    return Err(
                        "--mac-gb must be 8 or more: no Apple silicon Mac has less".to_owned()
                    );
                }
                matrix_options.mac_memory_gb.push(mac_memory_gb);
            }
            other => return Err(format!("unexpected small-mac-matrix argument: {other}")),
        }
    }
    if matrix_options.mac_memory_gb.is_empty() {
        matrix_options.mac_memory_gb = DEFAULT_MAC_MEMORY_GB.to_vec();
    }
    Ok(matrix_options)
}

/// Reads one chat answer: an `error` member is a failure named by its code,
/// anything else with choices is a reply.
pub fn parse_chat_outcome(chat_answer_text: &str, seconds_to_reply: f64) -> CellOutcome {
    let chat_answer: serde_json::Value =
        serde_json::from_str(chat_answer_text).unwrap_or(serde_json::Value::Null);
    if let Some(error_code) = chat_answer["error"]["code"].as_str() {
        return CellOutcome::Failed {
            error_code: error_code.to_owned(),
        };
    }
    if chat_answer["choices"].is_array() {
        return CellOutcome::Replied { seconds_to_reply };
    }
    CellOutcome::Failed {
        error_code: "no_answer".to_owned(),
    }
}

pub fn render_matrix(matrix_cells: &[MatrixCell]) -> String {
    let mut table_rows = Vec::with_capacity(matrix_cells.len());
    for matrix_cell in matrix_cells {
        let outcome_text = match &matrix_cell.outcome {
            CellOutcome::Replied { seconds_to_reply } => {
                format!("replied in {seconds_to_reply:.1} s")
            }
            CellOutcome::LimitRefused => "memory limit refused".to_owned(),
            CellOutcome::Failed { error_code } => format!("failed: {error_code}"),
        };
        table_rows.push(vec![
            matrix_cell.model_id.clone(),
            format!("{} GB", matrix_cell.mac_memory_gb),
            format!("{} GB", matrix_cell.memory_limit_gb),
            outcome_text,
        ]);
    }
    let mut matrix_text =
        String::from("SMALL-MAC MATRIX (simulated by memory limit on this Mac)\n");
    let mut model_column_width = "Model".len();
    for table_row in &table_rows {
        model_column_width = model_column_width.max(table_row[0].chars().count());
    }
    matrix_text.push_str(&format!(
        "  {:<model_column_width$}   {:>5}   {:>5}   Outcome\n",
        "Model", "Mac", "Limit"
    ));
    for table_row in &table_rows {
        matrix_text.push_str(&format!(
            "  {:<model_column_width$}   {:>5}   {:>5}   {}\n",
            table_row[0], table_row[1], table_row[2], table_row[3]
        ));
    }
    matrix_text.push_str(
        "  What this means: only the memory limit is simulated. A real small Mac also has less memory\n  for everything else and a slower chip, so a reply here is necessary, not sufficient; a\n  failure here is real. The time includes loading the model.\n",
    );
    matrix_text
}
