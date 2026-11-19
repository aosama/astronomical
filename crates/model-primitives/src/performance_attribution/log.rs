use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

use super::PerformanceAttributionReport;

/// Append-only writer for bounded attribution reports.
#[derive(Debug)]
pub struct PerformanceAttributionLog {
    writer: Option<BufWriter<File>>,
    performance_attribution_log_path: Option<PathBuf>,
}

impl PerformanceAttributionLog {
    /// Creates an inert writer for code paths that do not request attribution.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            writer: None,
            performance_attribution_log_path: None,
        }
    }

    /// Opens the report file only when attribution is enabled.
    pub fn open(
        performance_attribution_log_path: &Path,
        performance_attribution_enabled: bool,
    ) -> io::Result<Self> {
        if !performance_attribution_enabled {
            return Ok(Self::disabled());
        }
        Ok(Self {
            writer: Some(Self::open_writer(performance_attribution_log_path)?),
            performance_attribution_log_path: Some(performance_attribution_log_path.to_path_buf()),
        })
    }

    /// Serializes and flushes one report, retrying a failed writer on the next report.
    pub fn record(
        &mut self,
        performance_attribution_report: &PerformanceAttributionReport,
    ) -> io::Result<()> {
        self.reopen_if_log_path_changed()?;
        let Some(performance_attribution_writer) = self.writer.as_mut() else {
            return Ok(());
        };
        let serialized_report = match serde_json::to_vec(performance_attribution_report) {
            Ok(serialized_report) => serialized_report,
            Err(serialization_error) => {
                self.writer = None;
                return Err(io::Error::other(serialization_error));
            }
        };
        let write_outcome = performance_attribution_writer
            .write_all(&serialized_report)
            .and_then(|()| performance_attribution_writer.write_all(b"\n"))
            .and_then(|()| performance_attribution_writer.flush());
        if write_outcome.is_err() {
            self.writer = None;
        }
        write_outcome
    }

    fn reopen_if_log_path_changed(&mut self) -> io::Result<()> {
        let Some(performance_attribution_log_path) = self.performance_attribution_log_path.as_ref()
        else {
            return Ok(());
        };
        let should_reopen = match self.writer.as_ref() {
            None => true,
            Some(performance_attribution_writer) => {
                let opened_file_metadata = performance_attribution_writer.get_ref().metadata()?;
                match fs::metadata(performance_attribution_log_path) {
                    Ok(current_path_metadata) => {
                        opened_file_metadata.dev() != current_path_metadata.dev()
                            || opened_file_metadata.ino() != current_path_metadata.ino()
                    }
                    Err(path_metadata_error)
                        if path_metadata_error.kind() == io::ErrorKind::NotFound =>
                    {
                        true
                    }
                    Err(path_metadata_error) => return Err(path_metadata_error),
                }
            }
        };
        if should_reopen {
            self.writer = Some(Self::open_writer(performance_attribution_log_path)?);
        }
        Ok(())
    }

    fn open_writer(performance_attribution_log_path: &Path) -> io::Result<BufWriter<File>> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(performance_attribution_log_path)
            .map(BufWriter::new)
    }
}
