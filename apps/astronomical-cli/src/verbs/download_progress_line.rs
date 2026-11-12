//! The live model-download line `image` and `speak` show while the
//! lifecycle fetches their model: each report rewrites it in place on
//! standard error.

use std::io::Write;

use crate::model_lifecycle::ProgressSink;

pub(crate) struct DownloadProgressLine<'a> {
    stderr: &'a mut dyn Write,
    has_written: bool,
}

impl<'a> DownloadProgressLine<'a> {
    pub(crate) fn new(stderr: &'a mut dyn Write) -> Self {
        Self {
            stderr,
            has_written: false,
        }
    }

    /// Ends a line that was written, so the next message starts on its own.
    pub(crate) fn finish(self) {
        if self.has_written {
            let _ = writeln!(self.stderr);
        }
    }
}

impl ProgressSink for DownloadProgressLine<'_> {
    fn report(&mut self, progress_line: &str) {
        self.has_written = true;
        let _ = write!(self.stderr, "\r{progress_line}");
        let _ = self.stderr.flush();
    }
}
